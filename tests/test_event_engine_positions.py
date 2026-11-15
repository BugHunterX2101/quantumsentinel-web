"""The event engine treats a signal as a target position.

* A flat signal (``suggested_direction == 0``) closes the position.
* A signal in the direction already held keeps the position as it is.
* Any other signal trades to the sized target: it enters, flips, or (without
  shorting) only closes.
* Momentum decides once per bar, every fifth bar, whatever the universe size.
"""
from __future__ import annotations

import numpy as np
import pytest

from backend.services import event_simulator as es

QTY = 0.02  # the trade log rounds quantities to 2 decimals


@pytest.fixture
def run(monkeypatch):
    """run_event_backtest at zero cost, returning (result, portfolio)."""
    captured: list[es.Portfolio] = []

    class Capturing(es.Portfolio):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            captured.append(self)

    monkeypatch.setattr(es, "Portfolio", Capturing)

    def _run(prices: dict[str, np.ndarray], **kw):
        captured.clear()
        result = es.run_event_backtest(list(prices), prices, cost_model_name="zero_cost", **kw)
        return result, captured[0]

    return _run


def held(portfolio: es.Portfolio) -> dict[str, float]:
    return {t: q for t, q in portfolio.positions.items() if abs(q) > 1e-9}


def fills(portfolio: es.Portfolio, ticker: str) -> list[float]:
    return [f["qty"] for f in portfolio.trade_log if f["ticker"] == ticker]


# Around 100, a one-bar drop to 95 (z far below -2: long), then 30 bars at the
# mean (|z| < 0.5: a flat signal on every bar).
MEAN_REVERTING = np.array([100.5 if i % 2 else 99.5 for i in range(40)] + [95.0] + [100.0] * 30)


def test_a_flat_signal_closes_a_mean_reversion_position(run):
    _, pf = run({"A": MEAN_REVERTING}, strategy_name="mean_reversion",
                strategy_params={"window": 20, "n_std": 2.0, "fraction": 0.1})
    a = fills(pf, "A")
    assert len(a) == 2 and a[0] > 0
    assert a[1] == pytest.approx(-a[0], abs=QTY)
    assert held(pf) == {}


def test_a_flat_signal_closes_a_short_entered_from_flat(run):
    # The mirror image: a one-bar spike to 105 (z far above 2: short from
    # flat), then flat signals at the mean.
    spiking = 200.0 - MEAN_REVERTING
    _, pf = run({"A": spiking}, strategy_name="mean_reversion", allow_short=True,
                strategy_params={"window": 20, "n_std": 2.0, "fraction": 0.1})
    a = fills(pf, "A")
    assert len(a) == 2 and a[0] < 0
    assert a[1] == pytest.approx(-a[0], abs=QTY)
    assert held(pf) == {}


def _leader_then_laggard(T: int = 60) -> dict[str, np.ndarray]:
    t = np.arange(T, dtype=float)
    return {
        "A": np.where(t < 20, 100 * 1.03 ** t, 100 * 1.03 ** 19 * 0.999 ** (t - 19)),
        "B": np.where(t < 20, 100 * 1.001 ** t, 100 * 1.001 ** 19 * 1.02 ** (t - 19)),
        "C": np.where(t < 20, 100 * 1.0005 ** t, 100 * 1.0005 ** 19 * 1.015 ** (t - 19)),
        "D": 100 * 1.002 ** t, "E": 100 * 0.998 ** t, "F": 100 * 0.997 ** t,
    }


def test_momentum_enters_once_and_closes_a_name_that_leaves_the_top(run):
    # A leads for 20 bars (four rebalances), then B overtakes it.
    _, pf = run(_leader_then_laggard(), strategy_name="momentum",
                strategy_params={"lookback": 5, "n_long": 1, "n_short": 0, "fraction": 0.2})
    a = fills(pf, "A")
    assert len(a) == 2 and a[0] > 0               # one entry, no pyramiding
    assert a[1] == pytest.approx(-a[0], abs=QTY)  # then closed
    assert set(held(pf)) == {"B"}
    assert len(fills(pf, "B")) == 1


@pytest.mark.parametrize("n_symbols", [1, 2, 5, 6, 7])
def test_momentum_rebalances_every_fifth_bar_whatever_the_universe(n_symbols):
    strategy = es.MomentumStrategy(lookback=3, n_long=1, n_short=0)
    symbols = [f"S{i}" for i in range(n_symbols)]
    history = {s: [100.0 + i, 101.0 + i, 102.0 + i, 103.0 + i] for i, s in enumerate(symbols)}
    fired = [bar for bar in range(15) for s in symbols
             if strategy.on_market(es.MarketEvent(timestamp=bar, ticker=s), None, history)]
    assert fired == [4, 9, 14]


def _cross_up_then_down() -> np.ndarray:
    # Flat, rally (fast crosses above slow), then a slide (reverse cross).
    return np.array([100.0] * 20 + [100.0 + 2 * i for i in range(1, 16)]
                    + [130.0 - 3 * i for i in range(1, 26)])


def test_a_reverse_cross_closes_exactly_when_long_only(run):
    _, pf = run({"A": _cross_up_then_down()}, strategy_name="ma_crossover",
                strategy_params={"fast": 3, "slow": 8, "fraction": 0.1})
    a = fills(pf, "A")
    assert len(a) == 2 and a[0] > 0
    assert a[1] == pytest.approx(-a[0], abs=QTY)
    assert held(pf) == {}


def test_a_reverse_cross_flips_to_the_sized_short_when_shorting(run):
    _, pf = run({"A": _cross_up_then_down()}, strategy_name="ma_crossover", allow_short=True,
                strategy_params={"fast": 3, "slow": 8, "fraction": 0.1})
    a = fills(pf, "A")
    assert len(a) == 2 and a[0] > 0 and a[1] < 0
    short = held(pf)["A"]
    assert short < 0
    assert a[1] == pytest.approx(short - a[0], abs=QTY)  # closed the long, opened the short
    # The short is sized afresh (a fraction of equity at the signal bar's
    # price), not a mirror of the long.
    assert abs(abs(short) - a[0]) > 1.0


def test_two_signals_for_one_name_in_one_bar_do_not_double(run, monkeypatch):
    class Twice(es.BaseStrategy):
        def __init__(self, **_):
            pass

        def on_market(self, event, portfolio, history):
            if event.timestamp != 5:
                return []
            signal = es.SignalEvent(timestamp=5, ticker=event.ticker,
                                    signal_strength=1.0, suggested_direction=1)
            return [signal, signal]

    monkeypatch.setattr(es, "MACrossoverStrategy", Twice)
    _, pf = run({"A": np.full(20, 100.0)}, strategy_name="ma_crossover",
                strategy_params={"fraction": 0.1})
    assert fills(pf, "A") == [pytest.approx(100.0)]   # 10% of 100,000 at 100


def test_an_opposite_signal_the_sizer_cannot_size_still_closes(run, monkeypatch):
    # Long at bar 25 (vol-target sized on a noisy history); then the price is
    # constant, so at bar 55 the last 21 returns have no volatility and the
    # short cannot be sized. The long must still be closed, and nothing opened.
    class LongThenShort(es.BaseStrategy):
        def __init__(self, **_):
            pass

        def on_market(self, event, portfolio, history):
            direction = {25: 1, 55: -1}.get(event.timestamp)
            if direction is None:
                return []
            return [es.SignalEvent(timestamp=event.timestamp, ticker=event.ticker,
                                   signal_strength=1.0, suggested_direction=direction)]

    monkeypatch.setattr(es, "MACrossoverStrategy", LongThenShort)
    noisy = [100.0 + (3.0 if i % 2 else -3.0) for i in range(30)]
    _, pf = run({"A": np.array(noisy + [100.0] * 30)}, strategy_name="ma_crossover",
                allow_short=True, sizing_method="volatility_target")
    a = fills(pf, "A")
    assert len(a) == 2 and a[0] > 0
    assert a[1] == pytest.approx(-a[0], abs=QTY)
    assert held(pf) == {}


def test_a_signal_repeated_every_bar_enters_once(run, monkeypatch):
    class Every(es.BaseStrategy):
        def __init__(self, **_):
            pass

        def on_market(self, event, portfolio, history):
            return [es.SignalEvent(timestamp=event.timestamp, ticker=event.ticker,
                                   signal_strength=1.0, suggested_direction=1)]

    monkeypatch.setattr(es, "MACrossoverStrategy", Every)
    _, pf = run({"A": np.full(20, 100.0)}, strategy_name="ma_crossover",
                strategy_params={"fraction": 0.1})
    assert fills(pf, "A") == [pytest.approx(100.0)]
