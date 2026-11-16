"""The validation policy (services/validation_policy.py) and the risk
measurements the event engine reports for its risk gate."""
import hashlib
import math

import numpy as np
import pytest

from backend.services import strategy_spec, validation_policy
from backend.services.event_simulator import run_event_backtest
from backend.services.strategy_signals import ma_crossover_series
from backend.services.validation_policy import POLICY_V1, ValidationPolicy, policy_id, risk_gate
from test_strategy_spec import prices, spec

# --------------------------------------------------------------------------
# The policy and its id
# --------------------------------------------------------------------------

POLICY_V1_CANONICAL = (
    b'{"haircut_method":"bhy","max_declared_drawdown_bps":2500,"max_declared_position_bps":2500,'
    b'"max_pbo_bps":500,"min_dsr_bps":9500,"min_haircut_sharpe_annual_milli":500,"min_t_statistic_milli":3000,'
    b'"name":"qs-validation-policy-v1","pbo_min_block_sessions":21,"pbo_min_strategies":10,'
    b'"pbo_splits":[16,12,8],"sessions_per_year":[["CRYPTO_SPOT_EOD",365],["EQUITY_EOD",252],'
    b'["FUND_NAV_EOD",252],["FUTURES_CONTINUOUS_EOD",252],["FX_SPOT_EOD",252],["INDEX_EOD",252]]}')
POLICY_V1_ID = "qs-vp-sha256-v1:55369c1a3cdedb3d4eb1b5505063ded09b7fb085d3cd53a89e0c0e1321cd5bb0"


def test_policy_v1_has_its_golden_id():
    """Pinned, and recomputed here from the canonical JSON written out by
    hand, so a verdict naming the id can be checked by another
    implementation."""
    assert POLICY_V1_ID == f"qs-vp-sha256-v1:{hashlib.sha256(POLICY_V1_CANONICAL).hexdigest()}"
    assert validation_policy.policy_canonical(POLICY_V1) == POLICY_V1_CANONICAL
    assert policy_id(POLICY_V1) == POLICY_V1_ID
    assert validation_policy.POLICY_ID.fullmatch(POLICY_V1_ID)


def test_policy_v1_holds_the_thresholds_chosen_for_it():
    """The handoff table's values (decided 2026-11-15)."""
    p = POLICY_V1
    assert (p.max_declared_drawdown_bps, p.max_declared_position_bps) == (2_500, 2_500)
    assert (p.min_dsr_bps, p.max_pbo_bps, p.pbo_min_strategies) == (9_500, 500, 10)
    assert (p.pbo_splits, p.pbo_min_block_sessions) == ((16, 12, 8), 21)
    assert (p.haircut_method, p.min_haircut_sharpe_annual_milli, p.min_t_statistic_milli) == ("bhy", 500, 3_000)
    assert dict(p.sessions_per_year) == {"CRYPTO_SPOT_EOD": 365, "EQUITY_EOD": 252, "FUND_NAV_EOD": 252,
                                         "FUTURES_CONTINUOUS_EOD": 252, "FX_SPOT_EOD": 252, "INDEX_EOD": 252}


CHANGED = {"name": "other", "max_declared_drawdown_bps": 2_499, "max_declared_position_bps": 2_499,
           "min_dsr_bps": 9_501, "max_pbo_bps": 499, "pbo_min_strategies": 11, "pbo_splits": (16, 12),
           "pbo_min_block_sessions": 20, "haircut_method": "holm", "min_haircut_sharpe_annual_milli": 499,
           "min_t_statistic_milli": 2_999, "sessions_per_year": (("EQUITY_EOD", 252),)}


def test_every_field_is_listed_here():
    assert set(CHANGED) == set(POLICY_V1.__dict__)


@pytest.mark.parametrize("field", sorted(CHANGED))
def test_every_field_is_part_of_the_id(field):
    other = ValidationPolicy(**{**POLICY_V1.__dict__, field: CHANGED[field]})
    assert policy_id(other) != POLICY_V1_ID


@pytest.mark.parametrize("changes", [
    {"name": ""}, {"name": 1},
    {"max_declared_drawdown_bps": 0}, {"max_declared_drawdown_bps": 10_001},
    {"max_declared_drawdown_bps": 2_500.0}, {"max_declared_drawdown_bps": True},
    {"max_declared_position_bps": -1}, {"max_declared_position_bps": "2500"},
    {"min_dsr_bps": 0}, {"min_dsr_bps": 10_001}, {"min_dsr_bps": 0.95},
    {"max_pbo_bps": -1}, {"max_pbo_bps": 10_001},
    {"pbo_min_strategies": 1}, {"pbo_min_strategies": 1_001}, {"pbo_min_strategies": True},
    {"pbo_min_block_sessions": 1}, {"pbo_min_block_sessions": 10_001},
    {"min_haircut_sharpe_annual_milli": -1}, {"min_haircut_sharpe_annual_milli": 0.5},
    {"min_t_statistic_milli": -1}, {"min_t_statistic_milli": 100_001},
    {"pbo_splits": ()}, {"pbo_splits": [16, 12, 8]}, {"pbo_splits": (16, 13)}, {"pbo_splits": (8, 16)},
    {"pbo_splits": (16, 16)}, {"pbo_splits": (0,)}, {"pbo_splits": (16.0,)}, {"pbo_splits": (True,)},
    {"haircut_method": "bh"}, {"haircut_method": "BHY"},
    {"sessions_per_year": ()}, {"sessions_per_year": [("EQUITY_EOD", 252)]},
    {"sessions_per_year": (("EQUITY_EOD", 0),)}, {"sessions_per_year": (("EQUITY_EOD", 367),)},
    {"sessions_per_year": (("EQUITY_EOD", 252.0),)}, {"sessions_per_year": (("", 252),)},
    {"sessions_per_year": (("EQUITY_EOD", 252), ("CRYPTO_SPOT_EOD", 365))},
    {"sessions_per_year": (("EQUITY_EOD", 252), ("EQUITY_EOD", 252))},
    {"sessions_per_year": (("EQUITY_EOD", 252, 1),)},
])
def test_a_policy_refuses_thresholds_of_the_wrong_kind(changes):
    with pytest.raises(ValueError):
        ValidationPolicy(**{**POLICY_V1.__dict__, **changes})


def test_a_policy_is_frozen():
    with pytest.raises(AttributeError):
        POLICY_V1.max_declared_drawdown_bps = 10_000  # type: ignore[misc]


# --------------------------------------------------------------------------
# The engine's risk measurements
# --------------------------------------------------------------------------

def test_without_the_flag_the_engine_reports_exactly_what_it_did_before():
    compiled = strategy_spec.compile_spec(strategy_spec.parse(spec(**{"signal.fast": 5, "signal.slow": 20})))
    data = prices(compiled.tickers, bars=400)
    plain = compiled.run(data, 100_000.0, full_returns=True)
    measured = compiled.run(data, 100_000.0, full_returns=True, risk_metrics=True)
    assert "risk" not in plain
    risk = measured.pop("risk")
    assert set(risk) == {"max_drawdown", "max_position_share", "equity_exhausted"}
    np.testing.assert_array_equal(measured.pop("returns"), plain.pop("returns"))
    assert measured == plain


def _reference(closes, fast, slow, fraction, capital):
    """Equity, the largest position's share of it and the gross exposure's
    share of it at each bar, worked out independently of the engine for a
    long-only, zero-cost crossover on each ticker: a signal at bar s fills at
    bar s + 1's close; a long entry is sized at ``fraction`` of the equity
    marked at bar s - 1 (the last mark when bar s's signals are sized) at
    bar s's price, and a cross down closes it. Shares are taken both before
    and after each bar's fills."""
    tickers = sorted(closes)
    signals = {t: ma_crossover_series(closes[t], fast, slow) for t in tickers}
    cash, held, staged = capital, dict.fromkeys(tickers, 0.0), {}
    last_mark = capital
    equity, largest, gross = [], [], []
    for bar in range(len(closes[tickers[0]])):
        price = {t: closes[t][bar] for t in tickers}
        before = cash + sum(held[t] * price[t] for t in tickers)
        largest.append(max(abs(held[t]) * price[t] for t in tickers) / before)
        for t, qty in staged.items():
            cash -= qty * price[t]
            held[t] += qty
        staged = {}
        for t in tickers:
            if signals[t][bar] > 0 and held[t] == 0:
                staged[t] = fraction * last_mark / price[t]
            elif signals[t][bar] < 0 and held[t] > 0:
                staged[t] = -held[t]
        values = [abs(held[t]) * price[t] for t in tickers]
        last_mark = cash + sum(held[t] * price[t] for t in tickers)
        equity.append(last_mark)
        largest.append(max(values) / last_mark)
        gross.append(sum(values) / last_mark)
    return np.array(equity), np.array(largest), np.array(gross)


GAP_UP_EXIT = [100.0] * 8 + [110, 120, 130, 125, 90, 160, 150]
CASES = {  # label: (closes, fraction)
    "one ticker": ({"X": [100.0] * 8 + [110, 120, 130, 125, 90, 80, 85, 95, 100, 70]}, 0.5),
    "one ticker, re-entered": (
        {"X": [100.0] * 8 + [104, 112, 121, 133, 128, 140, 101, 97, 99, 108, 117, 126, 90, 70, 75]}, 0.5),
    "one ticker, a new high after the deepest fall": (
        {"X": [100.0] * 8 + [110, 120, 130, 100, 85, 95, 110, 130, 150, 170, 190, 210, 230, 250]}, 0.5),
    "one ticker, sold on a gap up": ({"X": GAP_UP_EXIT}, 0.3),
    "one ticker, falling from the bar it is bought": ({"X": [100.0] * 8 + [110, 120, 115, 105, 95, 90, 85]}, 0.5),
    "two tickers held together": ({
        "X": [100.0] * 8 + [110, 120, 130, 125, 90, 80, 85, 95, 100, 70],
        "Y": [100.0] * 8 + [105, 115, 140, 150, 145, 130, 100, 90, 95, 99]}, 0.3),
}


@pytest.mark.parametrize("label", CASES)
def test_the_measurements_match_an_independent_reference(label):
    closes = {t: np.array(c, dtype=float) for t, c in CASES[label][0].items()}
    fraction = CASES[label][1]
    result = run_event_backtest(sorted(closes), closes, "ma_crossover", {"fast": 2, "slow": 5, "fraction": fraction},
                                initial_capital=100_000.0, cost_model_name="zero_cost", risk_metrics=True)
    equity, largest, gross = _reference(closes, 2, 5, fraction, 100_000.0)
    assert result["n_trades"] >= 2 * len(closes)  # every ticker entered and left
    expected_drawdown = float(np.max(1 - equity / np.maximum.accumulate(equity)))
    assert expected_drawdown > 0
    if "falling from the bar" in label:  # largest the moment it is bought, at 120
        qty = 0.5 * 100_000 / 110
        assert result["risk"]["max_position_share"] == pytest.approx(qty * 120 / 100_000, rel=1e-12)
    if "new high" in label:
        # A later, higher peak: measuring from it would overstate the fall.
        assert equity.argmax() > np.argmax(1 - equity / np.maximum.accumulate(equity))
        assert float(np.max(1 - equity / equity.max())) > expected_drawdown + 0.05
    assert result["risk"]["max_drawdown"] == pytest.approx(expected_drawdown, rel=1e-12)
    assert result["risk"]["max_position_share"] == pytest.approx(float(largest.max()), rel=1e-12)
    assert result["risk"]["equity_exhausted"] is False
    if len(closes) > 1:  # the largest single position, not the gross exposure
        assert largest.max() < gross.max() - 0.05
    # Unrounded, unlike the engine's displayed max_drawdown.
    assert result["max_drawdown"] == round(-expected_drawdown, 4)


def test_the_first_case_by_hand():
    """Entry signalled at bar 8 (110), filled at bar 9 (120) with
    0.5 * 100,000 / 110 shares; at 130 the position is 59,090.9 of
    104,545.5 equity."""
    close = np.array([100.0] * 8 + [110, 120, 130, 125, 90, 80, 85, 95, 100, 70])
    result = run_event_backtest(["X"], {"X": close}, "ma_crossover", {"fast": 2, "slow": 5, "fraction": 0.5},
                                initial_capital=100_000.0, cost_model_name="zero_cost", risk_metrics=True)
    qty = 0.5 * 100_000 / 110
    cash = 100_000 - qty * 120
    assert result["risk"]["max_position_share"] == pytest.approx(qty * 130 / (cash + qty * 130), rel=1e-12)


def test_a_position_is_measured_at_the_price_it_is_sold_at():
    """Orders fill at their bar's price: the cross down signalled at 90 sells
    at the gap up to 160, so for that moment the position, 0.3 * 100,000 /
    110 shares, is 39.3% of equity, more than at any bar's close (34.5%,
    at 130)."""
    close = np.array(GAP_UP_EXIT)
    result = run_event_backtest(["X"], {"X": close}, "ma_crossover", {"fast": 2, "slow": 5, "fraction": 0.3},
                                initial_capital=100_000.0, cost_model_name="zero_cost", risk_metrics=True)
    qty = 0.3 * 100_000 / 110
    cash = 100_000 - qty * 120
    assert result["n_trades"] == 2  # bought at 120, sold at 160
    assert result["risk"]["max_position_share"] == pytest.approx(qty * 160 / (cash + qty * 160), rel=1e-12)
    assert result["risk"]["max_position_share"] > qty * 130 / (cash + qty * 130) + 0.04


def test_a_short_that_loses_everything_is_reported_as_exhausting_equity():
    """A full-size short into a steady climb: equity crosses zero once the
    price doubles past the entry. Shares of non-positive equity are not
    measured; the drawdown keeps counting past 100%."""
    close = 100.0 + 5.0 * np.arange(120)
    result = run_event_backtest(["X"], {"X": close}, "mean_reversion",
                                {"window": 20, "n_std": 1.0, "fraction": 1.0, "leverage_limit": 5.0},
                                initial_capital=100_000.0, cost_model_name="zero_cost", allow_short=True,
                                risk_metrics=True)
    risk = result["risk"]
    assert risk["equity_exhausted"] is True
    assert risk["max_drawdown"] > 1.0
    assert math.isfinite(risk["max_position_share"]) and risk["max_position_share"] > 1.0


# --------------------------------------------------------------------------
# The risk gate
# --------------------------------------------------------------------------

LIMITS = strategy_spec.RiskLimits(max_drawdown_bps=2_000, max_position_bps=2_500)
CLEAN = {"max_drawdown": 0.1, "max_position_share": 0.2, "equity_exhausted": False}


def test_a_backtest_within_its_declared_limits_passes():
    verdict = risk_gate(LIMITS, CLEAN)
    assert verdict["passed"] is True and verdict["gate"] == "risk_limits"
    assert verdict["policy"] == POLICY_V1_ID
    assert all(check["passed"] for check in verdict["checks"].values())
    assert verdict["checks"]["drawdown_within_declared"]["value_bps"] == pytest.approx(1_000)


def test_a_measurement_equal_to_its_limit_passes_and_just_above_fails():
    at = {**CLEAN, "max_drawdown": 0.2, "max_position_share": 0.25}
    assert risk_gate(LIMITS, at)["passed"] is True
    for key in ("max_drawdown", "max_position_share"):
        over = {**at, key: at[key] + 1e-9}
        assert risk_gate(LIMITS, over)["passed"] is False


@pytest.mark.parametrize("measurement, failing", [
    ({"max_drawdown": 0.21}, "drawdown_within_declared"),
    ({"max_position_share": 0.26}, "position_within_declared"),
    ({"equity_exhausted": True}, "equity_never_exhausted"),
])
def test_each_breach_alone_fails_the_gate(measurement, failing):
    verdict = risk_gate(LIMITS, {**CLEAN, **measurement})
    assert verdict["passed"] is False
    assert [name for name, check in verdict["checks"].items() if not check["passed"]] == [failing]


def test_a_declaration_at_the_policys_cap_passes():
    at_cap = strategy_spec.RiskLimits(max_drawdown_bps=2_500, max_position_bps=2_500)
    assert risk_gate(at_cap, CLEAN)["passed"] is True


@pytest.mark.parametrize("limits, failing", [
    (strategy_spec.RiskLimits(max_drawdown_bps=2_501, max_position_bps=2_500), "declared_drawdown_within_policy"),
    (strategy_spec.RiskLimits(max_drawdown_bps=2_000, max_position_bps=10_000), "declared_position_within_policy"),
])
def test_a_declaration_beyond_the_policys_cap_fails_however_small_the_risk_taken(limits, failing):
    verdict = risk_gate(limits, {"max_drawdown": 0.0, "max_position_share": 0.0, "equity_exhausted": False})
    assert verdict["passed"] is False
    assert [name for name, check in verdict["checks"].items() if not check["passed"]] == [failing]


_UNMEASURED = {"drawdown_within_declared", "position_within_declared", "equity_never_exhausted"}


@pytest.mark.parametrize("risk, failing", [
    (None, _UNMEASURED),
    ({}, _UNMEASURED),
    ([0.1, 0.2, False], _UNMEASURED),
    ({**CLEAN, "max_drawdown": math.nan}, {"drawdown_within_declared"}),
    ({**CLEAN, "max_drawdown": math.inf}, {"drawdown_within_declared"}),
    ({**CLEAN, "max_drawdown": -0.1}, {"drawdown_within_declared"}),
    ({**CLEAN, "max_drawdown": "0.1"}, {"drawdown_within_declared"}),
    ({**CLEAN, "max_drawdown": None}, {"drawdown_within_declared"}),
    ({**CLEAN, "max_position_share": math.nan}, {"position_within_declared"}),
    ({**CLEAN, "max_position_share": False}, {"position_within_declared"}),
    ({k: v for k, v in CLEAN.items() if k != "max_position_share"}, {"position_within_declared"}),
    ({**CLEAN, "equity_exhausted": 0}, {"equity_never_exhausted"}),
    ({**CLEAN, "equity_exhausted": "false"}, {"equity_never_exhausted"}),
    ({k: v for k, v in CLEAN.items() if k != "equity_exhausted"}, {"equity_never_exhausted"}),
])
def test_a_measurement_that_is_missing_or_not_a_share_fails_closed(risk, failing):
    verdict = risk_gate(LIMITS, risk)
    assert verdict["passed"] is False
    assert {name for name, check in verdict["checks"].items() if not check["passed"]} == failing
    for name in failing:
        assert "not measured" in verdict["checks"][name]["description"]


def test_the_gate_uses_the_policy_it_is_given():
    loose = ValidationPolicy(**{**POLICY_V1.__dict__, "name": "loose", "max_declared_drawdown_bps": 10_000,
                                "max_declared_position_bps": 10_000})
    limits = strategy_spec.RiskLimits(max_drawdown_bps=5_000, max_position_bps=5_000)
    assert risk_gate(limits, CLEAN)["passed"] is False
    verdict = risk_gate(limits, CLEAN, loose)
    assert verdict["passed"] is True and verdict["policy"] == policy_id(loose)
