"""QuantumSentinel — the one definition of each built-in strategy's signal.

The portfolio backtester, walk-forward validation and the event-driven
simulator all take their signals from here, so a strategy validated in one
engine is the same strategy in the others.

Timing contract: ``signal_series(close, strategy)[t]`` uses only
``close[:t + 1]`` — it is the decision available once bar ``t`` has closed.
Engines act on it no earlier than bar ``t + MIN_DECISION_DELAY_BARS``:
filling at ``close[t]`` would trade at the very price the decision was
computed from.

Signal values: > 0 buy (magnitude = strength), < 0 sell, 0 hold.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

MIN_DECISION_DELAY_BARS = 1


class StrategyType:
    MA_CROSSOVER = "ma_crossover"
    SBA_SIGNAL = "sba_signal"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"


@dataclass
class StrategyConfig:
    """Configuration for a built-in strategy."""
    strategy_type: str = StrategyType.MA_CROSSOVER
    fast_window: int = 20
    slow_window: int = 50
    # SBA-specific
    sba_buy_threshold: float = 0.15
    sba_sell_threshold: float = -0.15
    # Momentum
    momentum_lookback: int = 20
    momentum_entry: float = 0.05   # enter if momentum > 5%
    momentum_exit: float = -0.02   # exit if momentum < -2%
    # Mean reversion
    mr_lookback: int = 20
    mr_entry_zscore: float = -2.0  # buy when z < -2
    mr_exit_zscore: float = 0.0    # sell when z > 0


_SBA_MOMENTUM_LOOKBACK = 20   # signal_engine._momentum's default lookback


def warmup_bars(strategy: StrategyConfig) -> int:
    """Closes needed before the strategy can emit its first non-zero signal."""
    if strategy.strategy_type == StrategyType.MOMENTUM:
        return strategy.momentum_lookback + 1
    if strategy.strategy_type == StrategyType.MEAN_REVERSION:
        return strategy.mr_lookback + 1
    if strategy.strategy_type == StrategyType.SBA_SIGNAL:
        return _SBA_MOMENTUM_LOOKBACK + 1
    if strategy.strategy_type == StrategyType.MA_CROSSOVER:
        return max(strategy.fast_window, strategy.slow_window) + 1
    return 1


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """out[t] = mean(x[t-window+1 : t+1]); NaN until a full window exists."""
    out = np.full(len(x), np.nan)
    if window >= 1 and len(x) >= window:
        out[window - 1:] = sliding_window_view(x, window).mean(axis=1)
    return out


def ma_crossover_series(close: np.ndarray, fast: int, slow: int) -> np.ndarray:
    """+1 on the bar the fast SMA crosses above the slow SMA, -1 on the
    reverse cross, 0 otherwise. A cross compares bar t with bar t-1, so the
    first possible signal is at t = max(fast, slow)."""
    close = np.asarray(close, dtype=float).reshape(-1)
    out = np.zeros(len(close))
    first = max(fast, slow)
    if len(close) <= first:
        return out
    f, s = _rolling_mean(close, fast), _rolling_mean(close, slow)
    prev_f, prev_s, cur_f, cur_s = f[first - 1:-1], s[first - 1:-1], f[first:], s[first:]
    out[first:] = np.where((prev_f <= prev_s) & (cur_f > cur_s), 1.0,
                           np.where((prev_f >= prev_s) & (cur_f < cur_s), -1.0, 0.0))
    return out


def _momentum_series(close: np.ndarray, s: StrategyConfig) -> np.ndarray:
    out = np.zeros(len(close))
    lb = s.momentum_lookback
    if len(close) <= lb:
        return out
    mom = close[lb:] / close[:-lb] - 1.0
    out[lb:] = np.where(mom > s.momentum_entry, np.minimum(mom * 5, 1.0),
                        np.where(mom < s.momentum_exit, np.maximum(mom * 5, -1.0), 0.0))
    return out


def _mean_reversion_series(close: np.ndarray, s: StrategyConfig) -> np.ndarray:
    out = np.zeros(len(close))
    lb = s.mr_lookback
    if lb < 2 or len(close) <= lb:
        return out
    windows = sliding_window_view(close, lb)[1:]          # windows ending at t = lb .. n-1
    mu = windows.mean(axis=1)
    sigma = windows.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (close[lb:] - mu) / sigma
    strength = np.minimum(np.abs(z) / 3, 1.0)
    signal = np.where(z < s.mr_entry_zscore, strength,
                      np.where(z > s.mr_exit_zscore, -strength, 0.0))
    out[lb:] = np.where(sigma < 1e-9, 0.0, signal)
    return out


def _sba_series(close: np.ndarray, s: StrategyConfig, start: int) -> np.ndarray:
    from . import signal_engine  # heavy module; only needed for this strategy

    out = np.zeros(len(close))
    for t in range(max(start, _SBA_MOMENTUM_LOOKBACK), len(close)):
        spin = float(np.tanh(signal_engine.extract_features(close[:t + 1])["momentum"] * 5.0))
        if spin > s.sba_buy_threshold or spin < s.sba_sell_threshold:
            out[t] = spin
    return out


def signal_series(close: np.ndarray, strategy: StrategyConfig, start: int = 0) -> np.ndarray:
    """Signal for every bar of ``close``; element t depends only on close[:t+1].

    ``start`` lets an engine skip bars whose signal it will never read
    (only the SBA strategy is expensive enough for this to matter).
    """
    close = np.asarray(close, dtype=float).reshape(-1)
    kind = strategy.strategy_type
    if kind == StrategyType.MA_CROSSOVER:
        return ma_crossover_series(close, strategy.fast_window, strategy.slow_window)
    if kind == StrategyType.MOMENTUM:
        return _momentum_series(close, strategy)
    if kind == StrategyType.MEAN_REVERSION:
        return _mean_reversion_series(close, strategy)
    if kind == StrategyType.SBA_SIGNAL:
        return _sba_series(close, strategy, start)
    return np.zeros(len(close))


def latest_signal(close: np.ndarray, strategy: StrategyConfig) -> float:
    """The signal as of the last bar of ``close``."""
    close = np.asarray(close, dtype=float).reshape(-1)
    if not len(close):
        return 0.0
    return float(signal_series(close, strategy, start=len(close) - 1)[-1])
