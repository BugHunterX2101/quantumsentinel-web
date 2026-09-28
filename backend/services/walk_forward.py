"""QuantumSentinel — Walk-Forward Validation Engine.

Implements rolling-window and expanding-window walk-forward analysis
to address overfitting and evaluate out-of-sample strategy performance.

Workflow:
  1. Split data into train/test folds
  2. For each fold: optimize/fit on train, evaluate on test
  3. Aggregate all out-of-sample results
  4. Report parameter stability across windows
"""
from __future__ import annotations

import dataclasses
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import yfinance as yf

from . import historical_data
from .backtest_service import (
    BacktestConfig, BacktestEngine, StrategyConfig, StrategyType,
    _sharpe, _sortino, _max_drawdown, _var_cvar, _calmar,
    _omega_ratio, _downside_deviation, _compute_win_rate,
)
from .execution_model import ExecutionConfig, PositionSizer, SizingMethod, retail_config
from .strategy_signals import warmup_bars

log = logging.getLogger(__name__)


class WindowType(str, Enum):
    ROLLING = "rolling"
    EXPANDING = "expanding"


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""
    assets: list[str] = field(default_factory=lambda: ["AAPL"])
    window_type: str = WindowType.ROLLING
    train_years: int = 2
    test_years: int = 1
    step_years: int = 1       # how far to slide each fold
    total_years: int = 5      # total data period
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    execution: ExecutionConfig = field(default_factory=retail_config)
    benchmark: str = "SPY"
    # Parameter search ranges for stability analysis
    fast_window_range: list[int] = field(default_factory=lambda: [10, 15, 20, 25, 30])
    slow_window_range: list[int] = field(default_factory=lambda: [40, 50, 60, 75, 100])
    optimize_parameters: bool = True
    # None = equal-weight sleeves that together invest the whole account.
    sizer: PositionSizer | None = None


@dataclass
class FoldResult:
    """Result for a single walk-forward fold."""
    fold_index: int
    train_start: int      # bar index
    train_end: int
    test_start: int
    test_end: int
    # Train metrics
    train_sharpe: float = 0.0
    train_return: float = 0.0
    train_max_dd: float = 0.0
    # Test (OOS) metrics
    oos_sharpe: float = 0.0
    oos_sortino: float = 0.0
    oos_return: float = 0.0
    oos_max_dd: float = 0.0
    oos_calmar: float = 0.0
    oos_var_95: float = 0.0
    oos_cvar_95: float = 0.0
    oos_win_rate: float = 0.0
    oos_total_trades: int = 0
    oos_turnover: float = 0.0
    # Selected parameters
    best_fast_window: int = 20
    best_slow_window: int = 50
    # Equity curve
    oos_equity: list[float] = field(default_factory=list)
    oos_returns: list[float] = field(default_factory=list)
    # Calendar dates of the windows ("first..last"), when known
    train_dates: str = ""
    test_dates: str = ""

    def to_dict(self) -> dict:
        return {
            "fold": self.fold_index,
            "train_period": f"bar_{self.train_start}_to_{self.train_end}",
            "test_period": f"bar_{self.test_start}_to_{self.test_end}",
            "train_dates": self.train_dates,
            "test_dates": self.test_dates,
            "train_sharpe": round(self.train_sharpe, 3),
            "train_return": round(self.train_return, 4),
            "train_max_dd": round(self.train_max_dd, 4),
            "oos_sharpe": round(self.oos_sharpe, 3),
            "oos_sortino": round(self.oos_sortino, 3),
            "oos_return": round(self.oos_return, 4),
            "oos_max_dd": round(self.oos_max_dd, 4),
            "oos_calmar": round(self.oos_calmar, 3),
            "oos_var_95": round(self.oos_var_95, 4),
            "oos_cvar_95": round(self.oos_cvar_95, 4),
            "oos_win_rate": round(self.oos_win_rate, 3),
            "oos_total_trades": self.oos_total_trades,
            "best_fast_window": self.best_fast_window,
            "best_slow_window": self.best_slow_window,
        }


def _windowed_vol(close: np.ndarray, i: int, window: int = 22) -> float:
    """Trailing return volatility over up to `window` bars ending at close[i-1].

    Slices a single price window and derives both the numerator (price
    diffs) and denominator (prior prices) from it via ``[:-1]`` so their
    lengths always match by construction, instead of computing the two
    slices independently (which silently breaks alignment near array
    boundaries — a shape mismatch that previously raised at runtime).
    """
    prices = close[max(0, i - window):i]
    if len(prices) <= 2:
        return 0.02
    rets = np.diff(prices) / np.maximum(prices[:-1], 1e-9)
    return float(np.std(rets))


_WF_CAPITAL = 100_000.0


def _fully_invested_sizer() -> PositionSizer:
    """Equal-weight sleeves that together invest the whole account, the
    walk-forward's long-standing convention (it used to put 95% of capital
    into the single asset it evaluated)."""
    return PositionSizer(method=SizingMethod.EQUAL_WEIGHT, max_position_pct=1.0, max_leverage=1.0)


def _asset_data_from_closes(close_data: dict) -> dict:
    """Engine input from bare close arrays (no high/low/volume available)."""
    out = {}
    for ticker, close in close_data.items():
        close = np.asarray(close, dtype=float).reshape(-1)
        out[ticker] = {"close": close, "high": close, "low": close,
                       "volume": np.full(len(close), historical_data.DEFAULT_VOLUME)}
    return out


def _empty_metrics() -> dict:
    return {"sharpe": 0.0, "sortino": 0.0, "total_return": 0.0, "max_dd": 0.0, "calmar": 0.0,
            "var_95": 0.0, "cvar_95": 0.0, "win_rate": 0.0, "n_trades": 0,
            "equity": [_WF_CAPITAL], "returns": []}


class WalkForwardEngine:
    """Walk-forward validation engine.

    Every fold (each in-sample parameter trial and the out-of-sample test) is
    run by BacktestEngine on the date-aligned panel, so walk-forward
    validates exactly the strategy, costs, fill timing and multi-asset
    portfolio the backtester simulates. The strategy is the moving-average
    crossover, whose windows are the parameters searched.
    """

    def __init__(self, config: WalkForwardConfig):
        self.config = config
        # Long-only, as walk-forward has always been: a sell signal closes the
        # position rather than reversing into a short.
        self._execution = dataclasses.replace(
            config.execution, sizer=config.sizer or _fully_invested_sizer(), allow_short_selling=False)

    def _parameter_grid(self) -> list[tuple[int, int]]:
        cfg = self.config
        if not cfg.optimize_parameters:
            return [(cfg.strategy.fast_window, cfg.strategy.slow_window)]
        return [(fw, sw) for fw in cfg.fast_window_range for sw in cfg.slow_window_range if sw > fw]

    def run(self) -> dict:
        """Execute walk-forward validation."""
        t0 = time.perf_counter()
        cfg = self.config

        # ── Fetch all data upfront, aligned on one calendar ──
        period_map = {3: "3y", 4: "4y", 5: "5y", 6: "6y", 7: "7y",
                      8: "8y", 10: "10y"}
        yf_period = period_map.get(cfg.total_years, f"{cfg.total_years}y")
        try:
            data = yf.download(
                list(dict.fromkeys(cfg.assets)), period=yf_period, interval="1d",
                progress=False, auto_adjust=True,
            )
        except Exception as exc:
            raise ValueError(f"Data download failed: {exc}")

        if data is None or data.empty:
            raise ValueError("Empty data returned")

        panel = historical_data.aligned_panel(data, cfg.assets, min_rows=101)
        valid_assets = panel.tickers
        if not valid_assets:
            raise ValueError("No valid assets found")
        asset_data = panel.asset_data()
        dates = panel.date_strings()
        n_bars = len(panel)

        bars_per_year = 252
        train_bars = cfg.train_years * bars_per_year
        test_bars = cfg.test_years * bars_per_year
        step_bars = cfg.step_years * bars_per_year

        # ── Generate folds ──
        folds = []
        fold_idx = 0

        if cfg.window_type == WindowType.ROLLING:
            start = 0
            while start + train_bars + test_bars <= n_bars:
                folds.append({
                    "fold": fold_idx,
                    "train_start": start,
                    "train_end": start + train_bars,
                    "test_start": start + train_bars,
                    "test_end": min(start + train_bars + test_bars, n_bars),
                })
                fold_idx += 1
                start += step_bars
        else:  # EXPANDING
            start = 0
            test_start = train_bars
            while test_start + test_bars <= n_bars:
                folds.append({
                    "fold": fold_idx,
                    "train_start": start,  # always 0 for expanding
                    "train_end": test_start,
                    "test_start": test_start,
                    "test_end": min(test_start + test_bars, n_bars),
                })
                fold_idx += 1
                test_start += step_bars

        if not folds:
            raise ValueError(
                f"Not enough data for walk-forward: need {train_bars + test_bars} "
                f"bars, have {n_bars}"
            )

        # ── Run each fold ──
        fold_results: list[FoldResult] = []
        all_oos_returns = []

        for fold in folds:
            result = self._run_fold(asset_data, valid_assets, fold, cfg, dates)
            fold_results.append(result)
            all_oos_returns.extend(result.oos_returns)

        # ── Aggregate OOS results ──
        all_oos = np.array(all_oos_returns) if all_oos_returns else np.array([0.0])
        # Build aggregated equity from OOS returns.
        agg_equity = [_WF_CAPITAL]
        for r in all_oos:
            agg_equity.append(agg_equity[-1] * (1 + r))

        agg_sharpe = _sharpe(all_oos)
        agg_sortino = _sortino(all_oos)
        agg_return = float(agg_equity[-1] / agg_equity[0] - 1) if agg_equity[0] > 0 else 0
        agg_max_dd = _max_drawdown(agg_equity)
        agg_var95, agg_cvar95 = _var_cvar(all_oos, 0.05)
        agg_calmar = _calmar(all_oos, agg_max_dd)

        # ── Parameter stability ──
        param_stability = {
            "fast_windows": [f.best_fast_window for f in fold_results],
            "slow_windows": [f.best_slow_window for f in fold_results],
            "fast_std": round(float(np.std([f.best_fast_window for f in fold_results])), 2),
            "slow_std": round(float(np.std([f.best_slow_window for f in fold_results])), 2),
            "parameters_stable": float(np.std([f.best_fast_window for f in fold_results])) < 5,
        }

        # ── Overfitting detection ──
        train_sharpes = [f.train_sharpe for f in fold_results]
        oos_sharpes = [f.oos_sharpe for f in fold_results]
        avg_train_sharpe = float(np.mean(train_sharpes)) if train_sharpes else 0
        avg_oos_sharpe = float(np.mean(oos_sharpes)) if oos_sharpes else 0
        sharpe_decay = avg_train_sharpe - avg_oos_sharpe
        overfitting_score = sharpe_decay / max(abs(avg_train_sharpe), 1e-9)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Subsample equity curve
        max_points = 200
        step = max(1, len(agg_equity) // max_points)
        grid = self._parameter_grid()

        return {
            "window_type": cfg.window_type,
            "n_folds": len(fold_results),
            "train_years": cfg.train_years,
            "test_years": cfg.test_years,
            "assets": valid_assets,
            "data_start": dates[0],
            "data_end": dates[-1],
            "execution_delay_bars": self._execution.execution_delay_bars,
            # Distinct parameter sets evaluated: the trial count a Deflated
            # Sharpe test of the selected configuration must account for.
            "n_trials": len(grid),
            "parameter_grid": [{"fast_window": fw, "slow_window": sw} for fw, sw in grid],
            # Per-fold results
            "folds": [f.to_dict() for f in fold_results],
            # Aggregated OOS metrics
            "aggregated_oos": {
                "sharpe": round(agg_sharpe, 3),
                "sortino": round(agg_sortino, 3),
                "total_return": round(agg_return, 4),
                "max_drawdown": round(agg_max_dd, 4),
                "calmar": round(agg_calmar, 3),
                "var_95": round(agg_var95, 4),
                "cvar_95": round(agg_cvar95, 4),
                "n_oos_days": len(all_oos_returns),
                "equity_curve": [round(float(v), 2) for v in agg_equity[::step]],
                "daily_returns": [round(float(r), 6) for r in all_oos_returns],
            },
            # Overfitting analysis
            "overfitting_analysis": {
                "avg_train_sharpe": round(avg_train_sharpe, 3),
                "avg_oos_sharpe": round(avg_oos_sharpe, 3),
                "sharpe_decay": round(sharpe_decay, 3),
                "overfitting_score": round(overfitting_score, 3),
                "likely_overfit": overfitting_score > 0.5,
                "train_sharpes": [round(s, 3) for s in train_sharpes],
                "oos_sharpes": [round(s, 3) for s in oos_sharpes],
            },
            # Parameter stability
            "parameter_stability": param_stability,
            "execution_time_ms": round(elapsed_ms, 2),
        }

    def _run_fold(self, asset_data: dict, assets: list[str],
                  fold: dict, cfg: WalkForwardConfig,
                  dates: list[str] | None = None) -> FoldResult:
        """Run a single walk-forward fold."""
        train_start = fold["train_start"]
        train_end = fold["train_end"]
        test_start = fold["test_start"]
        test_end = fold["test_end"]

        # ── Parameter optimization on train set (same engine, same costs) ──
        best_fast = cfg.strategy.fast_window
        best_slow = cfg.strategy.slow_window
        if cfg.optimize_parameters:
            best_train_sharpe = -math.inf
            for fw, sw in self._parameter_grid():
                train_sharpe = self._evaluate(asset_data, assets, train_start, train_end, fw, sw)["sharpe"]
                if train_sharpe > best_train_sharpe:
                    best_train_sharpe = train_sharpe
                    best_fast = fw
                    best_slow = sw

        # ── Evaluate on test set with best parameters ──
        oos_metrics = self._evaluate(asset_data, assets, test_start, test_end, best_fast, best_slow)

        # In-sample metrics
        train_metrics = self._evaluate(asset_data, assets, train_start, train_end, best_fast, best_slow)

        def span(a: int, b: int) -> str:
            return f"{dates[a]}..{dates[b - 1]}" if dates and b > a else ""

        return FoldResult(
            fold_index=fold["fold"],
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            train_sharpe=train_metrics.get("sharpe", 0),
            train_return=train_metrics.get("total_return", 0),
            train_max_dd=train_metrics.get("max_dd", 0),
            oos_sharpe=oos_metrics.get("sharpe", 0),
            oos_sortino=oos_metrics.get("sortino", 0),
            oos_return=oos_metrics.get("total_return", 0),
            oos_max_dd=oos_metrics.get("max_dd", 0),
            oos_calmar=oos_metrics.get("calmar", 0),
            oos_var_95=oos_metrics.get("var_95", 0),
            oos_cvar_95=oos_metrics.get("cvar_95", 0),
            oos_win_rate=oos_metrics.get("win_rate", 0),
            oos_total_trades=oos_metrics.get("n_trades", 0),
            best_fast_window=best_fast,
            best_slow_window=best_slow,
            oos_equity=oos_metrics.get("equity", []),
            oos_returns=oos_metrics.get("returns", []),
            train_dates=span(train_start, train_end),
            test_dates=span(test_start, test_end),
        )

    def _evaluate(self, asset_data: dict, assets: list[str], start: int, end: int,
                  fast_w: int, slow_w: int) -> dict:
        """Portfolio backtest over bars [start, end) with the given MA windows.

        Signals may read closes before ``start`` (history known at the time);
        nothing at or after ``end`` is read. Each window starts flat with
        fresh capital.
        """
        assets = [a for a in assets if a in asset_data]
        if not assets:
            return _empty_metrics()
        strategy = dataclasses.replace(self.config.strategy, strategy_type=StrategyType.MA_CROSSOVER,
                                       fast_window=fast_w, slow_window=slow_w)
        engine = BacktestEngine(BacktestConfig(assets=assets, strategy=strategy,
                                               execution=self._execution, initial_capital=_WF_CAPITAL))
        end = min(end, min(len(asset_data[a]["close"]) for a in assets))
        first = max(start, warmup_bars(strategy) - 1 + engine._delay())
        if end - first < 2:
            return _empty_metrics()

        res = engine._run_strategy(asset_data, first, end)
        equity = [_WF_CAPITAL] + [float(v) for v in res["equity_curve_net"]]
        returns = [equity[i] / equity[i - 1] - 1 if equity[i - 1] > 0 else 0.0
                   for i in range(1, len(equity))]
        rets = np.array(returns)
        wins, closed = _compute_win_rate(res["trade_log"])
        max_dd = _max_drawdown(equity)
        var95, cvar95 = _var_cvar(rets, 0.05)
        return {
            "sharpe": _sharpe(rets),
            "sortino": _sortino(rets),
            "total_return": float(equity[-1] / equity[0] - 1),
            "max_dd": max_dd,
            "calmar": _calmar(rets, max_dd),
            "var_95": var95,
            "cvar_95": cvar95,
            "win_rate": wins / max(1, closed),
            "n_trades": len(res["trade_log"]),
            "equity": equity,
            "returns": returns,
        }

    def _evaluate_period(self, close_data: dict, assets: list[str],
                         start: int, end: int, fast_w: int,
                         slow_w: int,
                         cfg: WalkForwardConfig | None = None) -> dict:
        """``_evaluate`` for bare close arrays."""
        return self._evaluate(_asset_data_from_closes(close_data), assets, start, end, fast_w, slow_w)

    def _quick_eval(self, close_data: dict, assets: list[str],
                    start: int, end: int, fast_w: int,
                    slow_w: int) -> float:
        """In-sample objective of the parameter search: net Sharpe of the
        same backtest the out-of-sample evaluation runs."""
        return self._evaluate_period(close_data, assets, start, end, fast_w, slow_w)["sharpe"]
