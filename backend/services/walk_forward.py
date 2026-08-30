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

import logging
import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import yfinance as yf

from .backtest_service import (
    BacktestConfig, BacktestEngine, StrategyConfig, StrategyType,
    _sharpe, _sortino, _max_drawdown, _var_cvar, _calmar,
    _omega_ratio, _downside_deviation,
)
from .execution_model import ExecutionConfig, retail_config

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

    def to_dict(self) -> dict:
        return {
            "fold": self.fold_index,
            "train_period": f"bar_{self.train_start}_to_{self.train_end}",
            "test_period": f"bar_{self.test_start}_to_{self.test_end}",
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


class WalkForwardEngine:
    """Walk-forward validation engine."""

    def __init__(self, config: WalkForwardConfig):
        self.config = config

    def run(self) -> dict:
        """Execute walk-forward validation."""
        t0 = time.perf_counter()
        cfg = self.config

        # ── Fetch all data upfront ──
        period_map = {3: "3y", 4: "4y", 5: "5y", 6: "6y", 7: "7y",
                      8: "8y", 10: "10y"}
        yf_period = period_map.get(cfg.total_years, f"{cfg.total_years}y")

        all_tickers = list(set(cfg.assets + [cfg.benchmark]))
        try:
            data = yf.download(
                all_tickers, period=yf_period, interval="1d",
                progress=False, auto_adjust=True,
            )
        except Exception as exc:
            raise ValueError(f"Data download failed: {exc}")

        if data is None or data.empty:
            raise ValueError("Empty data returned")

        # Extract close prices
        import pandas as pd
        close_data = {}
        for ticker in all_tickers:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    series = data["Close"][ticker].dropna()
                else:
                    series = data["Close"].dropna()
                arr = series.to_numpy(dtype=float)
                if len(arr) > 100:
                    close_data[ticker] = arr
            except (KeyError, TypeError):
                continue

        valid_assets = [t for t in cfg.assets if t in close_data]
        if not valid_assets:
            raise ValueError("No valid assets found")

        # Use shortest series length
        min_len = min(len(close_data[t]) for t in valid_assets)
        bars_per_year = 252
        train_bars = cfg.train_years * bars_per_year
        test_bars = cfg.test_years * bars_per_year
        step_bars = cfg.step_years * bars_per_year

        # ── Generate folds ──
        folds = []
        fold_idx = 0

        if cfg.window_type == WindowType.ROLLING:
            start = 0
            while start + train_bars + test_bars <= min_len:
                folds.append({
                    "fold": fold_idx,
                    "train_start": start,
                    "train_end": start + train_bars,
                    "test_start": start + train_bars,
                    "test_end": min(start + train_bars + test_bars, min_len),
                })
                fold_idx += 1
                start += step_bars
        else:  # EXPANDING
            start = 0
            test_start = train_bars
            while test_start + test_bars <= min_len:
                folds.append({
                    "fold": fold_idx,
                    "train_start": start,  # always 0 for expanding
                    "train_end": test_start,
                    "test_start": test_start,
                    "test_end": min(test_start + test_bars, min_len),
                })
                fold_idx += 1
                test_start += step_bars

        if not folds:
            raise ValueError(
                f"Not enough data for walk-forward: need {train_bars + test_bars} "
                f"bars, have {min_len}"
            )

        # ── Run each fold ──
        fold_results: list[FoldResult] = []
        all_oos_returns = []

        for fold in folds:
            result = self._run_fold(
                close_data, valid_assets, fold, cfg
            )
            fold_results.append(result)
            all_oos_returns.extend(result.oos_returns)

        # ── Aggregate OOS results ──
        all_oos = np.array(all_oos_returns) if all_oos_returns else np.array([0.0])
        agg_equity = [self.config.execution.sizer.risk_per_trade]  # placeholder
        # Build aggregated equity from OOS returns
        agg_equity = [100_000.0]
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

        return {
            "window_type": cfg.window_type,
            "n_folds": len(fold_results),
            "train_years": cfg.train_years,
            "test_years": cfg.test_years,
            "assets": valid_assets,
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
                "n_oos_days": len(all_oos),
                "equity_curve": [round(float(v), 2) for v in agg_equity[::step]],
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

    def _run_fold(self, close_data: dict, assets: list[str],
                  fold: dict, cfg: WalkForwardConfig) -> FoldResult:
        """Run a single walk-forward fold."""
        train_start = fold["train_start"]
        train_end = fold["train_end"]
        test_start = fold["test_start"]
        test_end = fold["test_end"]

        # ── Parameter optimization on train set ──
        best_fast = cfg.strategy.fast_window
        best_slow = cfg.strategy.slow_window
        best_train_sharpe = -999.0

        if cfg.optimize_parameters:
            for fw in cfg.fast_window_range:
                for sw in cfg.slow_window_range:
                    if sw <= fw:
                        continue
                    # Quick in-sample evaluation
                    train_sharpe = self._quick_eval(
                        close_data, assets, train_start, train_end, fw, sw
                    )
                    if train_sharpe > best_train_sharpe:
                        best_train_sharpe = train_sharpe
                        best_fast = fw
                        best_slow = sw

        # ── Evaluate on test set with best parameters ──
        oos_metrics = self._evaluate_period(
            close_data, assets, test_start, test_end, best_fast, best_slow, cfg
        )

        # In-sample metrics
        train_metrics = self._evaluate_period(
            close_data, assets, train_start, train_end, best_fast, best_slow, cfg
        )

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
        )

    def _quick_eval(self, close_data: dict, assets: list[str],
                    start: int, end: int, fast_w: int,
                    slow_w: int) -> float:
        """Quick Sharpe evaluation for parameter search (no execution costs)."""
        all_returns = []
        for asset in assets:
            if asset not in close_data:
                continue
            close = close_data[asset]
            if end > len(close):
                continue
            # Simple MA crossover signal → return accumulation
            for i in range(start + slow_w + 1, min(end, len(close))):
                fast_ma = np.mean(close[i - fast_w:i])
                slow_ma = np.mean(close[i - slow_w:i])
                prev_fast = np.mean(close[i - fast_w - 1:i - 1])
                prev_slow = np.mean(close[i - slow_w - 1:i - 1])
                # Simplified: track if we're in position
                in_position = fast_ma > slow_ma
                if in_position and close[i - 1] > 0:
                    ret = close[i] / close[i - 1] - 1
                    all_returns.append(ret)
                else:
                    all_returns.append(0.0)

        returns = np.array(all_returns)
        return _sharpe(returns) if len(returns) > 5 else -999.0

    def _evaluate_period(self, close_data: dict, assets: list[str],
                         start: int, end: int, fast_w: int,
                         slow_w: int,
                         cfg: WalkForwardConfig) -> dict:
        """Full evaluation of a period with execution costs."""
        capital = 100_000.0
        position = 0.0
        equity = [capital]
        returns_list = []
        trades = 0
        wins = 0
        closed = 0
        entry_price = 0.0

        # Use first asset for simplicity in walk-forward
        asset = assets[0]
        if asset not in close_data:
            return {"sharpe": 0, "total_return": 0, "max_dd": 0,
                    "returns": [], "equity": [capital]}

        close = close_data[asset]
        if end > len(close):
            end = len(close)

        from .execution_model import ExecutionSimulator
        executor = ExecutionSimulator(cfg.execution)

        for i in range(start + slow_w + 1, end):
            if i >= len(close) or i < 1:
                continue

            price = float(close[i])
            prev_price = float(close[i - 1])

            # Signal
            fast_ma = np.mean(close[max(0, i - fast_w):i])
            slow_ma = np.mean(close[max(0, i - slow_w):i])
            prev_fast = np.mean(close[max(0, i - fast_w - 1):max(1, i - 1)])
            prev_slow = np.mean(close[max(0, i - slow_w - 1):max(1, i - 1)])

            # Crossover detection
            if position == 0 and prev_fast <= prev_slow and fast_ma > slow_ma:
                # Buy
                vol = float(np.std(np.diff(close[max(0, i - 22):i]) /
                            np.maximum(close[max(0, i - 21):max(1, i - 1)], 1e-9))) if i > 2 else 0.02
                desired = capital * 0.95 / price
                fill = executor.execute_order(
                    "buy", desired, price, vol, 1e6, asset, capital, 0
                )
                if fill.filled:
                    position = fill.fill_qty
                    entry_price = fill.fill_price
                    capital -= fill.fill_qty * fill.fill_price + fill.commission
                    trades += 1

            elif position > 0 and prev_fast >= prev_slow and fast_ma < slow_ma:
                # Sell
                vol = float(np.std(np.diff(close[max(0, i - 22):i]) /
                            np.maximum(close[max(0, i - 21):max(1, i - 1)], 1e-9))) if i > 2 else 0.02
                fill = executor.execute_order(
                    "sell", position, price, vol, 1e6, asset, capital, position
                )
                if fill.filled:
                    capital += fill.fill_qty * fill.fill_price - fill.commission
                    closed += 1
                    if fill.fill_price > entry_price:
                        wins += 1
                    position -= fill.fill_qty
                    trades += 1

            # Mark-to-market
            port_val = capital + position * price
            equity.append(port_val)
            if len(equity) > 1 and equity[-2] > 0:
                returns_list.append((equity[-1] - equity[-2]) / equity[-2])

        rets = np.array(returns_list)
        max_dd = _max_drawdown(equity)
        var95, cvar95 = _var_cvar(rets, 0.05)
        total_ret = float(equity[-1] / equity[0] - 1) if equity[0] > 0 else 0

        return {
            "sharpe": _sharpe(rets),
            "sortino": _sortino(rets),
            "total_return": total_ret,
            "max_dd": max_dd,
            "calmar": _calmar(rets, max_dd),
            "var_95": var95,
            "cvar_95": cvar95,
            "win_rate": wins / max(1, closed),
            "n_trades": trades,
            "equity": equity,
            "returns": returns_list,
        }
