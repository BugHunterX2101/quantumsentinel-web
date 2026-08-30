"""QuantumSentinel — Advanced Backtesting Engine.

Full-pipeline backtester with realistic execution simulation:
  Historical Data → Feature Engineering → Signal Generation →
  Portfolio Construction → Execution Simulator → Transaction Costs →
  Portfolio Returns → Risk / Performance Analysis

Replaces the simple MA-crossover backtester with a multi-strategy,
multi-asset engine supporting long/short, leverage, and position sizing.
"""
from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Callable

import numpy as np
import yfinance as yf

from .execution_model import (
    ExecutionConfig, ExecutionSimulator, PositionSizer, SizingMethod,
    zero_cost_config, retail_config, institutional_config, FillResult,
)
from . import signal_engine

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy definitions
# ---------------------------------------------------------------------------

class StrategyType:
    MA_CROSSOVER = "ma_crossover"
    SBA_SIGNAL = "sba_signal"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"


@dataclass
class StrategyConfig:
    """Configuration for a backtesting strategy."""
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


@dataclass
class BacktestConfig:
    """Full backtest configuration."""
    assets: list[str] = field(default_factory=lambda: ["AAPL"])
    period: str = "2y"
    initial_capital: float = 100_000.0
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    benchmark: str = "SPY"  # buy-and-hold benchmark


# ---------------------------------------------------------------------------
# Trade log entry
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    """Single trade in the backtest log."""
    date: str
    asset: str
    side: str          # "buy" or "sell"
    desired_qty: float
    filled_qty: float
    fill_price: float
    commission: float
    slippage_cost: float
    spread_cost: float
    total_cost: float
    partial: bool
    signal_value: float = 0.0
    position_after: float = 0.0

    def to_dict(self) -> dict:
        return {
            "date": self.date, "asset": self.asset, "side": self.side,
            "desired_qty": round(self.desired_qty, 4),
            "filled_qty": round(self.filled_qty, 4),
            "fill_price": round(self.fill_price, 4),
            "commission": round(self.commission, 4),
            "slippage_cost": round(self.slippage_cost, 4),
            "spread_cost": round(self.spread_cost, 4),
            "total_cost": round(self.total_cost, 4),
            "partial": self.partial,
            "signal_value": round(self.signal_value, 4),
            "position_after": round(self.position_after, 4),
        }


# ---------------------------------------------------------------------------
# Backtest Engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """Multi-asset backtesting engine with realistic execution."""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.executor = ExecutionSimulator(config.execution)
        self.sizer = config.execution.sizer

    def run(self) -> dict:
        """Execute the backtest and return results."""
        t0 = time.perf_counter()

        # ── 1. Fetch historical data ──
        all_tickers = list(set(self.config.assets + [self.config.benchmark]))
        data = self._fetch_data(all_tickers, self.config.period)
        if data is None:
            raise ValueError("Failed to fetch historical data")

        # Extract close, volume, high, low for each asset
        asset_data = {}
        for ticker in all_tickers:
            close = self._extract_series(data, ticker, "Close")
            volume = self._extract_series(data, ticker, "Volume")
            high = self._extract_series(data, ticker, "High")
            low = self._extract_series(data, ticker, "Low")
            if close is not None and len(close) > self.config.strategy.slow_window + 10:
                asset_data[ticker] = {
                    "close": close, "volume": volume,
                    "high": high, "low": low,
                }

        if not any(t in asset_data for t in self.config.assets):
            raise ValueError("No assets had sufficient price history")

        # ── 2. Determine common date range ──
        # Use the shortest series to align
        min_len = min(len(asset_data[t]["close"])
                      for t in self.config.assets if t in asset_data)
        start_bar = self.config.strategy.slow_window + 5  # warm-up

        if min_len <= start_bar + 2:
            raise ValueError("Not enough data after warm-up period")

        # ── 3. Run strategy ──
        results = self._run_strategy(asset_data, start_bar, min_len)

        # ── 4. Compute benchmark ──
        benchmark_result = None
        if self.config.benchmark in asset_data:
            benchmark_result = self._compute_benchmark(
                asset_data[self.config.benchmark]["close"],
                start_bar, min_len
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return self._compile_results(results, benchmark_result, elapsed_ms)

    def _fetch_data(self, tickers: list[str], period: str):
        """Download historical data for all tickers."""
        try:
            data = yf.download(
                tickers, period=period, interval="1d",
                progress=False, auto_adjust=True,
            )
            return data if not data.empty else None
        except Exception as exc:
            log.warning("Data download failed: %s", exc)
            return None

    def _extract_series(self, data, ticker: str, field: str) -> np.ndarray | None:
        """Extract a price series from yfinance download result."""
        import pandas as pd
        try:
            if isinstance(data.columns, pd.MultiIndex):
                series = data[field][ticker].dropna()
            else:
                series = data[field].dropna()
            arr = series.to_numpy(dtype=float)
            return arr if len(arr) > 0 else None
        except (KeyError, TypeError):
            return None

    def _run_strategy(self, asset_data: dict, start_bar: int,
                      n_bars: int) -> dict:
        """Execute the strategy bar-by-bar."""
        cfg = self.config
        strategy = cfg.strategy

        # Portfolio state
        capital = cfg.initial_capital
        positions: dict[str, float] = {}  # ticker → shares held
        equity_curve_gross = []
        equity_curve_net = []
        trade_log: list[TradeRecord] = []
        daily_returns_net = []
        daily_returns_gross = []
        total_commission = 0.0
        total_slippage = 0.0
        total_spread = 0.0
        total_borrow = 0.0
        turnover_shares = 0.0

        valid_assets = [t for t in cfg.assets if t in asset_data]
        n_assets = len(valid_assets)

        for bar in range(start_bar, n_bars):
            bar_date = f"bar_{bar}"

            # ── Generate signals ──
            signals = {}
            for ticker in valid_assets:
                ad = asset_data[ticker]
                if bar >= len(ad["close"]):
                    continue
                close_history = ad["close"][:bar + 1]
                sig = self._compute_signal(close_history, strategy)
                signals[ticker] = sig

            # ── Execute delayed orders (if delay > 0, execute previous bar's signals) ──
            # For simplicity, execute signals on the same bar
            for ticker, signal in signals.items():
                ad = asset_data[ticker]
                if bar >= len(ad["close"]):
                    continue
                price = float(ad["close"][bar])
                volume = float(ad["volume"][bar]) if ad["volume"] is not None and bar < len(ad["volume"]) else 1e6
                returns_window = np.diff(ad["close"][max(0, bar - 21):bar + 1]) / np.maximum(ad["close"][max(0, bar - 20):bar], 1e-9) if bar > 1 else np.array([0.01])
                daily_vol = float(np.std(returns_window)) if len(returns_window) > 1 else 0.02
                avg_volume = float(np.mean(ad["volume"][max(0, bar - 21):bar + 1])) if ad["volume"] is not None and bar > 1 else 1e6

                current_pos = positions.get(ticker, 0.0)

                if signal > 0 and current_pos <= 0:
                    # BUY signal — go long
                    desired = self.sizer.compute_shares(
                        capital, price, daily_vol,
                        n_assets=n_assets,
                    )
                    if current_pos < 0:
                        # Close short first
                        fill = self.executor.execute_order(
                            "buy", abs(current_pos), price, daily_vol,
                            avg_volume, ticker, capital, current_pos
                        )
                        if fill.filled:
                            capital -= fill.fill_qty * fill.fill_price + fill.commission
                            positions[ticker] = current_pos + fill.fill_qty
                            total_commission += fill.commission
                            total_slippage += fill.slippage_cost
                            total_spread += fill.spread_cost
                            turnover_shares += fill.fill_qty
                            trade_log.append(TradeRecord(
                                date=bar_date, asset=ticker, side="buy",
                                desired_qty=abs(current_pos),
                                filled_qty=fill.fill_qty,
                                fill_price=fill.fill_price,
                                commission=fill.commission,
                                slippage_cost=fill.slippage_cost,
                                spread_cost=fill.spread_cost,
                                total_cost=fill.total_cost,
                                partial=fill.partial,
                                signal_value=signal,
                                position_after=positions[ticker],
                            ))
                            current_pos = positions.get(ticker, 0.0)

                    if desired > 0:
                        fill = self.executor.execute_order(
                            "buy", desired, price, daily_vol,
                            avg_volume, ticker, capital, current_pos
                        )
                        if fill.filled:
                            capital -= fill.fill_qty * fill.fill_price + fill.commission
                            positions[ticker] = positions.get(ticker, 0.0) + fill.fill_qty
                            total_commission += fill.commission
                            total_slippage += fill.slippage_cost
                            total_spread += fill.spread_cost
                            turnover_shares += fill.fill_qty
                            trade_log.append(TradeRecord(
                                date=bar_date, asset=ticker, side="buy",
                                desired_qty=desired,
                                filled_qty=fill.fill_qty,
                                fill_price=fill.fill_price,
                                commission=fill.commission,
                                slippage_cost=fill.slippage_cost,
                                spread_cost=fill.spread_cost,
                                total_cost=fill.total_cost,
                                partial=fill.partial,
                                signal_value=signal,
                                position_after=positions[ticker],
                            ))

                elif signal < 0 and current_pos > 0:
                    # SELL signal — close long
                    fill = self.executor.execute_order(
                        "sell", abs(current_pos), price, daily_vol,
                        avg_volume, ticker, capital, current_pos
                    )
                    if fill.filled:
                        capital += fill.fill_qty * fill.fill_price - fill.commission
                        positions[ticker] = current_pos - fill.fill_qty
                        total_commission += fill.commission
                        total_slippage += fill.slippage_cost
                        total_spread += fill.spread_cost
                        turnover_shares += fill.fill_qty
                        trade_log.append(TradeRecord(
                            date=bar_date, asset=ticker, side="sell",
                            desired_qty=abs(current_pos),
                            filled_qty=fill.fill_qty,
                            fill_price=fill.fill_price,
                            commission=fill.commission,
                            slippage_cost=fill.slippage_cost,
                            spread_cost=fill.spread_cost,
                            total_cost=fill.total_cost,
                            partial=fill.partial,
                            signal_value=signal,
                            position_after=positions[ticker],
                        ))

                    # Optionally go short
                    if cfg.execution.allow_short_selling and signal < -0.5:
                        desired_short = self.sizer.compute_shares(
                            capital, price, daily_vol, n_assets=n_assets,
                        )
                        if desired_short > 0:
                            fill = self.executor.execute_order(
                                "sell", desired_short, price, daily_vol,
                                avg_volume, ticker, capital,
                                positions.get(ticker, 0.0)
                            )
                            if fill.filled:
                                capital += fill.fill_qty * fill.fill_price - fill.commission
                                positions[ticker] = positions.get(ticker, 0.0) - fill.fill_qty
                                total_commission += fill.commission
                                total_slippage += fill.slippage_cost
                                total_spread += fill.spread_cost
                                turnover_shares += fill.fill_qty
                                trade_log.append(TradeRecord(
                                    date=bar_date, asset=ticker, side="sell",
                                    desired_qty=desired_short,
                                    filled_qty=fill.fill_qty,
                                    fill_price=fill.fill_price,
                                    commission=fill.commission,
                                    slippage_cost=fill.slippage_cost,
                                    spread_cost=fill.spread_cost,
                                    total_cost=fill.total_cost,
                                    partial=fill.partial,
                                    signal_value=signal,
                                    position_after=positions[ticker],
                                ))

            # ── Daily borrow costs ──
            for ticker, shares in positions.items():
                if shares < 0 and ticker in asset_data:
                    ad = asset_data[ticker]
                    if bar < len(ad["close"]):
                        bc = self.executor.daily_borrow_cost(
                            ticker, shares, float(ad["close"][bar])
                        )
                        capital -= bc
                        total_borrow += bc

            # ── Mark-to-market ──
            port_value = capital
            for ticker, shares in positions.items():
                if ticker in asset_data and bar < len(asset_data[ticker]["close"]):
                    port_value += shares * float(asset_data[ticker]["close"][bar])

            equity_curve_net.append(port_value)

            # Gross = ignore transaction costs (for comparison)
            # We approximate gross by adding back cumulative costs
            equity_curve_gross.append(
                port_value + total_commission + total_slippage +
                total_spread + total_borrow
            )

            # Daily returns
            if len(equity_curve_net) > 1:
                prev = equity_curve_net[-2]
                if prev > 0:
                    daily_returns_net.append(
                        (equity_curve_net[-1] - prev) / prev
                    )
                    daily_returns_gross.append(
                        (equity_curve_gross[-1] - equity_curve_gross[-2]) / equity_curve_gross[-2]
                        if equity_curve_gross[-2] > 0 else 0.0
                    )

        # ── Liquidate remaining positions ──
        final_bar = min_len - 1
        for ticker, shares in list(positions.items()):
            if abs(shares) > 1e-9 and ticker in asset_data:
                ad = asset_data[ticker]
                if final_bar < len(ad["close"]):
                    price = float(ad["close"][final_bar])
                    if shares > 0:
                        capital += shares * price
                    else:
                        capital -= abs(shares) * price
                    positions[ticker] = 0.0

        return {
            "equity_curve_net": equity_curve_net,
            "equity_curve_gross": equity_curve_gross,
            "daily_returns_net": np.array(daily_returns_net),
            "daily_returns_gross": np.array(daily_returns_gross),
            "trade_log": trade_log,
            "total_commission": total_commission,
            "total_slippage": total_slippage,
            "total_spread": total_spread,
            "total_borrow": total_borrow,
            "turnover_shares": turnover_shares,
            "final_capital": equity_curve_net[-1] if equity_curve_net else self.config.initial_capital,
        }

    def _compute_signal(self, close: np.ndarray,
                        strategy: StrategyConfig) -> float:
        """Compute signal for a single asset at the current bar.

        Returns:
          > 0: buy signal (magnitude = strength)
          < 0: sell signal
          = 0: hold
        """
        if strategy.strategy_type == StrategyType.MA_CROSSOVER:
            if len(close) < strategy.slow_window + 1:
                return 0.0
            fast = np.mean(close[-strategy.fast_window:])
            slow = np.mean(close[-strategy.slow_window:])
            prev_fast = np.mean(close[-strategy.fast_window - 1:-1])
            prev_slow = np.mean(close[-strategy.slow_window - 1:-1])
            if prev_fast <= prev_slow and fast > slow:
                return 1.0  # bullish crossover
            elif prev_fast >= prev_slow and fast < slow:
                return -1.0  # bearish crossover
            return 0.0

        elif strategy.strategy_type == StrategyType.MOMENTUM:
            if len(close) < strategy.momentum_lookback + 1:
                return 0.0
            mom = close[-1] / close[-strategy.momentum_lookback - 1] - 1.0
            if mom > strategy.momentum_entry:
                return min(mom * 5, 1.0)
            elif mom < strategy.momentum_exit:
                return max(mom * 5, -1.0)
            return 0.0

        elif strategy.strategy_type == StrategyType.MEAN_REVERSION:
            if len(close) < strategy.mr_lookback + 1:
                return 0.0
            window = close[-strategy.mr_lookback:]
            mu = np.mean(window)
            sigma = np.std(window, ddof=1)
            if sigma < 1e-9:
                return 0.0
            z = (close[-1] - mu) / sigma
            if z < strategy.mr_entry_zscore:
                return min(abs(z) / 3, 1.0)
            elif z > strategy.mr_exit_zscore:
                return max(-abs(z) / 3, -1.0)
            return 0.0

        elif strategy.strategy_type == StrategyType.SBA_SIGNAL:
            # Use SBA features
            feats = signal_engine.extract_features(close)
            mom = feats["momentum"]
            spin = float(np.tanh(mom * 5.0))
            if spin > strategy.sba_buy_threshold:
                return spin
            elif spin < strategy.sba_sell_threshold:
                return spin
            return 0.0

        return 0.0

    def _compute_benchmark(self, close: np.ndarray, start_bar: int,
                           n_bars: int) -> dict:
        """Buy-and-hold benchmark for the same period."""
        if close is None or len(close) < n_bars:
            return {}
        benchmark_returns = []
        for i in range(start_bar + 1, n_bars):
            if close[i - 1] > 0:
                benchmark_returns.append(close[i] / close[i - 1] - 1)
        benchmark_returns = np.array(benchmark_returns)
        cumulative = np.cumprod(1 + benchmark_returns) if len(benchmark_returns) > 0 else np.array([1.0])
        equity = self.config.initial_capital * cumulative

        return {
            "returns": benchmark_returns,
            "equity_curve": equity.tolist(),
            "total_return": float(cumulative[-1] - 1) if len(cumulative) > 0 else 0.0,
            "sharpe": _sharpe(benchmark_returns),
            "max_drawdown": _max_drawdown(equity.tolist()),
        }

    def _compile_results(self, results: dict, benchmark: dict | None,
                         elapsed_ms: float) -> dict:
        """Compile all results into the final output dict."""
        cfg = self.config
        eq_net = results["equity_curve_net"]
        eq_gross = results["equity_curve_gross"]
        rets_net = results["daily_returns_net"]
        rets_gross = results["daily_returns_gross"]

        final_capital = results["final_capital"]
        total_return = final_capital / cfg.initial_capital - 1

        # Trade statistics
        trade_log = results["trade_log"]
        n_trades = len(trade_log)
        buy_trades = [t for t in trade_log if t.side == "buy"]
        sell_trades = [t for t in trade_log if t.side == "sell"]

        # Win rate from round-trips
        wins, closed_trades = _compute_win_rate(trade_log)

        # Risk metrics
        sharpe_net = _sharpe(rets_net)
        sharpe_gross = _sharpe(rets_gross)
        sortino_net = _sortino(rets_net)
        max_dd_net = _max_drawdown(eq_net)
        max_dd_gross = _max_drawdown(eq_gross)
        calmar = _calmar(rets_net, max_dd_net)

        # VaR and CVaR
        var95, cvar95 = _var_cvar(rets_net, 0.05)
        var99, cvar99 = _var_cvar(rets_net, 0.01)

        # Downside deviation
        downside_dev = _downside_deviation(rets_net)

        # Turnover
        turnover_notional = results["turnover_shares"]  # simplified

        # Cost breakdown
        cost_breakdown = {
            "total_commission": round(results["total_commission"], 2),
            "total_slippage": round(results["total_slippage"], 2),
            "total_spread": round(results["total_spread"], 2),
            "total_borrow": round(results["total_borrow"], 2),
            "total_costs": round(
                results["total_commission"] + results["total_slippage"] +
                results["total_spread"] + results["total_borrow"], 2
            ),
            "costs_pct_of_capital": round(
                (results["total_commission"] + results["total_slippage"] +
                 results["total_spread"] + results["total_borrow"])
                / cfg.initial_capital * 100, 2
            ),
        }

        # Subsample equity curves for response size
        max_points = 200
        step = max(1, len(eq_net) // max_points)

        output = {
            "asset": cfg.assets,
            "period": cfg.period,
            "strategy_type": cfg.strategy.strategy_type,
            "initial_capital": cfg.initial_capital,
            "final_capital": round(final_capital, 2),
            "total_return": round(total_return, 4),
            "total_return_gross": round(
                (eq_gross[-1] / cfg.initial_capital - 1) if eq_gross else 0, 4
            ),
            "sharpe_ratio_net": round(sharpe_net, 3),
            "sharpe_ratio_gross": round(sharpe_gross, 3),
            "sortino_ratio": round(sortino_net, 3),
            "calmar_ratio": round(calmar, 3),
            "max_drawdown_net": round(max_dd_net, 4),
            "max_drawdown_gross": round(max_dd_gross, 4),
            "var_95": round(var95, 4),
            "var_99": round(var99, 4),
            "cvar_95": round(cvar95, 4),
            "cvar_99": round(cvar99, 4),
            "downside_deviation": round(downside_dev, 6),
            "win_rate": round(wins / max(1, closed_trades), 3),
            "total_trades": n_trades,
            "buy_trades": len(buy_trades),
            "sell_trades": len(sell_trades),
            "cost_breakdown": cost_breakdown,
            "equity_curve_net": [round(float(v), 2) for v in eq_net[::step]],
            "equity_curve_gross": [round(float(v), 2) for v in eq_gross[::step]],
            "trade_log": [t.to_dict() for t in trade_log[:100]],  # cap at 100 trades
            "execution_time_ms": round(elapsed_ms, 2),
        }

        if benchmark:
            output["benchmark"] = {
                "ticker": cfg.benchmark,
                "total_return": round(benchmark.get("total_return", 0), 4),
                "sharpe": round(benchmark.get("sharpe", 0), 3),
                "max_drawdown": round(benchmark.get("max_drawdown", 0), 4),
                "equity_curve": [round(float(v), 2)
                                 for v in benchmark.get("equity_curve", [])[::step]],
            }
            # Alpha and Beta vs benchmark
            if len(rets_net) > 5 and len(benchmark.get("returns", [])) > 5:
                bm_rets = benchmark["returns"]
                min_len = min(len(rets_net), len(bm_rets))
                alpha, beta = _alpha_beta(
                    rets_net[:min_len], bm_rets[:min_len]
                )
                output["alpha"] = round(alpha, 4)
                output["beta"] = round(beta, 4)
                te = _tracking_error(rets_net[:min_len], bm_rets[:min_len])
                output["tracking_error"] = round(te, 4)
                ir = _information_ratio(rets_net[:min_len], bm_rets[:min_len])
                output["information_ratio"] = round(ir, 3)

        return output


# ---------------------------------------------------------------------------
# Risk metrics helper functions
# ---------------------------------------------------------------------------

def _sharpe(returns: np.ndarray, rf: float = 0.0) -> float:
    """Annualised Sharpe ratio (sample std)."""
    if len(returns) < 2:
        return 0.0
    excess = returns - rf / 252
    std = np.std(excess, ddof=1)
    if std < 1e-9:
        return 0.0
    return float(np.mean(excess) / std * math.sqrt(252))


def _sortino(returns: np.ndarray, rf: float = 0.0,
             target: float = 0.0) -> float:
    """Annualised Sortino ratio."""
    if len(returns) < 2:
        return 0.0
    excess = returns - rf / 252
    downside = returns[returns < target] - target
    if len(downside) < 1:
        return 0.0
    dd = np.sqrt(np.mean(downside ** 2))
    if dd < 1e-9:
        return 0.0
    return float(np.mean(excess) / dd * math.sqrt(252))


def _max_drawdown(equity_curve: list) -> float:
    """Maximum drawdown from peak."""
    if len(equity_curve) < 2:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for v in equity_curve:
        peak = max(peak, v)
        if peak > 0:
            dd = (peak - v) / peak
            max_dd = max(max_dd, dd)
    return float(max_dd)


def _calmar(returns: np.ndarray, max_dd: float) -> float:
    """Calmar ratio = annualised return / max drawdown."""
    if max_dd < 1e-9 or len(returns) < 2:
        return 0.0
    ann_return = float(np.mean(returns) * 252)
    return ann_return / max_dd


def _var_cvar(returns: np.ndarray, alpha: float) -> tuple[float, float]:
    """Empirical VaR and CVaR (Expected Shortfall)."""
    if len(returns) < 5:
        return 0.0, 0.0
    sorted_rets = np.sort(returns)
    idx = max(0, int(alpha * len(sorted_rets)))
    var = -float(sorted_rets[idx])
    # CVaR = average of all losses beyond VaR
    tail = sorted_rets[:idx + 1]
    cvar = -float(np.mean(tail)) if len(tail) > 0 else var
    return max(0, var), max(0, cvar)


def _downside_deviation(returns: np.ndarray, target: float = 0.0) -> float:
    """Downside deviation below target return."""
    if len(returns) < 2:
        return 0.0
    downside = returns[returns < target] - target
    if len(downside) < 1:
        return 0.0
    return float(np.sqrt(np.mean(downside ** 2)))


def _alpha_beta(strategy_returns: np.ndarray,
                benchmark_returns: np.ndarray) -> tuple[float, float]:
    """CAPM alpha and beta."""
    if len(strategy_returns) < 5:
        return 0.0, 1.0
    cov = np.cov(strategy_returns, benchmark_returns)
    var_b = cov[1, 1]
    if var_b < 1e-12:
        return 0.0, 1.0
    beta = float(cov[0, 1] / var_b)
    alpha = float((np.mean(strategy_returns) - beta * np.mean(benchmark_returns)) * 252)
    return alpha, beta


def _tracking_error(strategy_returns: np.ndarray,
                    benchmark_returns: np.ndarray) -> float:
    """Annualised tracking error."""
    diff = strategy_returns - benchmark_returns
    if len(diff) < 2:
        return 0.0
    return float(np.std(diff, ddof=1) * math.sqrt(252))


def _information_ratio(strategy_returns: np.ndarray,
                       benchmark_returns: np.ndarray) -> float:
    """Information ratio = excess return / tracking error."""
    te = _tracking_error(strategy_returns, benchmark_returns)
    if te < 1e-9:
        return 0.0
    excess = float(np.mean(strategy_returns - benchmark_returns) * 252)
    return excess / te


def _compute_win_rate(trade_log: list[TradeRecord]) -> tuple[int, int]:
    """Compute win rate from round-trip trades."""
    book: dict[str, list[float]] = {}  # asset → list of entry prices
    wins = 0
    closed = 0
    for t in trade_log:
        if t.side == "buy":
            book.setdefault(t.asset, []).append(t.fill_price)
        elif t.side == "sell" and t.asset in book and book[t.asset]:
            entry = book[t.asset].pop(0)
            closed += 1
            if t.fill_price > entry:
                wins += 1
    return wins, closed


# ---------------------------------------------------------------------------
# Omega ratio (used by extended risk metrics)
# ---------------------------------------------------------------------------

def _omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
    """Omega ratio: sum of gains above threshold / sum of losses below."""
    if len(returns) < 2:
        return 0.0
    gains = np.sum(np.maximum(returns - threshold, 0))
    losses = np.sum(np.maximum(threshold - returns, 0))
    if losses < 1e-9:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


# ---------------------------------------------------------------------------
# Legacy API compatibility — keeps the old endpoint working
# ---------------------------------------------------------------------------

def run_moving_average_backtest(asset: str, fast_window: int,
                                 slow_window: int, period: str,
                                 initial_capital: float = 100_000.0) -> dict:
    """Drop-in replacement for the old simple backtester.

    Now runs through the full execution pipeline with retail-level
    transaction costs. Returns a superset of the old response format.
    """
    config = BacktestConfig(
        assets=[asset],
        period=period,
        initial_capital=initial_capital,
        strategy=StrategyConfig(
            strategy_type=StrategyType.MA_CROSSOVER,
            fast_window=fast_window,
            slow_window=slow_window,
        ),
        execution=retail_config(),
        benchmark="SPY",
    )
    engine = BacktestEngine(config)
    result = engine.run()

    # Map to legacy format for backward compatibility
    legacy = {
        "asset": asset,
        "period": period,
        "fast_window": fast_window,
        "slow_window": slow_window,
        "initial_capital": initial_capital,
        "final_capital": result["final_capital"],
        "total_return": result["total_return"],
        "sharpe_ratio": result["sharpe_ratio_net"],
        "max_drawdown": result["max_drawdown_net"],
        "total_trades": result["total_trades"],
        "win_rate": result["win_rate"],
        "equity_curve": result["equity_curve_net"],
        # New fields
        "sharpe_ratio_gross": result["sharpe_ratio_gross"],
        "sortino_ratio": result["sortino_ratio"],
        "calmar_ratio": result["calmar_ratio"],
        "var_95": result["var_95"],
        "var_99": result["var_99"],
        "cvar_95": result["cvar_95"],
        "cvar_99": result["cvar_99"],
        "cost_breakdown": result["cost_breakdown"],
        "execution_note": (
            "Backtested under realistic transaction-cost and execution "
            "assumptions: commission, bid/ask spread, slippage."
        ),
    }
    if "benchmark" in result:
        legacy["benchmark"] = result["benchmark"]
    if "alpha" in result:
        legacy["alpha"] = result["alpha"]
        legacy["beta"] = result["beta"]

    return legacy
