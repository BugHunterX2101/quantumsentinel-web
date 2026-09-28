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
from . import historical_data
# Strategy definitions live in strategy_signals so every engine shares them;
# re-exported here for existing importers.
from .strategy_signals import (  # noqa: F401
    MIN_DECISION_DELAY_BARS, StrategyConfig, StrategyType, latest_signal, signal_series, warmup_bars,
)

log = logging.getLogger(__name__)


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
        cfg = self.config

        # ── 1. Fetch historical data and align it on one calendar ──
        # Row i of every asset is the same trading day; see historical_data.
        all_tickers = list(dict.fromkeys(cfg.assets + [cfg.benchmark]))
        data = self._fetch_data(all_tickers, cfg.period)
        if data is None:
            raise ValueError("Failed to fetch historical data")
        panel = historical_data.aligned_panel(data, cfg.assets,
                                              min_rows=cfg.strategy.slow_window + 11)
        if not panel.tickers:
            raise ValueError("No assets had sufficient price history")
        n_bars = len(panel)
        start_bar = self._start_bar()
        if n_bars <= start_bar + 2:
            raise ValueError("Not enough data after warm-up period")
        dates = panel.date_strings()

        # ── 2. Run strategy ──
        results = self._run_strategy(panel.asset_data(), start_bar, n_bars, dates=dates)

        # ── 3. Benchmark on the same calendar ──
        benchmark_result = None
        bench_close = historical_data.series_on_calendar(data, cfg.benchmark, panel.dates)
        if bench_close is not None:
            benchmark_result = self._compute_benchmark(bench_close, start_bar, n_bars)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        output = self._compile_results(results, benchmark_result, elapsed_ms)
        output.update({
            "assets_traded": panel.tickers,
            "data_start": dates[0],
            "data_end": dates[-1],
            "first_trading_date": dates[start_bar],
            "n_bars": n_bars,
            "execution_delay_bars": self._delay(),
        })
        return output

    def _delay(self) -> int:
        delay = int(self.config.execution.execution_delay_bars)
        if delay < MIN_DECISION_DELAY_BARS:
            raise ValueError("execution_delay_bars must be at least 1: a signal cannot "
                             "fill at the close it was computed from")
        return delay

    def _start_bar(self) -> int:
        """First bar that may trade: past the warm-up, and late enough that the
        decision bar (start - delay) already has the strategy's full lookback."""
        strategy = self.config.strategy
        return max(strategy.slow_window + 5, warmup_bars(strategy) - 1 + self._delay())

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

    def _run_strategy(self, asset_data: dict, start_bar: int,
                      n_bars: int, dates: list[str] | None = None) -> dict:
        """Execute the strategy bar-by-bar.

        The order filled at ``close[bar]`` acts on the signal computed at the
        close of ``bar - delay``, and everything used to size it (volatility,
        volume, equity) was known at that close too.
        """
        cfg = self.config
        strategy = cfg.strategy
        delay = self._delay()

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
        signals = {t: signal_series(asset_data[t]["close"][:n_bars], strategy,
                                    start=max(0, start_bar - delay))
                   for t in valid_assets}

        for bar in range(max(start_bar, delay), n_bars):
            bar_date = dates[bar] if dates is not None else f"bar_{bar}"
            decision_bar = bar - delay
            # Sizing base: equity marked at the decision bar's closes (cash
            # alone would under-size every asset after the first one bought).
            sizing_equity = capital + sum(
                shares * float(asset_data[t]["close"][decision_bar])
                for t, shares in positions.items()
                if t in asset_data and decision_bar < len(asset_data[t]["close"])
            )

            # Exits settle before entries (an entry may be funded by a same-bar
            # exit), and each group runs in ticker order: when cash cannot
            # cover every entry, which one is capped must not depend on the
            # order the assets were listed in.
            exits, entries = [], []
            for ticker in sorted(valid_assets):
                if bar >= len(asset_data[ticker]["close"]):
                    continue
                signal = float(signals[ticker][decision_bar])
                current_pos = positions.get(ticker, 0.0)
                if signal < 0 and current_pos > 0:
                    exits.append((ticker, signal))
                elif signal > 0 and current_pos <= 0:
                    entries.append((ticker, signal))

            for ticker, signal in exits + entries:
                ad = asset_data[ticker]
                current_pos = positions.get(ticker, 0.0)
                price = float(ad["close"][bar])
                daily_vol, avg_volume = _pre_trade_liquidity(ad, decision_bar)

                if signal > 0 and current_pos <= 0:
                    # BUY signal — go long
                    desired = self.sizer.compute_shares(
                        sizing_equity, price, daily_vol,
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
                            sizing_equity, price, daily_vol, n_assets=n_assets,
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
        # FIX: this method's length parameter is named n_bars (see the
        # signature above) — min_len is the caller's (run()'s) local name for
        # the same value and doesn't exist in this scope, so every backtest
        # raised NameError here right after the bar loop finished, i.e. on
        # essentially every successful run (any position still open at the
        # end of the simulation window hits this liquidation step).
        final_bar = n_bars - 1
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
        """Signal as of the last bar of ``close`` (see strategy_signals)."""
        return latest_signal(close, strategy)

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
            # Full-resolution daily net returns (NOT subsampled by `step`).
            # The equity curves above are downsampled to ~200 points purely
            # to keep the chart payload small; recomputing returns from that
            # downsampled series (as the frontend statistical-tests panel
            # used to) silently changes the return periodicity — e.g. a 3-day
            # gap gets treated as one "daily" observation — which corrupts
            # every downstream statistic (t-test, bootstrap CI, Deflated
            # Sharpe, Ljung-Box) that assumes true daily returns. Exposing
            # the real series here is what /api/research/stat-test consumes.
            "daily_returns_net": [round(float(r), 6) for r in rets_net],
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


def _pre_trade_liquidity(ad: dict, decision_bar: int) -> tuple[float, float]:
    """Daily volatility and average volume known at the decision bar's close.

    Both slices of the returns window start at the same index, so element i
    of the diff divides by element i of the base-price window.
    """
    close = ad["close"]
    if decision_bar > 1:
        win_start = max(0, decision_bar - 21)
        returns_window = (np.diff(close[win_start:decision_bar + 1])
                          / np.maximum(close[win_start:decision_bar], 1e-9))
    else:
        returns_window = np.array([0.01])
    daily_vol = float(np.std(returns_window)) if len(returns_window) > 1 else 0.02
    volume = ad.get("volume")
    avg_volume = (float(np.mean(volume[max(0, decision_bar - 21):decision_bar + 1]))
                  if volume is not None and decision_bar > 1 else 1e6)
    return daily_vol, avg_volume


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
    """Annualised Sortino ratio.

    Downside deviation = sqrt(mean(min(r - target, 0)^2)) over ALL periods
    (not just the subset below target) — periods at/above target contribute
    a zero term rather than being dropped from the average. Dividing by only
    the count of downside periods (dropping the zero terms) overstates the
    downside deviation and understates the ratio.
    """
    if len(returns) < 2:
        return 0.0
    excess = returns - rf / 252
    downside_sq = np.minimum(returns - target, 0.0) ** 2
    dd = np.sqrt(np.mean(downside_sq))
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
    """Downside deviation below target return.

    RMS shortfall below target averaged over ALL periods (periods at/above
    target contribute a zero term) — not averaged over only the subset of
    periods that fall below target, which would overstate the deviation.
    """
    if len(returns) < 2:
        return 0.0
    downside_sq = np.minimum(returns - target, 0.0) ** 2
    return float(np.sqrt(np.mean(downside_sq)))


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
        # Full-resolution daily returns for statistical tests (the curve
        # above is downsampled for charting).
        "daily_returns_net": result["daily_returns_net"],
        "data_start": result.get("data_start"),
        "data_end": result.get("data_end"),
        "execution_delay_bars": result.get("execution_delay_bars"),
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
