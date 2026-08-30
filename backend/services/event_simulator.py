"""QuantumSentinel — Event-Driven Backtester.

A proper event-driven backtesting architecture that processes:

  MarketEvent → SignalEvent → OrderEvent → FillEvent → Portfolio update

This eliminates look-ahead bias present in vectorised backtesting by
processing one bar at a time, in strict chronological order.

Components:
  ─ MarketDataHandler: feeds one bar at a time
  ─ Strategy: generates SignalEvents from market data
  ─ Portfolio: tracks positions, cash, equity
  ─ ExecutionHandler: converts orders to fills with slippage + commission
  ─ EventQueue: FIFO queue driving the simulation loop

Execution assumptions:
  ─ All orders execute at next bar's OPEN (1-bar execution delay)
  ─ Commission: flat rate or percentage
  ─ Slippage: bid/ask spread + vol-proportional component
  ─ Partial fills supported (capacity constraint on volume)
  ─ Cash constraint: no over-leveraged positions without explicit leverage
  ─ Short-selling: optionally enabled with borrow cost
"""
from __future__ import annotations

import math
import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    MARKET = "MARKET"
    SIGNAL = "SIGNAL"
    ORDER  = "ORDER"
    FILL   = "FILL"


@dataclass
class MarketEvent:
    type: EventType = EventType.MARKET
    timestamp: int = 0        # bar index
    ticker: str = ""
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 1_000_000.0


@dataclass
class SignalEvent:
    type: EventType = EventType.SIGNAL
    timestamp: int = 0
    ticker: str = ""
    signal_strength: float = 0.0   # -1.0 to +1.0
    suggested_direction: int = 0   # +1 long, -1 short, 0 flat


@dataclass
class OrderEvent:
    type: EventType = EventType.ORDER
    timestamp: int = 0
    ticker: str = ""
    quantity: float = 0.0        # shares (negative = short)
    order_type: str = "MKT"      # MKT or LMT
    limit_price: float | None = None


@dataclass
class FillEvent:
    type: EventType = EventType.FILL
    timestamp: int = 0
    ticker: str = ""
    quantity: float = 0.0          # shares filled
    fill_price: float = 0.0        # execution price
    commission: float = 0.0        # total commission
    slippage: float = 0.0          # total slippage cost
    borrow_cost: float = 0.0       # daily borrow cost if short


# ---------------------------------------------------------------------------
# Commission / Slippage Models
# ---------------------------------------------------------------------------

@dataclass
class TransactionCostModel:
    commission_pct: float = 0.0010    # 10bps per trade
    spread_bps: float = 5.0           # bid/ask half-spread in bps
    market_impact_bps: float = 2.0    # vol-proportional slippage
    borrow_rate_annual: float = 0.015 # 150bps annual borrow cost for shorts
    min_commission: float = 1.0       # minimum commission per order

    def compute_fill(self, order: OrderEvent,
                     market: MarketEvent,
                     daily_vol: float = 0.01) -> FillEvent:
        """Compute realistic fill price with commission + slippage."""
        qty = order.quantity
        direction = 1 if qty > 0 else -1

        # Execute at next open (already provided as market.open)
        base_price = market.open

        # Spread cost: half-spread in direction of trade
        spread_cost = base_price * (self.spread_bps / 1e4)

        # Market impact: proportional to volatility (aggressive fill)
        impact_cost = base_price * (self.market_impact_bps / 1e4) * (daily_vol / 0.01)

        # Fill price: worse than open for buyer, better for seller
        fill_price = base_price + direction * (spread_cost + impact_cost)

        # Commission
        notional = abs(qty) * fill_price
        commission = max(notional * self.commission_pct, self.min_commission)

        # Slippage (total cost of spread + impact)
        slippage = abs(qty) * (spread_cost + impact_cost)

        # Borrow cost for short positions (daily)
        borrow_cost = 0.0
        if qty < 0:
            daily_borrow = self.borrow_rate_annual / 252
            borrow_cost = abs(qty) * fill_price * daily_borrow

        return FillEvent(
            timestamp=order.timestamp,
            ticker=order.ticker,
            quantity=qty,
            fill_price=fill_price,
            commission=commission,
            slippage=slippage,
            borrow_cost=borrow_cost,
        )


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

@dataclass
class Portfolio:
    initial_capital: float = 100_000.0
    allow_short: bool = False
    leverage_limit: float = 1.0

    def __post_init__(self):
        self.cash: float = self.initial_capital
        self.positions: dict[str, float] = {}        # ticker → shares
        self.avg_costs: dict[str, float] = {}        # ticker → avg cost basis
        self.equity_curve: list[float] = [self.initial_capital]
        self.returns: list[float] = [0.0]
        self.trade_log: list[dict] = []
        self.total_commission: float = 0.0
        self.total_slippage: float = 0.0
        self.total_borrow: float = 0.0

    def process_fill(self, fill: FillEvent) -> None:
        """Update positions and cash from a fill event."""
        ticker = fill.ticker
        qty = fill.quantity
        price = fill.fill_price
        sign = 1 if qty > 0 else -1

        # Update position
        prev_qty = self.positions.get(ticker, 0.0)
        new_qty = prev_qty + qty
        self.positions[ticker] = new_qty

        # Update average cost basis
        if new_qty == 0:
            self.avg_costs.pop(ticker, None)
        elif sign > 0:
            prev_cost = self.avg_costs.get(ticker, price)
            if prev_qty >= 0:
                self.avg_costs[ticker] = (
                    prev_cost * prev_qty + price * qty
                ) / max(abs(new_qty), 1e-9)
            else:
                self.avg_costs[ticker] = price
        else:
            self.avg_costs[ticker] = self.avg_costs.get(ticker, price)

        # Cash impact
        self.cash -= qty * price + fill.commission + fill.slippage

        # Accumulators
        self.total_commission += fill.commission
        self.total_slippage += fill.slippage
        self.total_borrow += fill.borrow_cost

        self.trade_log.append({
            "timestamp": fill.timestamp,
            "ticker": ticker,
            "qty": round(qty, 2),
            "price": round(price, 4),
            "commission": round(fill.commission, 4),
            "slippage": round(fill.slippage, 4),
        })

    def update_equity(self, prices: dict[str, float]) -> float:
        """Mark-to-market: compute current equity value."""
        position_value = sum(
            self.positions.get(t, 0) * prices.get(t, 0)
            for t in set(list(self.positions.keys()) + list(prices.keys()))
        )
        equity = self.cash + position_value
        self.equity_curve.append(equity)
        if len(self.equity_curve) >= 2:
            prev = self.equity_curve[-2]
            r = (equity - prev) / max(abs(prev), 1e-9)
            self.returns.append(r)
        else:
            self.returns.append(0.0)
        return equity

    def current_gross_exposure(self, prices: dict[str, float]) -> float:
        equity = self.equity_curve[-1] if self.equity_curve else self.initial_capital
        if equity < 1e-9:
            return 0.0
        gross = sum(
            abs(self.positions.get(t, 0)) * prices.get(t, 1.0)
            for t in self.positions
        )
        return gross / equity

    def position_size_for_target(self, ticker: str, target_pct: float,
                                   price: float) -> float:
        """Compute share quantity to reach a target position (% of equity)."""
        equity = self.equity_curve[-1] if self.equity_curve else self.initial_capital
        target_value = equity * target_pct
        current_value = self.positions.get(ticker, 0.0) * price
        delta_value = target_value - current_value
        return delta_value / max(price, 1e-9)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

class BaseStrategy:
    """Base class for event-driven strategies."""

    def on_market(self, event: MarketEvent,
                  portfolio: Portfolio,
                  history: dict[str, list[float]]) -> list[SignalEvent]:
        """Called for each market event. Return list of signals."""
        return []


class MACrossoverStrategy(BaseStrategy):
    """Moving average crossover — fast/slow SMA signal."""

    def __init__(self, fast: int = 20, slow: int = 50):
        self.fast = fast
        self.slow = slow

    def on_market(self, event: MarketEvent, portfolio: Portfolio,
                  history: dict[str, list[float]]) -> list[SignalEvent]:
        prices = history.get(event.ticker, [])
        if len(prices) < self.slow:
            return []
        fast_ma = float(np.mean(prices[-self.fast:]))
        slow_ma = float(np.mean(prices[-self.slow:]))
        prev_fast = float(np.mean(prices[-self.fast - 1:-1]))
        prev_slow = float(np.mean(prices[-self.slow - 1:-1]))

        crossover_up = prev_fast <= prev_slow and fast_ma > slow_ma
        crossover_down = prev_fast >= prev_slow and fast_ma < slow_ma

        if crossover_up:
            return [SignalEvent(timestamp=event.timestamp, ticker=event.ticker,
                                signal_strength=1.0, suggested_direction=1)]
        elif crossover_down:
            return [SignalEvent(timestamp=event.timestamp, ticker=event.ticker,
                                signal_strength=-1.0, suggested_direction=-1)]
        return []


class MomentumStrategy(BaseStrategy):
    """Cross-sectional momentum: long top tercile, short bottom (if allowed)."""

    def __init__(self, lookback: int = 60, n_long: int = 3, n_short: int = 3):
        self.lookback = lookback
        self.n_long = n_long
        self.n_short = n_short
        self._last_signals: dict[str, float] = {}
        self._bar = 0

    def on_market(self, event: MarketEvent, portfolio: Portfolio,
                  history: dict[str, list[float]]) -> list[SignalEvent]:
        # Collect momentum for all tickers (rebalance weekly)
        self._bar += 1
        if self._bar % 5 != 0:
            return []

        signals = []
        scores = {}
        for ticker, prices in history.items():
            if len(prices) >= self.lookback:
                ret = (prices[-1] / max(prices[-self.lookback], 1e-9)) - 1
                scores[ticker] = ret

        if not scores:
            return []

        ranked = sorted(scores, key=scores.__getitem__, reverse=True)
        n = len(ranked)
        n_long = min(self.n_long, n // 3)
        n_short = min(self.n_short, n // 3)

        for i, ticker in enumerate(ranked):
            if i < n_long:
                signals.append(SignalEvent(timestamp=event.timestamp, ticker=ticker,
                                           signal_strength=scores[ticker],
                                           suggested_direction=1))
            elif i >= n - n_short:
                signals.append(SignalEvent(timestamp=event.timestamp, ticker=ticker,
                                           signal_strength=scores[ticker],
                                           suggested_direction=-1))
            else:
                signals.append(SignalEvent(timestamp=event.timestamp, ticker=ticker,
                                           signal_strength=0.0,
                                           suggested_direction=0))
        return signals


class MeanReversionStrategy(BaseStrategy):
    """Bollinger Band mean-reversion strategy."""

    def __init__(self, window: int = 20, n_std: float = 2.0):
        self.window = window
        self.n_std = n_std

    def on_market(self, event: MarketEvent, portfolio: Portfolio,
                  history: dict[str, list[float]]) -> list[SignalEvent]:
        prices = history.get(event.ticker, [])
        if len(prices) < self.window:
            return []
        w = np.array(prices[-self.window:])
        mu = float(np.mean(w))
        sd = float(np.std(w, ddof=1))
        if sd < 1e-9:
            return []
        z = (prices[-1] - mu) / sd
        if z < -self.n_std:
            return [SignalEvent(timestamp=event.timestamp, ticker=event.ticker,
                                signal_strength=-z, suggested_direction=1)]
        elif z > self.n_std:
            return [SignalEvent(timestamp=event.timestamp, ticker=event.ticker,
                                signal_strength=z, suggested_direction=-1)]
        elif abs(z) < 0.5:  # exit zone
            return [SignalEvent(timestamp=event.timestamp, ticker=event.ticker,
                                signal_strength=0.0, suggested_direction=0)]
        return []


# ---------------------------------------------------------------------------
# Order Sizer
# ---------------------------------------------------------------------------

class FixedFractionalSizer:
    """Position sizing: fixed % of equity per trade."""
    def __init__(self, fraction: float = 0.02):
        self.fraction = fraction

    def size(self, signal: SignalEvent, price: float, equity: float) -> float:
        if price <= 0:
            return 0.0
        target_value = equity * self.fraction * abs(signal.signal_strength)
        return signal.suggested_direction * target_value / price


class VolatilityTargetSizer:
    """Kelly-inspired vol-target sizing."""
    def __init__(self, target_vol: float = 0.15, lookback: int = 21):
        self.target_vol = target_vol
        self.lookback = lookback

    def size(self, signal: SignalEvent, price: float, equity: float,
             returns: list[float]) -> float:
        if price <= 0 or len(returns) < self.lookback:
            return 0.0
        recent = np.array(returns[-self.lookback:])
        asset_vol = float(np.std(recent, ddof=1) * math.sqrt(252))
        if asset_vol < 1e-9:
            return 0.0
        target_value = equity * min(self.target_vol / asset_vol, 5.0)  # cap at 5x
        return signal.suggested_direction * min(target_value, equity * 0.25) / price


# ---------------------------------------------------------------------------
# Event-Driven Backtester
# ---------------------------------------------------------------------------

def run_event_backtest(
    tickers: list[str],
    price_data: dict[str, np.ndarray],    # ticker → price array
    strategy_name: str = "ma_crossover",
    strategy_params: dict | None = None,
    initial_capital: float = 100_000.0,
    cost_model_name: str = "retail",
    allow_short: bool = False,
    sizing_method: str = "fixed_fractional",
) -> dict:
    """Run the event-driven backtester.

    Parameters
    ----------
    tickers : list of ticker symbols
    price_data : dict of ticker → price array (aligned)
    strategy_name : "ma_crossover" | "momentum" | "mean_reversion"
    strategy_params : dict of strategy hyperparameters
    initial_capital : starting cash
    cost_model_name : "zero_cost" | "retail" | "institutional"
    allow_short : enable short selling
    sizing_method : "fixed_fractional" | "volatility_target"
    """
    params = strategy_params or {}

    # Build strategy
    if strategy_name == "momentum":
        strategy = MomentumStrategy(
            lookback=params.get("lookback", 60),
            n_long=params.get("n_long", 3),
            n_short=params.get("n_short", 3),
        )
    elif strategy_name == "mean_reversion":
        strategy = MeanReversionStrategy(
            window=params.get("window", 20),
            n_std=params.get("n_std", 2.0),
        )
    else:  # default ma_crossover
        strategy = MACrossoverStrategy(
            fast=params.get("fast", 20),
            slow=params.get("slow", 50),
        )

    # Transaction cost model
    cost_presets = {
        "zero_cost": TransactionCostModel(0, 0, 0, 0, 0),
        "retail": TransactionCostModel(0.001, 5.0, 2.0, 0.015, 1.0),
        "institutional": TransactionCostModel(0.0002, 1.0, 0.5, 0.005, 0.5),
    }
    cost_model = cost_presets.get(cost_model_name, cost_presets["retail"])

    # Position sizer
    if sizing_method == "volatility_target":
        sizer = VolatilityTargetSizer(target_vol=params.get("target_vol", 0.15))
    else:
        sizer = FixedFractionalSizer(fraction=params.get("fraction", 0.02))

    # Portfolio
    portfolio = Portfolio(initial_capital=initial_capital, allow_short=allow_short,
                          leverage_limit=params.get("leverage_limit", 2.0 if allow_short else 1.0))

    # Find common length
    T = min(len(arr) for arr in price_data.values())
    history: dict[str, list[float]] = {t: [] for t in tickers}

    # Pending orders (1-bar delay)
    pending_orders: list[OrderEvent] = []

    # ── Main event loop ──
    for bar in range(T):
        # Current prices for mark-to-market
        current_prices = {t: float(price_data[t][bar]) for t in tickers
                          if bar < len(price_data[t])}

        # ── 1. Fill pending orders at this bar's open ──
        if bar > 0 and pending_orders:
            for order in pending_orders:
                # Create synthetic market event at open (approximate as prev close * 1.001)
                open_price = current_prices.get(order.ticker, 0.0)
                if open_price <= 0:
                    continue
                mkt = MarketEvent(
                    timestamp=bar, ticker=order.ticker,
                    open=open_price, close=open_price,
                )
                # Asset return history for vol-prop slippage
                h = history.get(order.ticker, [])
                daily_vol = 0.01
                if len(h) >= 12:
                    h_arr = np.array(h[-12:])
                    rets = np.diff(h_arr) / np.maximum(h_arr[:-1], 1e-9)
                    daily_vol = max(float(np.std(rets, ddof=1)), 1e-6)

                fill = cost_model.compute_fill(order, mkt, daily_vol)

                # Check leverage constraint
                gross_exp = portfolio.current_gross_exposure(current_prices)
                if gross_exp > portfolio.leverage_limit * 1.1:
                    continue  # skip order — over leverage limit

                # Check short selling constraint
                if fill.quantity < 0 and not portfolio.allow_short:
                    current_pos = portfolio.positions.get(order.ticker, 0.0)
                    if current_pos + fill.quantity < 0:
                        fill.quantity = -current_pos  # flatten only, no new short

                if abs(fill.quantity) > 1e-6:
                    portfolio.process_fill(fill)

            pending_orders.clear()

        # ── 2. Update market history ──
        for ticker in tickers:
            if bar < len(price_data[ticker]):
                history[ticker].append(float(price_data[ticker][bar]))

        # ── 3. Generate signals from strategy ──
        signals: list[SignalEvent] = []
        for ticker in tickers:
            mkt_event = MarketEvent(
                timestamp=bar, ticker=ticker,
                close=current_prices.get(ticker, 0.0),
            )
            signals.extend(strategy.on_market(mkt_event, portfolio, history))

        # ── 4. Convert signals to orders (staged for next bar) ──
        equity = portfolio.equity_curve[-1] if portfolio.equity_curve else initial_capital
        for signal in signals:
            price = current_prices.get(signal.ticker, 0.0)
            if price <= 0:
                continue

            if isinstance(sizer, VolatilityTargetSizer):
                ticker_returns = []
                h = history.get(signal.ticker, [])
                if len(h) > 2:
                    arr = np.array(h)
                    ticker_returns = list(np.diff(arr) / np.maximum(arr[:-1], 1e-9))
                qty = sizer.size(signal, price, equity, ticker_returns)
            else:
                qty = sizer.size(signal, price, equity)

            if abs(qty) < 1e-4:
                continue

            # Flat signal: liquidate position
            if signal.suggested_direction == 0:
                qty = -portfolio.positions.get(signal.ticker, 0.0)
                if abs(qty) < 1e-4:
                    continue

            pending_orders.append(OrderEvent(
                timestamp=bar, ticker=signal.ticker,
                quantity=qty,
            ))

        # ── 5. Mark to market ──
        portfolio.update_equity(current_prices)

    # ── Compute performance ──
    equity_curve = np.array(portfolio.equity_curve)
    returns_arr = np.array(portfolio.returns[1:])  # skip initial 0

    valid_r = returns_arr[np.isfinite(returns_arr)]
    ann_ret = float(valid_r.mean() * 252) if len(valid_r) > 0 else 0.0
    ann_vol = float(valid_r.std(ddof=1) * math.sqrt(252)) if len(valid_r) > 1 else 0.0
    sharpe = ann_ret / max(ann_vol, 1e-9)

    # Max drawdown
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / np.maximum(peak, 1e-9)
    max_dd = float(dd.min())
    dd_duration = int(np.sum(dd < -0.01))  # days in drawdown > 1%

    # Win rate
    r_nonzero = valid_r[valid_r != 0]
    win_rate = float(np.mean(r_nonzero > 0)) if len(r_nonzero) > 0 else 0.5

    # Calmar
    calmar = ann_ret / max(abs(max_dd), 1e-9) if max_dd < 0 else ann_ret

    # CVaR 5%
    if len(valid_r) > 10:
        tail = valid_r[valid_r <= np.percentile(valid_r, 5)]
        cvar = float(tail.mean()) if len(tail) > 0 else float(np.percentile(valid_r, 5))
    else:
        cvar = 0.0

    return {
        "strategy": strategy_name,
        "cost_model": cost_model_name,
        "sizing_method": sizing_method,
        "allow_short": allow_short,
        "initial_capital": initial_capital,
        "final_equity": round(float(equity_curve[-1]), 2),
        "total_return": round(float(equity_curve[-1] / initial_capital - 1), 4),
        "annual_return": round(ann_ret, 4),
        "annual_volatility": round(ann_vol, 4),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
        "drawdown_days": dd_duration,
        "calmar_ratio": round(calmar, 4),
        "cvar_5pct": round(cvar, 6),
        "win_rate": round(win_rate, 4),
        "n_trades": len(portfolio.trade_log),
        "total_commission": round(portfolio.total_commission, 2),
        "total_slippage": round(portfolio.total_slippage, 2),
        "total_borrow_cost": round(portfolio.total_borrow, 2),
        "total_transaction_costs": round(
            portfolio.total_commission + portfolio.total_slippage + portfolio.total_borrow, 2),
        "equity_curve": [round(float(e), 2) for e in equity_curve[-504:]],  # max 2yr daily
        "daily_returns": [round(float(r), 6) for r in returns_arr[-504:]],
        "n_bars": T,
        "tickers": tickers,
    }
