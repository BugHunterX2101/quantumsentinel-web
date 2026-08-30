"""QuantumSentinel — Execution Model.

Realistic transaction-cost and execution simulation framework for backtesting.

Models:
  - CommissionModel: Fixed + percentage trading commissions
  - SlippageModel: Market-impact slippage based on volatility and order size
  - SpreadModel: Bid/ask spread estimation
  - BorrowCostModel: Short-selling borrow costs
  - PositionSizer: Fixed-fractional, vol-targeting, Kelly criterion
  - ExecutionSimulator: Full execution pipeline combining all models
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Commission Model
# ---------------------------------------------------------------------------

@dataclass
class CommissionModel:
    """Configurable commission structure.

    Supports fixed per-share cost, percentage of notional, and minimum
    per-trade fee.  Default: Interactive Brokers–like pricing.
    """
    per_share: float = 0.005       # USD per share
    pct_of_notional: float = 0.001  # 0.1% of trade notional
    min_per_trade: float = 1.00     # minimum commission per trade
    max_pct_of_notional: float = 0.005  # cap at 0.5% of notional

    def compute(self, shares: float, price: float) -> float:
        notional = abs(shares) * price
        raw = max(self.min_per_trade,
                  abs(shares) * self.per_share + notional * self.pct_of_notional)
        cap = notional * self.max_pct_of_notional
        return min(raw, cap)


# ---------------------------------------------------------------------------
# Slippage Model
# ---------------------------------------------------------------------------

@dataclass
class SlippageModel:
    """Linear slippage model: slippage proportional to volatility and order
    size relative to average volume.

    slippage_bps = base_bps + volatility_factor * daily_vol * (order_size / avg_volume)
    """
    base_bps: float = 1.0          # minimum slippage in basis points
    volatility_factor: float = 0.1  # scale factor for vol-adjusted slippage
    volume_participation_limit: float = 0.05  # max 5% of avg daily volume

    def compute(self, price: float, shares: float, daily_vol: float,
                avg_volume: float) -> float:
        """Return the slippage cost per share (positive = unfavourable)."""
        if avg_volume <= 0:
            participation = 1.0
        else:
            participation = min(abs(shares) / avg_volume,
                                self.volume_participation_limit)
        slippage_bps = (self.base_bps +
                        self.volatility_factor * daily_vol * 10_000 *
                        participation)
        return price * slippage_bps / 10_000

    def can_fill_fully(self, shares: float, avg_volume: float) -> bool:
        """Return True if order size is within volume participation limit."""
        if avg_volume <= 0:
            return False
        return abs(shares) <= avg_volume * self.volume_participation_limit

    def partial_fill_qty(self, shares: float, avg_volume: float) -> float:
        """Return the maximum shares that can be filled this bar."""
        max_qty = avg_volume * self.volume_participation_limit
        return min(abs(shares), max_qty) * np.sign(shares)


# ---------------------------------------------------------------------------
# Spread Model
# ---------------------------------------------------------------------------

@dataclass
class SpreadModel:
    """Bid/ask spread estimation from daily volatility.

    Uses the Corwin-Schultz (2012) high-low spread estimator principle:
    wider spreads for more volatile / less liquid assets.
    """
    base_spread_bps: float = 2.0   # minimum spread in bps
    vol_multiplier: float = 0.5     # spread widens with volatility

    def estimate_spread(self, price: float, daily_vol: float) -> float:
        """Return estimated half-spread (one-way cost) in price terms."""
        spread_bps = self.base_spread_bps + self.vol_multiplier * daily_vol * 10_000
        half_spread = price * spread_bps / (2 * 10_000)
        return half_spread

    def adjust_fill_price(self, price: float, daily_vol: float,
                          side: str) -> float:
        """Return the fill price adjusted for spread.

        Buy  → price + half_spread  (cross the ask)
        Sell → price - half_spread  (hit the bid)
        """
        hs = self.estimate_spread(price, daily_vol)
        if side == "buy":
            return price + hs
        return price - hs


# ---------------------------------------------------------------------------
# Borrow Cost Model (for short selling)
# ---------------------------------------------------------------------------

@dataclass
class BorrowCostModel:
    """Annual borrow cost for short positions.

    General collateral = low borrow rate.  Hard-to-borrow stocks get a
    higher rate.  Cost is accrued daily on the mark-to-market value of
    short positions.
    """
    general_annual_rate: float = 0.005  # 0.5% p.a. for easy-to-borrow
    hard_to_borrow_rate: float = 0.05   # 5% p.a. for HTB stocks
    hard_to_borrow_tickers: set = field(default_factory=set)

    def daily_cost(self, ticker: str, shares: float, price: float) -> float:
        """Return daily borrow cost for a short position (positive = cost)."""
        if shares >= 0:
            return 0.0  # long or flat — no borrow cost
        rate = (self.hard_to_borrow_rate
                if ticker in self.hard_to_borrow_tickers
                else self.general_annual_rate)
        return abs(shares) * price * rate / 252


# ---------------------------------------------------------------------------
# Position Sizing
# ---------------------------------------------------------------------------

class SizingMethod(str, Enum):
    FIXED_FRACTIONAL = "fixed_fractional"
    VOLATILITY_TARGET = "volatility_target"
    KELLY = "kelly"
    EQUAL_WEIGHT = "equal_weight"


@dataclass
class PositionSizer:
    """Position sizing algorithms."""

    method: SizingMethod = SizingMethod.FIXED_FRACTIONAL
    risk_per_trade: float = 0.02    # 2% of capital per trade
    target_volatility: float = 0.15  # 15% annualised target vol
    max_position_pct: float = 0.20   # max 20% of capital in one name
    max_leverage: float = 1.0        # 1.0 = no leverage

    def compute_shares(self, capital: float, price: float,
                       daily_vol: float = 0.02,
                       win_rate: float = 0.5,
                       avg_win_loss_ratio: float = 1.5,
                       n_assets: int = 1) -> float:
        """Return the number of shares to trade."""
        if price <= 0 or capital <= 0:
            return 0.0

        if self.method == SizingMethod.FIXED_FRACTIONAL:
            risk_capital = capital * self.risk_per_trade
            shares = risk_capital / price

        elif self.method == SizingMethod.VOLATILITY_TARGET:
            # Scale position so portfolio-level vol ≈ target
            ann_vol = daily_vol * math.sqrt(252)
            if ann_vol < 1e-9:
                ann_vol = 0.2
            vol_scalar = self.target_volatility / ann_vol
            notional = capital * vol_scalar / max(n_assets, 1)
            shares = notional / price

        elif self.method == SizingMethod.KELLY:
            # Kelly fraction: f* = (p * b - q) / b
            p = max(0.01, min(0.99, win_rate))
            b = max(0.01, avg_win_loss_ratio)
            q = 1.0 - p
            kelly_f = (p * b - q) / b
            kelly_f = max(0.0, min(kelly_f, 0.5))  # half-Kelly cap
            notional = capital * kelly_f * 0.5  # use half-Kelly
            shares = notional / price

        elif self.method == SizingMethod.EQUAL_WEIGHT:
            notional = capital / max(n_assets, 1)
            shares = notional / price

        else:
            shares = 0.0

        # Apply position cap
        max_notional = capital * self.max_position_pct * self.max_leverage
        shares = min(shares, max_notional / price)

        return round(shares, 6)


# ---------------------------------------------------------------------------
# Execution Simulator — combines all models
# ---------------------------------------------------------------------------

@dataclass
class ExecutionConfig:
    """Bundled configuration for the execution simulator."""
    commission: CommissionModel = field(default_factory=CommissionModel)
    slippage: SlippageModel = field(default_factory=SlippageModel)
    spread: SpreadModel = field(default_factory=SpreadModel)
    borrow: BorrowCostModel = field(default_factory=BorrowCostModel)
    sizer: PositionSizer = field(default_factory=PositionSizer)

    execution_delay_bars: int = 0    # 0 = fill same bar, 1 = next bar, etc.
    allow_short_selling: bool = True
    cash_reserve_pct: float = 0.0    # keep X% as cash buffer
    leverage_limit: float = 1.0       # max gross leverage


@dataclass
class FillResult:
    """Result of a simulated order execution."""
    filled: bool
    fill_price: float
    fill_qty: float
    commission: float
    slippage_cost: float
    spread_cost: float
    total_cost: float
    partial: bool = False
    reason: str = ""


class ExecutionSimulator:
    """Simulates realistic order execution with transaction costs."""

    def __init__(self, config: ExecutionConfig | None = None):
        self.config = config or ExecutionConfig()

    def execute_order(self, side: str, desired_shares: float,
                      price: float, daily_vol: float,
                      avg_volume: float, ticker: str = "",
                      available_cash: float = float("inf"),
                      current_position: float = 0.0) -> FillResult:
        """Simulate execution of a single order.

        Parameters
        ----------
        side : "buy" or "sell"
        desired_shares : number of shares (always positive)
        price : reference price (e.g. close price of the bar)
        daily_vol : daily return volatility of the asset
        avg_volume : average daily volume in shares
        ticker : asset ticker for borrow cost lookup
        available_cash : cash available for buying
        current_position : current holding (positive = long, negative = short)

        Returns
        -------
        FillResult with all cost components
        """
        cfg = self.config
        desired_shares = abs(desired_shares)

        if desired_shares < 1e-9 or price < 1e-9:
            return FillResult(False, 0, 0, 0, 0, 0, 0, reason="zero order")

        # Check short-selling permission
        if side == "sell" and current_position - desired_shares < 0:
            if not cfg.allow_short_selling:
                # Can only sell what we own
                desired_shares = min(desired_shares, max(0, current_position))
                if desired_shares < 1e-9:
                    return FillResult(False, 0, 0, 0, 0, 0, 0,
                                     reason="short selling disabled")

        # Partial fill check
        partial = False
        fill_qty = desired_shares
        if not cfg.slippage.can_fill_fully(desired_shares, avg_volume):
            fill_qty = abs(cfg.slippage.partial_fill_qty(desired_shares,
                                                         avg_volume))
            partial = True

        # Spread-adjusted fill price
        fill_price = cfg.spread.adjust_fill_price(price, daily_vol, side)

        # Slippage
        slippage_per_share = cfg.slippage.compute(price, fill_qty,
                                                   daily_vol, avg_volume)
        if side == "buy":
            fill_price += slippage_per_share
        else:
            fill_price -= slippage_per_share
            fill_price = max(fill_price, price * 0.5)  # floor at 50% of price

        # Cash constraint for buys
        if side == "buy":
            usable_cash = available_cash * (1 - cfg.cash_reserve_pct)
            max_affordable = usable_cash / fill_price if fill_price > 0 else 0
            if fill_qty > max_affordable:
                fill_qty = max_affordable
                partial = True

        # Leverage constraint
        # gross_exposure = abs(current_position * price) + fill_qty * fill_price
        # Not enforced here — checked at portfolio level

        # Commission
        commission = cfg.commission.compute(fill_qty, fill_price)

        # Cost breakdown
        spread_cost = abs(cfg.spread.estimate_spread(price, daily_vol)) * fill_qty
        slippage_cost = slippage_per_share * fill_qty
        total_cost = commission + spread_cost + slippage_cost

        if fill_qty < 1e-9:
            return FillResult(False, 0, 0, 0, 0, 0, 0,
                              reason="insufficient cash or volume")

        return FillResult(
            filled=True,
            fill_price=round(fill_price, 6),
            fill_qty=round(fill_qty, 6),
            commission=round(commission, 4),
            slippage_cost=round(slippage_cost, 4),
            spread_cost=round(spread_cost, 4),
            total_cost=round(total_cost, 4),
            partial=partial,
        )

    def daily_borrow_cost(self, ticker: str, position: float,
                          price: float) -> float:
        """Compute daily borrow cost for a short position."""
        return self.config.borrow.daily_cost(ticker, position, price)


# ---------------------------------------------------------------------------
# Convenience factory for common configurations
# ---------------------------------------------------------------------------

def zero_cost_config() -> ExecutionConfig:
    """Perfect execution — for comparison with realistic models."""
    return ExecutionConfig(
        commission=CommissionModel(per_share=0, pct_of_notional=0,
                                  min_per_trade=0, max_pct_of_notional=1),
        slippage=SlippageModel(base_bps=0, volatility_factor=0,
                               volume_participation_limit=1.0),
        spread=SpreadModel(base_spread_bps=0, vol_multiplier=0),
        borrow=BorrowCostModel(general_annual_rate=0, hard_to_borrow_rate=0),
    )


def retail_config() -> ExecutionConfig:
    """Typical retail broker costs."""
    return ExecutionConfig(
        commission=CommissionModel(per_share=0.005, pct_of_notional=0.001,
                                  min_per_trade=1.0),
        slippage=SlippageModel(base_bps=2.0, volatility_factor=0.15),
        spread=SpreadModel(base_spread_bps=3.0, vol_multiplier=0.6),
    )


def institutional_config() -> ExecutionConfig:
    """Institutional-grade costs (lower commissions, tighter spreads)."""
    return ExecutionConfig(
        commission=CommissionModel(per_share=0.002, pct_of_notional=0.0005,
                                  min_per_trade=0.50),
        slippage=SlippageModel(base_bps=0.5, volatility_factor=0.08),
        spread=SpreadModel(base_spread_bps=1.0, vol_multiplier=0.3),
    )
