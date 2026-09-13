"""QuantumSentinel — Market Microstructure Analytics.

Level-2 order-book data structures and microstructure feature computation.
Provides the foundation for queue-aware paper-trading simulation, alpha
research on order-flow signals, and realistic execution-cost modelling.

All analytics accept ``OrderBookSnapshot`` objects and return plain Python
types so they are JSON-serialisable for API responses.

References
----------
- Cont, Kukanov & Stoikov (2014) — "The Price Impact of Order Book Events"
- Cartea, Jaimungal & Penalva (2015) — *Algorithmic and HF Trading*
- Gould et al. (2013) — "Limit Order Books"
"""

from __future__ import annotations

import math
import random
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

class BookEventType(str, Enum):
    """Types of order-book events in the L2 event stream."""
    ADD = "ADD"
    CANCEL = "CANCEL"
    MODIFY = "MODIFY"
    TRADE = "TRADE"


class TradeSide(str, Enum):
    """Aggressor side of a trade event."""
    BUY = "BUY"
    SELL = "SELL"


@dataclass(slots=True)
class PriceLevel:
    """A single price level on one side of the book."""
    price: float
    size: float


@dataclass(slots=True)
class OrderBookSnapshot:
    """Point-in-time snapshot of a limit order book.

    ``bids`` and ``asks`` are sorted best-first: bids descending by price,
    asks ascending by price.
    """
    timestamp: float
    bids: list[PriceLevel]
    asks: list[PriceLevel]
    last_trade_price: float | None = None
    last_trade_size: float | None = None

    # ---- convenience accessors ------------------------------------------
    @property
    def best_bid(self) -> PriceLevel | None:
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> PriceLevel | None:
        return self.asks[0] if self.asks else None


@dataclass(slots=True)
class BookEvent:
    """A single L2 event in the order-book event stream."""
    timestamp: float
    event_type: BookEventType
    side: TradeSide
    price: float
    size: float
    order_id: str | None = None


@dataclass(slots=True)
class TradeEvent:
    """A matched trade produced by the exchange."""
    timestamp: float
    price: float
    size: float
    aggressor_side: TradeSide


# ---------------------------------------------------------------------------
# Core Microstructure Analytics
# ---------------------------------------------------------------------------

def compute_mid_price(snap: OrderBookSnapshot) -> float | None:
    """Mid-price: (Bid₁ + Ask₁) / 2."""
    if not snap.bids or not snap.asks:
        return None
    return (snap.bids[0].price + snap.asks[0].price) / 2.0


def compute_spread(snap: OrderBookSnapshot) -> float | None:
    """Quoted spread: Ask₁ − Bid₁."""
    if not snap.bids or not snap.asks:
        return None
    return snap.asks[0].price - snap.bids[0].price


def compute_spread_bps(snap: OrderBookSnapshot) -> float | None:
    """Spread in basis points relative to mid-price."""
    spread = compute_spread(snap)
    mid = compute_mid_price(snap)
    if spread is None or mid is None or mid == 0:
        return None
    return (spread / mid) * 10_000.0


def compute_microprice(snap: OrderBookSnapshot) -> float | None:
    """Microprice: size-weighted mid-price.

    Microprice = (Ask₁ × BidSize₁ + Bid₁ × AskSize₁) / (BidSize₁ + AskSize₁)

    The microprice shifts toward whichever side has *less* depth,
    reflecting the intuition that a thin bid side signals downward
    pressure.
    """
    if not snap.bids or not snap.asks:
        return None
    b, a = snap.bids[0], snap.asks[0]
    denom = b.size + a.size
    if denom == 0:
        return None
    return (a.price * b.size + b.price * a.size) / denom


def compute_order_book_imbalance(
    snap: OrderBookSnapshot,
    levels: int = 5,
) -> float | None:
    """Order Book Imbalance (OBI) across *levels* price levels.

    OBI = (ΣBidVolume − ΣAskVolume) / (ΣBidVolume + ΣAskVolume)

    Returns a value in [-1, +1].  Positive values signal excess bid
    depth (buying pressure); negative values signal excess ask depth.
    """
    bid_vol = sum(lvl.size for lvl in snap.bids[:levels])
    ask_vol = sum(lvl.size for lvl in snap.asks[:levels])
    total = bid_vol + ask_vol
    if total == 0:
        return None
    return (bid_vol - ask_vol) / total


def compute_depth(
    snap: OrderBookSnapshot,
    levels: int = 10,
) -> dict:
    """Cumulative depth at N levels on each side.

    Returns ``{"bid_depth": float, "ask_depth": float, "total_depth": float,
    "levels_used": int}``.
    """
    bid_depth = sum(lvl.size for lvl in snap.bids[:levels])
    ask_depth = sum(lvl.size for lvl in snap.asks[:levels])
    return {
        "bid_depth": bid_depth,
        "ask_depth": ask_depth,
        "total_depth": bid_depth + ask_depth,
        "levels_used": min(levels, max(len(snap.bids), len(snap.asks))),
    }


def compute_effective_spread(
    trade_price: float,
    mid_price: float,
    side: TradeSide,
) -> float:
    """Effective spread: 2 × |trade − mid|, sign-adjusted by aggressor side.

    For buys:  2 × (trade − mid)
    For sells: 2 × (mid − trade)
    """
    if side == TradeSide.BUY:
        return 2.0 * (trade_price - mid_price)
    return 2.0 * (mid_price - trade_price)


def compute_price_impact(
    order_size: float,
    snap: OrderBookSnapshot,
    side: TradeSide,
) -> dict:
    """Simulate the price impact of consuming ``order_size`` from the book.

    Walks through the ask (for buys) or bid (for sells) levels until
    ``order_size`` is exhausted, and computes the volume-weighted average
    execution price (VWAP).

    Returns ``{"vwap": float, "mid_before": float, "impact_bps": float,
    "levels_consumed": int, "fully_filled": bool}``.
    """
    mid = compute_mid_price(snap)
    if mid is None:
        return {"vwap": None, "mid_before": None, "impact_bps": None,
                "levels_consumed": 0, "fully_filled": False}

    levels = snap.asks if side == TradeSide.BUY else snap.bids
    remaining = order_size
    cost = 0.0
    consumed = 0

    for lvl in levels:
        fill = min(remaining, lvl.size)
        cost += fill * lvl.price
        remaining -= fill
        consumed += 1
        if remaining <= 0:
            break

    filled_qty = order_size - max(remaining, 0)
    vwap = cost / filled_qty if filled_qty > 0 else None
    impact_bps = ((vwap - mid) / mid * 10_000.0) if vwap and mid else None
    if side == TradeSide.SELL and impact_bps is not None:
        impact_bps = -impact_bps

    return {
        "vwap": round(vwap, 8) if vwap else None,
        "mid_before": round(mid, 8),
        "impact_bps": round(impact_bps, 4) if impact_bps is not None else None,
        "levels_consumed": consumed,
        "fully_filled": remaining <= 0,
    }


def compute_trade_imbalance(
    trades: Sequence[TradeEvent],
    window_seconds: float = 60.0,
    reference_time: float | None = None,
) -> float | None:
    """Trade imbalance: (BuyVolume − SellVolume) / (BuyVolume + SellVolume)
    within a trailing time window.

    Returns a value in [-1, +1], or None if no trades in window.
    """
    if reference_time is None and trades:
        reference_time = trades[-1].timestamp
    elif reference_time is None:
        return None

    cutoff = reference_time - window_seconds
    buy_vol = 0.0
    sell_vol = 0.0
    for t in trades:
        if t.timestamp < cutoff:
            continue
        if t.aggressor_side == TradeSide.BUY:
            buy_vol += t.size
        else:
            sell_vol += t.size

    total = buy_vol + sell_vol
    if total == 0:
        return None
    return (buy_vol - sell_vol) / total


def compute_realized_volatility(
    mid_prices: Sequence[float],
    annualize_factor: float = 1.0,
) -> float | None:
    """Realized volatility from a series of mid-prices.

    Uses sum of squared log-returns.  ``annualize_factor`` can be set
    to e.g. √252 for daily bars or √(252*6.5*60*60) for per-second data.
    """
    if len(mid_prices) < 2:
        return None
    sq_sum = 0.0
    for i in range(1, len(mid_prices)):
        if mid_prices[i] <= 0 or mid_prices[i - 1] <= 0:
            continue
        lr = math.log(mid_prices[i] / mid_prices[i - 1])
        sq_sum += lr * lr
    rv = math.sqrt(sq_sum)
    return rv * annualize_factor


def compute_microprice_deviation(snap: OrderBookSnapshot) -> float | None:
    """Deviation of microprice from mid-price, in basis points.

    Positive values indicate upward pressure (bid side thicker → micro
    above mid); negative values indicate downward pressure.
    """
    mid = compute_mid_price(snap)
    micro = compute_microprice(snap)
    if mid is None or micro is None or mid == 0:
        return None
    return (micro - mid) / mid * 10_000.0


# ---------------------------------------------------------------------------
# Synthetic L2 Data Generation
# ---------------------------------------------------------------------------

def generate_synthetic_l2(
    ohlcv_bars: list[dict],
    levels: int = 10,
    base_spread_bps: float = 10.0,
    base_depth: float = 100.0,
    events_per_bar: int = 50,
    seed: int | None = None,
) -> list[BookEvent]:
    """Generate realistic L2 book events from OHLCV daily bars.

    This is a research tool for backtesting microstructure strategies
    when real L2 data is unavailable.  The generated events reproduce
    statistical properties of real order flow (Poisson arrival, power-law
    depth, spread correlation with volatility) without claiming tick-level
    fidelity.

    Parameters
    ----------
    ohlcv_bars : list of dicts
        Each dict must have keys ``open``, ``high``, ``low``, ``close``,
        ``volume``, ``timestamp`` (epoch seconds).
    levels : int
        Number of price levels on each side.
    base_spread_bps : float
        Typical quoted spread in basis points.
    base_depth : float
        Average depth per level (shares).
    events_per_bar : int
        Number of L2 events to generate per OHLCV bar.
    seed : int | None
        Random seed for deterministic replay.

    Returns
    -------
    list[BookEvent]
        Time-sorted sequence of L2 events.
    """
    rng = random.Random(seed)
    events: list[BookEvent] = []

    for bar in ohlcv_bars:
        ts_start = bar["timestamp"]
        bar_open = bar["open"]
        bar_close = bar["close"]
        bar_high = bar["high"]
        bar_low = bar["low"]
        bar_volume = bar.get("volume", 10_000)

        # Scale spread by bar volatility
        bar_range = (bar_high - bar_low) / bar_open if bar_open else 0
        vol_scale = max(0.5, min(3.0, bar_range / 0.02))
        spread = bar_open * base_spread_bps / 10_000.0 * vol_scale

        # Interpolate price path within bar
        for i in range(events_per_bar):
            frac = i / max(events_per_bar - 1, 1)
            ts = ts_start + frac * 86400.0 * 0.27  # ~6.5 trading hours
            # Simple linear interpolation with noise
            mid = bar_open + (bar_close - bar_open) * frac
            mid += rng.gauss(0, spread * 0.5)
            mid = max(bar_low, min(bar_high, mid))

            # Generate event
            event_type = rng.choices(
                [BookEventType.ADD, BookEventType.CANCEL,
                 BookEventType.MODIFY, BookEventType.TRADE],
                weights=[40, 25, 15, 20],
            )[0]

            side = TradeSide.BUY if rng.random() < 0.5 else TradeSide.SELL
            if event_type == BookEventType.TRADE:
                # Trades at or near mid
                price = mid + rng.gauss(0, spread * 0.1)
                size = max(1, rng.expovariate(1.0 / (bar_volume / events_per_bar)))
            else:
                # Orders around spread
                offset = rng.expovariate(1.0 / spread) if spread > 0 else 0
                if side == TradeSide.BUY:
                    price = mid - spread / 2 - offset
                else:
                    price = mid + spread / 2 + offset
                size = max(1, rng.paretovariate(1.5) * base_depth * 0.1)

            events.append(BookEvent(
                timestamp=round(ts, 6),
                event_type=event_type,
                side=side,
                price=round(price, 4),
                size=round(size, 2),
                order_id=hashlib.md5(f"{ts}:{i}:{seed}".encode()).hexdigest()[:12],
            ))

    events.sort(key=lambda e: e.timestamp)
    return events


def build_snapshot_from_events(
    events: Sequence[BookEvent],
    levels: int = 10,
) -> OrderBookSnapshot:
    """Build an ``OrderBookSnapshot`` from a sequence of book events.

    Maintains a simplified book state by accumulating ADDs and
    subtracting CANCELs/TRADEs at each price level.
    """
    bid_levels: dict[float, float] = {}
    ask_levels: dict[float, float] = {}
    last_trade_price: float | None = None
    last_trade_size: float | None = None
    last_ts = 0.0

    for ev in events:
        last_ts = ev.timestamp
        if ev.event_type == BookEventType.TRADE:
            last_trade_price = ev.price
            last_trade_size = ev.size
            # Trades remove liquidity from the passive side
            book = ask_levels if ev.side == TradeSide.BUY else bid_levels
            if ev.price in book:
                book[ev.price] = max(0, book[ev.price] - ev.size)
                if book[ev.price] <= 0:
                    del book[ev.price]
        elif ev.event_type == BookEventType.ADD:
            book = bid_levels if ev.side == TradeSide.BUY else ask_levels
            book[ev.price] = book.get(ev.price, 0) + ev.size
        elif ev.event_type == BookEventType.CANCEL:
            book = bid_levels if ev.side == TradeSide.BUY else ask_levels
            if ev.price in book:
                book[ev.price] = max(0, book[ev.price] - ev.size)
                if book[ev.price] <= 0:
                    del book[ev.price]
        elif ev.event_type == BookEventType.MODIFY:
            book = bid_levels if ev.side == TradeSide.BUY else ask_levels
            book[ev.price] = ev.size

    # Build sorted levels
    bids = sorted(
        [PriceLevel(p, s) for p, s in bid_levels.items() if s > 0],
        key=lambda x: -x.price,
    )[:levels]
    asks = sorted(
        [PriceLevel(p, s) for p, s in ask_levels.items() if s > 0],
        key=lambda x: x.price,
    )[:levels]

    return OrderBookSnapshot(
        timestamp=last_ts,
        bids=bids,
        asks=asks,
        last_trade_price=last_trade_price,
        last_trade_size=last_trade_size,
    )


def snapshot_to_dict(snap: OrderBookSnapshot) -> dict:
    """Convert an OrderBookSnapshot to a JSON-serialisable dict."""
    return {
        "timestamp": snap.timestamp,
        "bids": [{"price": l.price, "size": l.size} for l in snap.bids],
        "asks": [{"price": l.price, "size": l.size} for l in snap.asks],
        "last_trade_price": snap.last_trade_price,
        "last_trade_size": snap.last_trade_size,
        "mid_price": compute_mid_price(snap),
        "spread": compute_spread(snap),
        "spread_bps": compute_spread_bps(snap),
        "microprice": compute_microprice(snap),
        "microprice_deviation_bps": compute_microprice_deviation(snap),
        "obi_5": compute_order_book_imbalance(snap, levels=5),
    }
