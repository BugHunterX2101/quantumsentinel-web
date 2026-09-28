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
from decimal import Decimal
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
    tick_size: float = 0.01,
) -> list[BookEvent]:
    """Generate a synthetic, internally consistent L2 event stream from OHLCV bars.

    This is *synthetic* data for research when real L2 is unavailable — it
    does not claim tick-level fidelity to any venue. What it does guarantee
    is that the stream is a valid book history:

    * every price is on the ``tick_size`` grid;
    * the book is never crossed or locked (best bid < best ask);
    * CANCEL / MODIFY target a live level and never remove more than rests;
    * each TRADE executes against the current best opposite level and never
      exceeds its size;
    * the book mid follows the bar's open→close path (aggressor side and
      quote placement lean toward it), staying within the bar's range.

    MODIFY carries the level's new total size (aggregated-L2 semantics).
    Exactly ``events_per_bar`` events are emitted per bar.
    """
    rng = random.Random(seed)
    events: list[BookEvent] = []
    book: dict[TradeSide, dict[int, float]] = {TradeSide.BUY: {}, TradeSide.SELL: {}}
    decimals = max(0, -Decimal(str(tick_size)).normalize().as_tuple().exponent)
    seq = 0

    def to_ticks(price: float) -> int:
        return int(round(price / tick_size))

    def px(ticks: int) -> float:
        return round(ticks * tick_size, decimals)

    def depth_size() -> float:
        return round(max(1.0, rng.paretovariate(1.5) * base_depth * 0.1), 2)

    for bar in ohlcv_bars:
        ts_start = bar["timestamp"]
        bar_open, bar_close = bar["open"], bar["close"]
        bar_high, bar_low = bar["high"], bar["low"]
        bar_volume = bar.get("volume", 10_000)

        bar_range = (bar_high - bar_low) / bar_open if bar_open else 0
        vol_scale = max(0.5, min(3.0, bar_range / 0.02))
        spread_ticks = max(1, to_ticks(bar_open * base_spread_bps / 10_000.0 * vol_scale))
        low_t, high_t = to_ticks(bar_low), to_ticks(bar_high)
        trade_scale = max(1.0, bar_volume / max(events_per_bar, 1))

        for i in range(events_per_bar):
            frac = i / max(events_per_bar - 1, 1)
            ts = round(ts_start + frac * 86400.0 * 0.27, 6)   # ~6.5 trading hours
            path = bar_open + (bar_close - bar_open) * frac
            target = min(high_t, max(low_t, to_ticks(path) + int(round(rng.gauss(0, spread_ticks * 0.5)))))
            bids, asks = book[TradeSide.BUY], book[TradeSide.SELL]
            best_bid = max(bids) if bids else None
            best_ask = min(asks) if asks else None

            if best_bid is None or best_ask is None:
                # Replenish an empty side first, anchored so it cannot cross.
                side = TradeSide.BUY if best_bid is None else TradeSide.SELL
                half = max(1, spread_ticks // 2)
                if side == TradeSide.BUY:
                    ticks = target - half if best_ask is None else min(target - half, best_ask - 1)
                else:
                    ticks = target + half if best_bid is None else max(target + half, best_bid + 1)
                ticks = max(1, ticks)
                etype, size = BookEventType.ADD, depth_size()
                book[side][ticks] = book[side].get(ticks, 0.0) + size
            else:
                etype = rng.choices(
                    [BookEventType.ADD, BookEventType.CANCEL, BookEventType.MODIFY, BookEventType.TRADE],
                    weights=[40, 25, 15, 20],
                )[0]
                drift = target - (best_bid + best_ask) / 2.0
                lean = max(-0.45, min(0.45, drift / (2.0 * spread_ticks)))
                # When the mid lags the bar's path by more than a spread,
                # flow on the path's side becomes urgent: aggressors clear the
                # whole best level and quotes improve inside the spread.
                urgent = abs(drift) > spread_ticks
                toward = TradeSide.BUY if drift > 0 else TradeSide.SELL
                in_the_way = TradeSide.SELL if toward == TradeSide.BUY else TradeSide.BUY
                if etype == BookEventType.TRADE:
                    side = TradeSide.BUY if rng.random() < 0.5 + lean else TradeSide.SELL
                    passive = TradeSide.SELL if side == TradeSide.BUY else TradeSide.BUY
                    ticks = best_ask if side == TradeSide.BUY else best_bid
                    available = book[passive][ticks]
                    if urgent and side == toward:
                        size = round(available, 2)
                    else:
                        size = round(min(available, max(1.0, rng.expovariate(1.0 / trade_scale))), 2)
                    if len(book[passive]) == 1:
                        size = _keep_side_alive(size, available)
                    if size <= 0:
                        # Nothing tradable without emptying the side: replenish it instead.
                        etype, side, size = BookEventType.ADD, passive, depth_size()
                        book[passive][ticks] = available + size
                    else:
                        remaining = round(available - size, 2)
                        if remaining <= 0:
                            del book[passive][ticks]
                        else:
                            book[passive][ticks] = remaining
                elif etype == BookEventType.ADD:
                    if urgent and len(book[in_the_way]) == 1:
                        # Re-quote the blocking side near the path first, so its
                        # stale last level can be cleared without emptying it.
                        side = in_the_way
                    else:
                        side = TradeSide.BUY if rng.random() < 0.5 + lean else TradeSide.SELL
                    gap = best_ask - best_bid
                    offset = min(levels - 1, int(rng.expovariate(0.5)))
                    half = max(1, spread_ticks // 2)
                    if urgent and side == toward:
                        # Quote toward the path, as far as the far side allows.
                        ticks = (min(best_ask - 1, target - half) if side == TradeSide.BUY
                                 else max(best_bid + 1, target + half))
                        if (side == TradeSide.BUY and ticks <= best_bid) or (side == TradeSide.SELL and ticks >= best_ask):
                            ticks = best_bid if side == TradeSide.BUY else best_ask
                    elif urgent:
                        # The side in the way re-quotes at the path, not at its stale best.
                        ticks = (min(best_bid - offset, target - half) if side == TradeSide.BUY
                                 else max(best_ask + offset, target + half))
                    elif gap > 1 and rng.random() < 0.3:
                        # Improve the quote inside the spread, never crossing it.
                        ticks = best_bid + 1 if side == TradeSide.BUY else best_ask - 1
                    else:
                        ticks = best_bid - offset if side == TradeSide.BUY else best_ask + offset
                    ticks = max(1, ticks)
                    size = depth_size()
                    book[side][ticks] = book[side].get(ticks, 0.0) + size
                else:
                    if urgent and rng.random() < 0.5 + abs(lean):
                        side = in_the_way
                    else:
                        side = TradeSide.BUY if rng.random() < 0.5 else TradeSide.SELL
                    side_book = book[side]
                    choices = sorted(side_book)
                    if urgent and side == in_the_way:
                        ticks = max(choices) if side == TradeSide.BUY else min(choices)
                    else:
                        ticks = rng.choices(choices, weights=[side_book[t] for t in choices])[0]
                    level_size = side_book[ticks]
                    if etype == BookEventType.CANCEL:
                        low = 1.0 if urgent and side == in_the_way else 0.2
                        size = round(min(level_size, max(0.01, level_size * rng.uniform(low, 1.0))), 2)
                        if len(side_book) == 1:
                            size = _keep_side_alive(size, level_size)
                        remaining = round(level_size - size, 2)
                        if remaining <= 0:
                            del side_book[ticks]
                        else:
                            side_book[ticks] = remaining
                    else:
                        size = round(max(1.0, level_size * rng.uniform(0.5, 1.5)), 2)
                        side_book[ticks] = size

            events.append(BookEvent(
                timestamp=ts,
                event_type=etype,
                side=side,
                price=px(ticks),
                size=size,
                order_id=f"syn-{seed}-{seq}",
            ))
            seq += 1

    events.sort(key=lambda e: e.timestamp)
    return events


def _keep_side_alive(size: float, available: float) -> float:
    """Cap a trade/cancel so it never empties the last level on a side."""
    if size < available:
        return size
    return max(0.01, round(available / 2, 2)) if available > 0.02 else 0.0


class L2DataError(ValueError):
    """An L2 event stream violates a market-data invariant."""


def validate_events(events: Sequence[BookEvent], tick_size: float = 0.01,
                    max_issues: int = 20) -> list[str]:
    """Check an L2 stream against market-data invariants.

    * price is finite and > 0, size is finite and >= 0 (TRADE size > 0)
    * timestamps never decrease
    * ADD order ids are unique
    * after every event the book is neither crossed nor locked (bid < ask)

    Returns the list of violations (empty when the stream is valid).
    """
    from backend.services.order_book import OrderBook

    issues: list[str] = []
    book = OrderBook(tick_size=tick_size)
    seen_ids: set[str] = set()
    last_ts = -math.inf
    for i, ev in enumerate(events):
        if len(issues) >= max_issues:
            break
        if not (math.isfinite(ev.price) and ev.price > 0):
            issues.append(f"event {i}: price must be finite and positive (got {ev.price})")
            continue
        if not (math.isfinite(ev.size) and ev.size >= 0) or (ev.event_type == BookEventType.TRADE and ev.size <= 0):
            issues.append(f"event {i}: invalid size {ev.size} for {ev.event_type.value}")
            continue
        if ev.timestamp < last_ts:
            issues.append(f"event {i}: timestamp {ev.timestamp} is earlier than the previous event")
        last_ts = max(last_ts, ev.timestamp)
        if ev.event_type == BookEventType.ADD and ev.order_id:
            if ev.order_id in seen_ids:
                issues.append(f"event {i}: duplicate order id {ev.order_id}")
            seen_ids.add(ev.order_id)
        book.apply_market_event(ev)
        bb, ba = book.best_bid, book.best_ask
        if bb is not None and ba is not None and bb >= ba:
            issues.append(f"event {i}: book crossed or locked (bid {bb} >= ask {ba})")
    return issues


def build_snapshot_from_events(
    events: Sequence[BookEvent],
    levels: int = 10,
    tick_size: float = 0.01,
) -> OrderBookSnapshot:
    """Snapshot of the book after applying ``events`` in order.

    Uses the same ``OrderBook.apply_market_event`` semantics as the replay
    engine and paper exchange, so a snapshot here always matches what they
    would see after the same events.
    """
    from backend.services.order_book import OrderBook

    book = OrderBook(tick_size=tick_size)
    for ev in events:
        book.apply_market_event(ev)
    return book.snapshot(levels)


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
