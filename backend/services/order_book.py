"""QuantumSentinel — L2 Order Book.

In-memory limit order book with price-time priority for the research paper
exchange.

Design
------
* **Integer ticks.** Price levels are keyed by ``round(price / tick_size)``,
  so 100.1, 100.10 and 100.100000001 are the same level. Prices are exposed
  as ``ticks * tick_size`` rounded to the tick's decimal places.
* **Explicit ownership.** Every resting order is either ``MARKET`` liquidity
  (replayed from the L2 feed, or seeded) or a ``PAPER`` order (ours). Both
  sit in one FIFO queue per level, in arrival order, which is what gives a
  paper order a meaningful queue position.
* **Queue position is derived, not decremented.** ``queue_ahead`` is the
  remaining size of everything in front of the order in its level's FIFO,
  recomputed whenever the level changes. Fills or cancels of orders ahead
  therefore advance every order behind them.
* **Full event model.** ``apply_market_event`` handles ADD, CANCEL, MODIFY
  and TRADE, so replaying a feed evolves the book like the venue's book.
"""

from __future__ import annotations

import bisect
import time
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum

from backend.services.market_microstructure import (
    BookEvent,
    BookEventType,
    OrderBookSnapshot,
    PriceLevel,
    TradeSide,
)


# ---------------------------------------------------------------------------
# Order Types & Status
# ---------------------------------------------------------------------------

class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    SUBMITTED = "SUBMITTED"
    ACCEPTED = "ACCEPTED"
    QUEUED = "QUEUED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class TimeInForce(str, Enum):
    DAY = "DAY"
    GTC = "GTC"       # Good 'til cancelled
    IOC = "IOC"       # Immediate or cancel
    FOK = "FOK"       # Fill or kill


class Owner(str, Enum):
    MARKET = "MARKET"   # liquidity from the replayed feed (or seeded)
    PAPER = "PAPER"     # our simulated orders


_EPS = 1e-9


# ---------------------------------------------------------------------------
# Order
# ---------------------------------------------------------------------------

@dataclass
class Order:
    """A resting or working order with queue-position tracking."""
    order_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    symbol: str = ""
    side: TradeSide = TradeSide.BUY
    order_type: OrderType = OrderType.LIMIT
    quantity: float = 0.0
    limit_price: float | None = None
    stop_price: float | None = None
    time_in_force: TimeInForce = TimeInForce.GTC
    submitted_at: float = field(default_factory=time.time)
    expires_at: float | None = None
    owner: Owner = Owner.MARKET

    # -- Mutable state (managed by the book / matching engine) --
    status: OrderStatus = OrderStatus.SUBMITTED
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    queue_ahead: float = 0.0
    queue_ahead_at_entry: float = 0.0
    queue_ahead_peak: float = 0.0
    entered_book_at: float | None = None
    filled_at: float | None = None
    active_at: float | None = None   # when the order reaches the exchange (latency)

    @property
    def remaining_quantity(self) -> float:
        return self.quantity - self.filled_quantity

    @property
    def is_active(self) -> bool:
        return self.status in (
            OrderStatus.ACCEPTED,
            OrderStatus.QUEUED,
            OrderStatus.PARTIALLY_FILLED,
        )

    def record_fill(self, price: float, quantity: float, timestamp: float) -> None:
        """Apply one execution: volume-weighted average price and status."""
        previous = self.filled_quantity
        self.filled_quantity = previous + quantity
        self.avg_fill_price = (self.avg_fill_price * previous + price * quantity) / self.filled_quantity
        self.filled_at = timestamp
        self.status = (OrderStatus.FILLED if self.remaining_quantity <= _EPS
                       else OrderStatus.PARTIALLY_FILLED)

    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": self.quantity,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "time_in_force": self.time_in_force.value,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "remaining_quantity": self.remaining_quantity,
            "avg_fill_price": round(self.avg_fill_price, 8) if self.avg_fill_price else 0,
            "queue_ahead": self.queue_ahead,
            "queue_ahead_at_entry": self.queue_ahead_at_entry,
            "submitted_at": self.submitted_at,
            "active_at": self.active_at,
        }


# ---------------------------------------------------------------------------
# Fill
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Fill:
    """A single execution of one order."""
    order_id: str
    fill_price: float
    fill_quantity: float
    timestamp: float
    is_partial: bool = False
    aggressor: bool = False
    side: TradeSide | None = None

    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "fill_price": round(self.fill_price, 8),
            "fill_quantity": round(self.fill_quantity, 8),
            "timestamp": self.timestamp,
            "is_partial": self.is_partial,
            "liquidity": "taker" if self.aggressor else "maker",
            "side": self.side.value if self.side else None,
        }


# ---------------------------------------------------------------------------
# Price Level Queue
# ---------------------------------------------------------------------------

@dataclass
class LevelQueue:
    """FIFO queue of orders (market and paper) at one price level."""
    price: float
    ticks: int = 0
    orders: list[Order] = field(default_factory=list)

    @property
    def total_size(self) -> float:
        return sum(o.remaining_quantity for o in self.orders)

    @property
    def market_size(self) -> float:
        return sum(o.remaining_quantity for o in self.orders if o.owner == Owner.MARKET)

    def add(self, order: Order) -> None:
        self.orders.append(order)

    def remove(self, order_id: str) -> Order | None:
        for i, o in enumerate(self.orders):
            if o.order_id == order_id:
                return self.orders.pop(i)
        return None

    def is_empty(self) -> bool:
        return self.total_size <= _EPS

    def refresh_queue_positions(self) -> None:
        """queue_ahead of each paper order = remaining size in front of it."""
        ahead = 0.0
        for o in self.orders:
            if o.owner == Owner.PAPER:
                o.queue_ahead = ahead
                o.queue_ahead_peak = max(o.queue_ahead_peak, ahead)
            ahead += o.remaining_quantity


# ---------------------------------------------------------------------------
# Order Book
# ---------------------------------------------------------------------------

class OrderBook:
    """L2 limit order book with price-time priority on an integer tick grid."""

    def __init__(self, symbol: str = "", tick_size: float = 0.01):
        if tick_size <= 0:
            raise ValueError("tick_size must be positive")
        self.symbol = symbol
        self.tick_size = tick_size
        self._decimals = max(0, -Decimal(str(tick_size)).normalize().as_tuple().exponent)
        self._bids: dict[int, LevelQueue] = {}
        self._asks: dict[int, LevelQueue] = {}
        self._bid_ticks: list[int] = []   # ascending; best bid is last
        self._ask_ticks: list[int] = []   # ascending; best ask is first
        self._orders: dict[str, Order] = {}
        self._last_trade_price: float | None = None
        self._last_trade_size: float | None = None
        self._last_event_time: float = 0.0

    # ---- Tick grid ------------------------------------------------------

    def to_ticks(self, price: float) -> int:
        return int(round(price / self.tick_size))

    def from_ticks(self, ticks: int) -> float:
        return round(ticks * self.tick_size, self._decimals)

    def is_on_grid(self, price: float) -> bool:
        units = price / self.tick_size
        return abs(units - round(units)) < 1e-6

    # ---- Level bookkeeping ----------------------------------------------

    def _side(self, side: TradeSide) -> tuple[dict[int, LevelQueue], list[int]]:
        return (self._bids, self._bid_ticks) if side == TradeSide.BUY else (self._asks, self._ask_ticks)

    def _level(self, side: TradeSide, ticks: int, create: bool = False) -> LevelQueue | None:
        levels, keys = self._side(side)
        level = levels.get(ticks)
        if level is None and create:
            level = LevelQueue(price=self.from_ticks(ticks), ticks=ticks)
            levels[ticks] = level
            bisect.insort(keys, ticks)
        return level

    def _prune(self, side: TradeSide, ticks: int) -> None:
        """Drop exhausted orders from a level, refresh queues, drop empty level."""
        levels, keys = self._side(side)
        level = levels.get(ticks)
        if level is None:
            return
        kept = []
        for o in level.orders:
            if o.remaining_quantity > _EPS and o.is_active:
                kept.append(o)
            elif o.owner == Owner.MARKET:
                # Paper orders stay addressable after completion; exhausted
                # feed orders are dropped so long replays don't grow memory.
                self._orders.pop(o.order_id, None)
        level.orders = kept
        level.refresh_queue_positions()
        if not level.orders:
            del levels[ticks]
            idx = bisect.bisect_left(keys, ticks)
            if idx < len(keys) and keys[idx] == ticks:
                keys.pop(idx)

    # ---- Sorted level accessors -----------------------------------------

    @property
    def bid_levels(self) -> list[LevelQueue]:
        """Bid levels best-first (descending price)."""
        return [self._bids[t] for t in reversed(self._bid_ticks)]

    @property
    def ask_levels(self) -> list[LevelQueue]:
        """Ask levels best-first (ascending price)."""
        return [self._asks[t] for t in self._ask_ticks]

    @property
    def best_bid(self) -> float | None:
        return self.from_ticks(self._bid_ticks[-1]) if self._bid_ticks else None

    @property
    def best_ask(self) -> float | None:
        return self.from_ticks(self._ask_ticks[0]) if self._ask_ticks else None

    @property
    def mid_price(self) -> float | None:
        bb, ba = self.best_bid, self.best_ask
        if bb is None or ba is None:
            return None
        return (bb + ba) / 2.0

    @property
    def spread(self) -> float | None:
        bb, ba = self.best_bid, self.best_ask
        if bb is None or ba is None:
            return None
        return ba - bb

    # ---- Resting orders -------------------------------------------------

    def add_order(self, order: Order, timestamp: float | None = None) -> None:
        """Rest an order at the back of its price level's FIFO.

        ``timestamp`` should be on the caller's clock (simulated time during
        a replay) so later fill-time analytics compare like with like.
        """
        if order.limit_price is None:
            return  # market orders never rest
        ticks = self.to_ticks(order.limit_price)
        if order.owner == Owner.MARKET:
            order.limit_price = self.from_ticks(ticks)  # feed noise snaps to the grid
        level = self._level(order.side, ticks, create=True)
        ahead = level.total_size
        order.queue_ahead = ahead
        order.queue_ahead_at_entry = ahead
        order.queue_ahead_peak = ahead
        order.entered_book_at = timestamp if timestamp is not None else time.time()
        # A marketable limit that partially filled before resting keeps
        # its PARTIALLY_FILLED status.
        if order.filled_quantity <= 0:
            order.status = OrderStatus.QUEUED
        level.add(order)
        self._orders[order.order_id] = order

    def cancel_order(self, order_id: str) -> Order | None:
        """Cancel a resting order and remove it from its level."""
        order = self._orders.get(order_id)
        if order is None or not order.is_active or order.limit_price is None:
            return None
        ticks = self.to_ticks(order.limit_price)
        level = self._level(order.side, ticks)
        order.status = OrderStatus.CANCELLED
        if level is not None:
            level.remove(order_id)
            self._prune(order.side, ticks)
        return order

    def get_order(self, order_id: str) -> Order | None:
        return self._orders.get(order_id)

    # ---- Aggressive execution (our marketable orders) -------------------

    def consume_liquidity(
        self,
        side: TradeSide,
        quantity: float,
        timestamp: float,
        limit_price: float | None = None,
        skip_paper: bool = True,
    ) -> list[Fill]:
        """Execute an aggressor of ``side`` against the opposite side.

        Walks levels best-first and FIFO within each level. ``limit_price``
        bounds the sweep (a limit order never trades through its limit).
        ``skip_paper`` prevents our own resting paper orders from being
        matched by our own aggressor (self-trade prevention).

        Returns one Fill per resting order touched, priced at its level; the
        fills' ``order_id`` is the *resting* order's id.
        """
        opposite = TradeSide.SELL if side == TradeSide.BUY else TradeSide.BUY
        levels, keys = self._side(opposite)
        ordered = list(keys) if side == TradeSide.BUY else list(reversed(keys))
        limit_ticks = self.to_ticks(limit_price) if limit_price is not None else None
        fills: list[Fill] = []
        remaining = quantity

        for ticks in ordered:
            if remaining <= _EPS:
                break
            if limit_ticks is not None:
                if side == TradeSide.BUY and ticks > limit_ticks:
                    break
                if side == TradeSide.SELL and ticks < limit_ticks:
                    break
            level = levels[ticks]
            for resting in list(level.orders):
                if remaining <= _EPS:
                    break
                if skip_paper and resting.owner == Owner.PAPER:
                    continue
                qty = min(remaining, resting.remaining_quantity)
                if qty <= _EPS:
                    continue
                resting.record_fill(level.price, qty, timestamp)
                remaining -= qty
                fills.append(Fill(order_id=resting.order_id, fill_price=level.price,
                                  fill_quantity=qty, timestamp=timestamp,
                                  is_partial=resting.status != OrderStatus.FILLED,
                                  side=resting.side))
            self._prune(opposite, ticks)

        if fills:
            self._last_trade_price = fills[-1].fill_price
            self._last_trade_size = sum(f.fill_quantity for f in fills)
        return fills

    # ---- Market data (L2 feed) ------------------------------------------

    def market_depth(self, side: TradeSide, price: float) -> float:
        level = self._level(side, self.to_ticks(price))
        return level.market_size if level else 0.0

    def apply_market_event(self, event: BookEvent) -> list[Fill]:
        """Apply one L2 feed event to the book.

        ADD / CANCEL / MODIFY change market liquidity on ``event.side``
        (the side of the resting order). A CANCEL or MODIFY that names a
        known order changes that order; otherwise (aggregated L2) the change
        is spread over the level's market orders in proportion to their
        size — the unbiased estimate of where in the queue it happened.

        TRADE is an aggressor of ``event.side`` printing ``event.size`` at
        ``event.price``. It consumes the passive side FIFO, best level first,
        through every level at or better than the print. Resting paper
        orders in its path are filled at their own price: an order priced
        through the print would have had priority, so it fills too.

        Returns the fills of resting paper orders caused by the event.
        """
        self._last_event_time = event.timestamp
        ticks = self.to_ticks(event.price)
        et = event.event_type

        if et == BookEventType.ADD:
            if event.size > _EPS:
                self.add_order(Order(
                    order_id=event.order_id or uuid.uuid4().hex[:16], symbol=self.symbol,
                    side=event.side, order_type=OrderType.LIMIT, quantity=event.size,
                    limit_price=self.from_ticks(ticks), submitted_at=event.timestamp,
                    owner=Owner.MARKET,
                ), timestamp=event.timestamp)
            return []

        if et in (BookEventType.CANCEL, BookEventType.MODIFY):
            level = self._level(event.side, ticks)
            if level is None:
                if et == BookEventType.MODIFY and event.size > _EPS:
                    return self.apply_market_event(BookEvent(event.timestamp, BookEventType.ADD,
                                                             event.side, event.price, event.size,
                                                             event.order_id))
                return []
            known = self._orders.get(event.order_id) if event.order_id else None
            if known is not None and known.owner == Owner.MARKET and any(o is known for o in level.orders):
                if et == BookEventType.CANCEL:
                    known.quantity = max(known.filled_quantity, known.quantity - event.size)
                elif event.size <= known.remaining_quantity:
                    known.quantity = known.filled_quantity + event.size   # decrease keeps priority
                else:
                    level.remove(known.order_id)                          # increase loses priority
                    known.quantity = known.filled_quantity + event.size
                    level.add(known)
            else:
                current = level.market_size
                target = max(0.0, current - event.size) if et == BookEventType.CANCEL else event.size
                if target < current:
                    self._reduce_market(level, current - target)
                elif target > current + _EPS:
                    self.add_order(Order(
                        order_id=event.order_id or uuid.uuid4().hex[:16], symbol=self.symbol,
                        side=event.side, order_type=OrderType.LIMIT, quantity=target - current,
                        limit_price=level.price, submitted_at=event.timestamp, owner=Owner.MARKET,
                    ), timestamp=event.timestamp)
            self._prune(event.side, ticks)
            return []

        if et == BookEventType.TRADE:
            self._last_trade_price = self.from_ticks(ticks)
            self._last_trade_size = event.size
            passive = TradeSide.SELL if event.side == TradeSide.BUY else TradeSide.BUY
            levels, keys = self._side(passive)
            if event.side == TradeSide.BUY:
                path = [t for t in keys if t <= ticks]
            else:
                path = [t for t in reversed(keys) if t >= ticks]
            remaining = event.size
            fills: list[Fill] = []
            for level_ticks in path:
                if remaining <= _EPS:
                    break
                level = levels[level_ticks]
                for resting in list(level.orders):
                    if remaining <= _EPS:
                        break
                    qty = min(remaining, resting.remaining_quantity)
                    if qty <= _EPS:
                        continue
                    remaining -= qty
                    if resting.owner == Owner.PAPER:
                        resting.record_fill(level.price, qty, event.timestamp)
                        fills.append(Fill(order_id=resting.order_id, fill_price=level.price,
                                          fill_quantity=qty, timestamp=event.timestamp,
                                          is_partial=resting.status != OrderStatus.FILLED,
                                          side=resting.side))
                    else:
                        resting.quantity -= qty
                self._prune(passive, level_ticks)
            return fills

        return []

    @staticmethod
    def _reduce_market(level: LevelQueue, amount: float) -> None:
        market = [o for o in level.orders if o.owner == Owner.MARKET and o.remaining_quantity > _EPS]
        total = sum(o.remaining_quantity for o in market)
        if total <= _EPS:
            return
        share = min(1.0, amount / total)
        for o in market:
            o.quantity -= o.remaining_quantity * share

    # ---- Snapshot -------------------------------------------------------

    def snapshot(self, levels: int = 10) -> OrderBookSnapshot:
        """Current book state (market and paper liquidity), best levels first."""
        bids = [PriceLevel(q.price, q.total_size) for q in self.bid_levels[:levels]]
        asks = [PriceLevel(q.price, q.total_size) for q in self.ask_levels[:levels]]
        return OrderBookSnapshot(
            timestamp=self._last_event_time,
            bids=bids,
            asks=asks,
            last_trade_price=self._last_trade_price,
            last_trade_size=self._last_trade_size,
        )

    def to_dict(self) -> dict:
        from backend.services.market_microstructure import snapshot_to_dict
        d = snapshot_to_dict(self.snapshot())
        d["symbol"] = self.symbol
        d["tick_size"] = self.tick_size
        d["total_bid_orders"] = sum(len(q.orders) for q in self._bids.values())
        d["total_ask_orders"] = sum(len(q.orders) for q in self._asks.values())
        return d
