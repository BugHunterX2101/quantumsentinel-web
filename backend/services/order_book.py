"""QuantumSentinel — L2 Order Book.

In-memory limit order book with price-time priority, supporting the
paper exchange's queue-aware matching engine.

The book maintains sorted bid/ask sides with O(log n) insert/delete.
Each price level is a FIFO queue of resting orders, enabling realistic
queue-position tracking for paper-trading simulation.
"""

from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator

from backend.services.market_microstructure import (
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


# ---------------------------------------------------------------------------
# Order
# ---------------------------------------------------------------------------

@dataclass
class Order:
    """A paper-trading order with queue-position tracking."""
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

    # -- Mutable state (managed by matching engine) --
    status: OrderStatus = OrderStatus.SUBMITTED
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    queue_ahead: float = 0.0
    queue_ahead_at_entry: float = 0.0
    queue_ahead_peak: float = 0.0
    entered_book_at: float | None = None
    filled_at: float | None = None

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

    @property
    def time_in_queue(self) -> float | None:
        if self.entered_book_at is None:
            return None
        return time.time() - self.entered_book_at

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
        }


# ---------------------------------------------------------------------------
# Fill
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Fill:
    """A single fill event from the matching engine."""
    order_id: str
    fill_price: float
    fill_quantity: float
    timestamp: float
    is_partial: bool = False
    aggressor: bool = False

    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "fill_price": round(self.fill_price, 8),
            "fill_quantity": round(self.fill_quantity, 8),
            "timestamp": self.timestamp,
            "is_partial": self.is_partial,
        }


# ---------------------------------------------------------------------------
# Price Level Queue
# ---------------------------------------------------------------------------

@dataclass
class LevelQueue:
    """FIFO queue of orders at a single price level."""
    price: float
    orders: list[Order] = field(default_factory=list)

    @property
    def total_size(self) -> float:
        return sum(o.remaining_quantity for o in self.orders)

    def add(self, order: Order) -> None:
        self.orders.append(order)

    def remove(self, order_id: str) -> Order | None:
        for i, o in enumerate(self.orders):
            if o.order_id == order_id:
                return self.orders.pop(i)
        return None

    def is_empty(self) -> bool:
        return len(self.orders) == 0 or self.total_size <= 0


# ---------------------------------------------------------------------------
# Order Book
# ---------------------------------------------------------------------------

class OrderBook:
    """In-memory L2 limit order book with price-time priority.

    Bid side: sorted descending by price (best bid first).
    Ask side: sorted ascending by price (best ask first).

    Each price level maintains a FIFO queue of resting orders.
    """

    def __init__(self, symbol: str = ""):
        self.symbol = symbol
        self._bids: dict[float, LevelQueue] = {}  # price -> queue
        self._asks: dict[float, LevelQueue] = {}
        self._orders: dict[str, Order] = {}        # order_id -> order
        self._last_trade_price: float | None = None
        self._last_trade_size: float | None = None

    # ---- Sorted level accessors -----------------------------------------

    @property
    def bid_levels(self) -> list[LevelQueue]:
        """Bid levels sorted best-first (descending price)."""
        return sorted(
            (q for q in self._bids.values() if not q.is_empty()),
            key=lambda q: -q.price,
        )

    @property
    def ask_levels(self) -> list[LevelQueue]:
        """Ask levels sorted best-first (ascending price)."""
        return sorted(
            (q for q in self._asks.values() if not q.is_empty()),
            key=lambda q: q.price,
        )

    @property
    def best_bid(self) -> float | None:
        levels = self.bid_levels
        return levels[0].price if levels else None

    @property
    def best_ask(self) -> float | None:
        levels = self.ask_levels
        return levels[0].price if levels else None

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

    # ---- Order management -----------------------------------------------

    def add_order(self, order: Order, timestamp: float | None = None) -> None:
        """Add a resting order to the book.

        ``timestamp``, when given, should be on the same clock as the
        caller's other order timestamps (e.g. simulated replay time during
        a backtest) so that later fill-time analytics aren't comparing
        wall-clock and simulated time. Defaults to wall-clock for callers
        that don't track their own clock.
        """
        side_book = self._bids if order.side == TradeSide.BUY else self._asks
        price = order.limit_price
        if price is None:
            return  # Market orders don't rest

        if price not in side_book:
            side_book[price] = LevelQueue(price=price)

        # Calculate queue-ahead (total resting volume at this price before us)
        queue_ahead = side_book[price].total_size
        order.queue_ahead = queue_ahead
        order.queue_ahead_at_entry = queue_ahead
        order.queue_ahead_peak = queue_ahead
        order.entered_book_at = timestamp if timestamp is not None else time.time()
        # A marketable limit order can partially fill before resting the
        # remainder — don't stomp that PARTIALLY_FILLED status with QUEUED.
        if order.filled_quantity <= 0:
            order.status = OrderStatus.QUEUED

        side_book[price].add(order)
        self._orders[order.order_id] = order

    def cancel_order(self, order_id: str) -> Order | None:
        """Cancel a resting order and remove from book."""
        order = self._orders.get(order_id)
        if order is None or not order.is_active:
            return None

        side_book = self._bids if order.side == TradeSide.BUY else self._asks
        price = order.limit_price
        if price and price in side_book:
            side_book[price].remove(order_id)
            if side_book[price].is_empty():
                del side_book[price]

        order.status = OrderStatus.CANCELLED
        return order

    def get_order(self, order_id: str) -> Order | None:
        return self._orders.get(order_id)

    # ---- Book consumption (for market orders) ---------------------------

    def consume_liquidity(
        self,
        side: TradeSide,
        quantity: float,
        timestamp: float,
        limit_price: float | None = None,
    ) -> list[Fill]:
        """Consume liquidity from the opposite side of the book.

        For a BUY aggressor, consume from asks (ascending price).
        For a SELL aggressor, consume from bids (descending price).

        ``limit_price``, when given, bounds the sweep: a BUY will never
        consume a level priced above it, and a SELL will never consume a
        level priced below it (a limit order must never trade through its
        own limit). Leave it ``None`` for an unbounded market-order sweep.

        Returns a list of fills generated.
        """
        levels = self.ask_levels if side == TradeSide.BUY else self.bid_levels
        fills: list[Fill] = []
        remaining = quantity

        for level in levels:
            if remaining <= 0:
                break
            if limit_price is not None:
                if side == TradeSide.BUY and level.price > limit_price:
                    break
                if side == TradeSide.SELL and level.price < limit_price:
                    break

            orders_to_remove = []
            for order in level.orders:
                if remaining <= 0:
                    break

                fill_qty = min(remaining, order.remaining_quantity)
                order.filled_quantity += fill_qty
                remaining -= fill_qty

                is_fully_filled = order.remaining_quantity <= 0
                if is_fully_filled:
                    order.status = OrderStatus.FILLED
                    orders_to_remove.append(order.order_id)
                else:
                    order.status = OrderStatus.PARTIALLY_FILLED
                order.filled_at = timestamp

                order.avg_fill_price = (
                    (order.avg_fill_price * (order.filled_quantity - fill_qty) +
                     level.price * fill_qty) / order.filled_quantity
                )

                fills.append(Fill(
                    order_id=order.order_id,
                    fill_price=level.price,
                    fill_quantity=fill_qty,
                    timestamp=timestamp,
                    is_partial=not is_fully_filled,
                ))

            # Clean up fully filled orders
            for oid in orders_to_remove:
                level.remove(oid)

        # Clean up empty levels
        side_book = self._asks if side == TradeSide.BUY else self._bids
        empty_prices = [p for p, q in side_book.items() if q.is_empty()]
        for p in empty_prices:
            del side_book[p]

        if fills:
            self._last_trade_price = fills[-1].fill_price
            self._last_trade_size = sum(f.fill_quantity for f in fills)

        return fills

    # ---- Queue updates (when external trades occur at a price level) ----

    def update_queue_positions(
        self,
        price: float,
        executed_volume: float,
        side: TradeSide,
        timestamp: float | None = None,
    ) -> list[Fill]:
        """Update queue positions when an external trade occurs at a level.

        When an execution event arrives at a price on our side, we
        decrement ``queue_ahead`` for each resting order.  If queue_ahead
        drops to zero, the order fills.

        ``timestamp`` should be on the same clock the caller used for
        ``entered_book_at`` (see ``add_order``), so fill-time analytics
        aren't comparing wall-clock and simulated time. Defaults to
        wall-clock when not given.
        """
        side_book = self._bids if side == TradeSide.BUY else self._asks
        if price not in side_book:
            return []

        level = side_book[price]
        fills: list[Fill] = []
        remaining_exec = executed_volume
        ts = timestamp if timestamp is not None else time.time()

        for order in list(level.orders):
            if remaining_exec <= 0:
                break

            if order.queue_ahead > 0:
                deducted = min(order.queue_ahead, remaining_exec)
                order.queue_ahead -= deducted
                remaining_exec -= deducted

                if order.queue_ahead <= 0:
                    # Our order is at the front. Only the trade volume left
                    # over *after* clearing the phantom queue ahead of us
                    # can fill us — adding `deducted` back here would count
                    # that volume twice (once against the queue, once
                    # against our own order), fabricating fills beyond what
                    # the trade actually executed.
                    fill_qty = min(order.remaining_quantity, remaining_exec)
                    if fill_qty <= 0:
                        # Queue exactly cleared with no leftover volume —
                        # we're now at the front but not yet executed.
                        continue

                    order.filled_quantity += fill_qty
                    remaining_exec -= fill_qty
                    is_full = order.remaining_quantity <= 0
                    order.status = OrderStatus.FILLED if is_full else OrderStatus.PARTIALLY_FILLED
                    order.filled_at = ts

                    order.avg_fill_price = (
                        (order.avg_fill_price * (order.filled_quantity - fill_qty) +
                         price * fill_qty) / order.filled_quantity
                        if order.filled_quantity > 0 else price
                    )

                    fills.append(Fill(
                        order_id=order.order_id,
                        fill_price=price,
                        fill_quantity=fill_qty,
                        timestamp=ts,
                        is_partial=not is_full,
                    ))

                    if is_full:
                        level.remove(order.order_id)
            else:
                # Already at front, this trade fills us
                fill_qty = min(order.remaining_quantity, remaining_exec)
                order.filled_quantity += fill_qty
                remaining_exec -= fill_qty

                is_full = order.remaining_quantity <= 0
                order.status = OrderStatus.FILLED if is_full else OrderStatus.PARTIALLY_FILLED
                order.avg_fill_price = price
                order.filled_at = ts

                fills.append(Fill(
                    order_id=order.order_id,
                    fill_price=price,
                    fill_quantity=fill_qty,
                    timestamp=ts,
                    is_partial=not is_full,
                ))

                if is_full:
                    level.remove(order.order_id)

        # Update peak queue for remaining orders
        for order in level.orders:
            order.queue_ahead_peak = max(order.queue_ahead_peak, order.queue_ahead)

        if level.is_empty():
            del side_book[price]

        return fills

    # ---- Snapshot -------------------------------------------------------

    def snapshot(self, levels: int = 10) -> OrderBookSnapshot:
        """Generate an ``OrderBookSnapshot`` of the current book state."""
        bids = [
            PriceLevel(q.price, q.total_size)
            for q in self.bid_levels[:levels]
        ]
        asks = [
            PriceLevel(q.price, q.total_size)
            for q in self.ask_levels[:levels]
        ]
        return OrderBookSnapshot(
            timestamp=time.time(),
            bids=bids,
            asks=asks,
            last_trade_price=self._last_trade_price,
            last_trade_size=self._last_trade_size,
        )

    def to_dict(self) -> dict:
        """JSON-serialisable representation of the book."""
        from backend.services.market_microstructure import snapshot_to_dict
        snap = self.snapshot()
        d = snapshot_to_dict(snap)
        d["symbol"] = self.symbol
        d["total_bid_orders"] = sum(len(q.orders) for q in self._bids.values())
        d["total_ask_orders"] = sum(len(q.orders) for q in self._asks.values())
        return d
