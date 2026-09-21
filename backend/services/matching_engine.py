"""QuantumSentinel — Queue-Aware Matching Engine.

A paper-trading matching engine that processes incoming orders against
the order book with realistic queue-position tracking.

Key behaviours:
- Market orders consume liquidity across multiple price levels (VWAP fills)
- Limit orders rest on the book with tracked queue_ahead
- Stop orders activate when the trigger price is reached
- Stop-limit orders place a limit order upon trigger
- Partial fills, cancels, and expiry are all supported
- No live trading — 100% paper simulation
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from backend.services.market_microstructure import (
    BookEvent,
    BookEventType,
    OrderBookSnapshot,
    TradeSide,
)
from backend.services.order_book import (
    Fill,
    LevelQueue,
    Order,
    OrderBook,
    OrderStatus,
    OrderType,
    TimeInForce,
)


# ---------------------------------------------------------------------------
# Exchange Events
# ---------------------------------------------------------------------------

class ExchangeEventType(str, Enum):
    """Events emitted by the matching engine."""
    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    ORDER_ACCEPTED = "ORDER_ACCEPTED"
    ORDER_QUEUED = "ORDER_QUEUED"
    ORDER_PARTIALLY_FILLED = "ORDER_PARTIALLY_FILLED"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_CANCELLED = "ORDER_CANCELLED"
    ORDER_REJECTED = "ORDER_REJECTED"
    ORDER_EXPIRED = "ORDER_EXPIRED"
    QUOTE_UPDATE = "QUOTE_UPDATE"
    TRADE_EVENT = "TRADE_EVENT"
    BOOK_UPDATE = "BOOK_UPDATE"


@dataclass(slots=True)
class ExchangeEvent:
    """An event produced by the matching engine for audit/analytics."""
    timestamp: float
    event_type: ExchangeEventType
    order_id: str | None = None
    symbol: str = ""
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type.value,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Matching Engine
# ---------------------------------------------------------------------------

class MatchingEngine:
    """Queue-position-aware matching engine for paper trading.

    This engine processes orders against the ``OrderBook`` and handles:
    - Immediate execution of market orders (book consumption)
    - Limit order resting with queue-position tracking
    - Stop/stop-limit trigger monitoring
    - IOC and FOK time-in-force semantics
    - Order expiry
    """

    def __init__(self, book: OrderBook):
        self.book = book
        self.event_log: list[ExchangeEvent] = []
        self._pending_stops: list[Order] = []

    # ---- Order submission -----------------------------------------------

    def submit_order(self, order: Order, timestamp: float | None = None) -> list[ExchangeEvent]:
        """Process an incoming order.

        ``timestamp``, when given, is used for this order's events and its
        ``entered_book_at`` if it rests — pass the caller's simulated clock
        during a backtest/replay so it stays comparable to fill timestamps
        recorded via ``on_market_event``. Defaults to wall-clock.

        Returns a list of exchange events produced.
        """
        events: list[ExchangeEvent] = []
        now = timestamp if timestamp is not None else time.time()

        # Emit submission event
        events.append(ExchangeEvent(
            timestamp=now,
            event_type=ExchangeEventType.ORDER_SUBMITTED,
            order_id=order.order_id,
            symbol=order.symbol,
            details={"side": order.side.value, "type": order.order_type.value,
                      "quantity": order.quantity},
        ))

        # Validate
        if order.quantity <= 0:
            order.status = OrderStatus.REJECTED
            events.append(ExchangeEvent(
                timestamp=now,
                event_type=ExchangeEventType.ORDER_REJECTED,
                order_id=order.order_id,
                symbol=order.symbol,
                details={"reason": "quantity must be positive"},
            ))
            self.event_log.extend(events)
            return events

        if order.order_type == OrderType.LIMIT and order.limit_price is None:
            order.status = OrderStatus.REJECTED
            events.append(ExchangeEvent(
                timestamp=now,
                event_type=ExchangeEventType.ORDER_REJECTED,
                order_id=order.order_id,
                symbol=order.symbol,
                details={"reason": "limit order requires limit_price"},
            ))
            self.event_log.extend(events)
            return events

        if order.order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and order.stop_price is None:
            order.status = OrderStatus.REJECTED
            events.append(ExchangeEvent(
                timestamp=now,
                event_type=ExchangeEventType.ORDER_REJECTED,
                order_id=order.order_id,
                symbol=order.symbol,
                details={"reason": "stop order requires stop_price"},
            ))
            self.event_log.extend(events)
            return events

        order.status = OrderStatus.ACCEPTED
        events.append(ExchangeEvent(
            timestamp=now,
            event_type=ExchangeEventType.ORDER_ACCEPTED,
            order_id=order.order_id,
            symbol=order.symbol,
        ))

        # Route by order type
        if order.order_type == OrderType.MARKET:
            events.extend(self._execute_market_order(order, now))
        elif order.order_type == OrderType.LIMIT:
            events.extend(self._execute_limit_order(order, now))
        elif order.order_type in (OrderType.STOP, OrderType.STOP_LIMIT):
            self._pending_stops.append(order)
        
        self.event_log.extend(events)
        return events

    def cancel_order(self, order_id: str) -> list[ExchangeEvent]:
        """Cancel a resting or pending order."""
        events: list[ExchangeEvent] = []
        now = time.time()

        # Check pending stops
        for i, o in enumerate(self._pending_stops):
            if o.order_id == order_id:
                o.status = OrderStatus.CANCELLED
                self._pending_stops.pop(i)
                events.append(ExchangeEvent(
                    timestamp=now,
                    event_type=ExchangeEventType.ORDER_CANCELLED,
                    order_id=order_id,
                    symbol=o.symbol,
                ))
                self.event_log.extend(events)
                return events

        # Check book orders
        order = self.book.cancel_order(order_id)
        if order:
            events.append(ExchangeEvent(
                timestamp=now,
                event_type=ExchangeEventType.ORDER_CANCELLED,
                order_id=order_id,
                symbol=order.symbol,
                details={"filled_quantity": order.filled_quantity},
            ))
        self.event_log.extend(events)
        return events

    # ---- Market order execution ----------------------------------------

    def _execute_market_order(self, order: Order, now: float) -> list[ExchangeEvent]:
        """Execute a market order by consuming book liquidity."""
        events: list[ExchangeEvent] = []

        fills = self.book.consume_liquidity(
            side=order.side,
            quantity=order.quantity,
            timestamp=now,
        )

        for fill in fills:
            fill.order_id = order.order_id
            fill.aggressor = True
            order.filled_quantity += fill.fill_quantity
            order.avg_fill_price = (
                (order.avg_fill_price * (order.filled_quantity - fill.fill_quantity) +
                 fill.fill_price * fill.fill_quantity) / order.filled_quantity
                if order.filled_quantity > 0 else fill.fill_price
            )

        if order.filled_quantity >= order.quantity:
            order.status = OrderStatus.FILLED
            events.append(ExchangeEvent(
                timestamp=now,
                event_type=ExchangeEventType.ORDER_FILLED,
                order_id=order.order_id,
                symbol=order.symbol,
                details={"avg_price": round(order.avg_fill_price, 8),
                          "quantity": order.quantity},
            ))
        elif order.filled_quantity > 0:
            order.status = OrderStatus.PARTIALLY_FILLED
            events.append(ExchangeEvent(
                timestamp=now,
                event_type=ExchangeEventType.ORDER_PARTIALLY_FILLED,
                order_id=order.order_id,
                symbol=order.symbol,
                details={"filled": order.filled_quantity,
                          "remaining": order.remaining_quantity},
            ))
        else:
            # No liquidity available
            order.status = OrderStatus.REJECTED
            events.append(ExchangeEvent(
                timestamp=now,
                event_type=ExchangeEventType.ORDER_REJECTED,
                order_id=order.order_id,
                symbol=order.symbol,
                details={"reason": "insufficient liquidity"},
            ))

        return events

    # ---- Limit order execution -----------------------------------------

    def _execute_limit_order(self, order: Order, now: float) -> list[ExchangeEvent]:
        """Process a limit order: check for immediate cross, then rest."""
        events: list[ExchangeEvent] = []

        # Check for immediate cross (marketable limit)
        if self._is_marketable(order):
            fills = self.book.consume_liquidity(
                side=order.side,
                quantity=order.quantity,
                timestamp=now,
                limit_price=order.limit_price,
            )
            for fill in fills:
                fill.order_id = order.order_id
                fill.aggressor = True
                order.filled_quantity += fill.fill_quantity
                order.avg_fill_price = (
                    (order.avg_fill_price * (order.filled_quantity - fill.fill_quantity) +
                     fill.fill_price * fill.fill_quantity) / order.filled_quantity
                    if order.filled_quantity > 0 else fill.fill_price
                )

            if order.remaining_quantity <= 0:
                order.status = OrderStatus.FILLED
                events.append(ExchangeEvent(
                    timestamp=now,
                    event_type=ExchangeEventType.ORDER_FILLED,
                    order_id=order.order_id,
                    symbol=order.symbol,
                    details={"avg_price": round(order.avg_fill_price, 8)},
                ))
                return events

        # Handle IOC: cancel unfilled portion
        if order.time_in_force == TimeInForce.IOC:
            if order.filled_quantity > 0:
                order.status = OrderStatus.PARTIALLY_FILLED
                events.append(ExchangeEvent(
                    timestamp=now,
                    event_type=ExchangeEventType.ORDER_PARTIALLY_FILLED,
                    order_id=order.order_id,
                    symbol=order.symbol,
                    details={"filled": order.filled_quantity, "cancelled_remainder": True},
                ))
            else:
                order.status = OrderStatus.CANCELLED
                events.append(ExchangeEvent(
                    timestamp=now,
                    event_type=ExchangeEventType.ORDER_CANCELLED,
                    order_id=order.order_id,
                    symbol=order.symbol,
                    details={"reason": "IOC - no immediate fill"},
                ))
            return events

        # Handle FOK: must fill entirely or reject
        if order.time_in_force == TimeInForce.FOK:
            if order.filled_quantity < order.quantity:
                order.status = OrderStatus.REJECTED
                events.append(ExchangeEvent(
                    timestamp=now,
                    event_type=ExchangeEventType.ORDER_REJECTED,
                    order_id=order.order_id,
                    symbol=order.symbol,
                    details={"reason": "FOK - insufficient liquidity for full fill"},
                ))
                return events

        # Rest remaining quantity on the book
        if order.remaining_quantity > 0:
            if order.filled_quantity > 0:
                # Marketable limit that partially filled against the book
                # before its unfilled remainder rests — surface the fill
                # that already happened, not just the queue event.
                events.append(ExchangeEvent(
                    timestamp=now,
                    event_type=ExchangeEventType.ORDER_PARTIALLY_FILLED,
                    order_id=order.order_id,
                    symbol=order.symbol,
                    details={"filled": order.filled_quantity,
                              "remaining": order.remaining_quantity},
                ))
            self.book.add_order(order, timestamp=now)
            events.append(ExchangeEvent(
                timestamp=now,
                event_type=ExchangeEventType.ORDER_QUEUED,
                order_id=order.order_id,
                symbol=order.symbol,
                details={"price": order.limit_price,
                          "quantity": order.remaining_quantity,
                          "queue_ahead": order.queue_ahead},
            ))

        return events

    def _is_marketable(self, order: Order) -> bool:
        """Check if a limit order crosses the current book."""
        if order.side == TradeSide.BUY:
            best_ask = self.book.best_ask
            return best_ask is not None and order.limit_price is not None and order.limit_price >= best_ask
        else:
            best_bid = self.book.best_bid
            return best_bid is not None and order.limit_price is not None and order.limit_price <= best_bid

    # ---- Market event processing ---------------------------------------

    def on_market_event(self, event: BookEvent) -> list[ExchangeEvent]:
        """Process an incoming market data event.

        Updates the book, checks stop triggers, and processes queue fills.
        """
        exchange_events: list[ExchangeEvent] = []
        now = event.timestamp

        if event.event_type == BookEventType.TRADE:
            # Check stop triggers
            exchange_events.extend(self._check_stop_triggers(event.price, now))

            # Update queue positions for resting orders at the traded price
            # Trades on the ask side affect BUY orders; trades on bid side affect SELL orders
            if event.side == TradeSide.BUY:
                # Buyer aggressed → ask liquidity consumed → our SELL orders at this price may fill
                fills = self.book.update_queue_positions(event.price, event.size, TradeSide.SELL, timestamp=now)
            else:
                # Seller aggressed → bid liquidity consumed → our BUY orders may fill
                fills = self.book.update_queue_positions(event.price, event.size, TradeSide.BUY, timestamp=now)

            for fill in fills:
                etype = ExchangeEventType.ORDER_FILLED if not fill.is_partial else ExchangeEventType.ORDER_PARTIALLY_FILLED
                exchange_events.append(ExchangeEvent(
                    timestamp=now,
                    event_type=etype,
                    order_id=fill.order_id,
                    symbol=self.book.symbol,
                    details=fill.to_dict(),
                ))

        self.event_log.extend(exchange_events)
        return exchange_events

    def _check_stop_triggers(self, trade_price: float, now: float) -> list[ExchangeEvent]:
        """Check if any pending stop orders are triggered."""
        events: list[ExchangeEvent] = []
        triggered = []

        for i, order in enumerate(self._pending_stops):
            is_triggered = False
            if order.side == TradeSide.BUY and trade_price >= (order.stop_price or float("inf")):
                is_triggered = True
            elif order.side == TradeSide.SELL and trade_price <= (order.stop_price or 0):
                is_triggered = True

            if is_triggered:
                triggered.append(i)
                if order.order_type == OrderType.STOP:
                    # Convert to market order
                    order.order_type = OrderType.MARKET
                    events.extend(self._execute_market_order(order, now))
                elif order.order_type == OrderType.STOP_LIMIT:
                    # Convert to limit order
                    order.order_type = OrderType.LIMIT
                    events.extend(self._execute_limit_order(order, now))

        # Remove triggered orders (reverse order to preserve indices)
        for i in reversed(triggered):
            self._pending_stops.pop(i)

        return events

    # ---- Expiry check ---------------------------------------------------

    def expire_orders(self, current_time: float | None = None) -> list[ExchangeEvent]:
        """Expire orders past their expiry time or DAY orders at EOD."""
        if current_time is None:
            current_time = time.time()

        events: list[ExchangeEvent] = []
        to_cancel: list[str] = []

        for oid, order in self.book._orders.items():
            if not order.is_active:
                continue
            if order.expires_at and current_time >= order.expires_at:
                to_cancel.append(oid)

        for oid in to_cancel:
            order = self.book.cancel_order(oid)
            if order:
                order.status = OrderStatus.EXPIRED
                events.append(ExchangeEvent(
                    timestamp=current_time,
                    event_type=ExchangeEventType.ORDER_EXPIRED,
                    order_id=oid,
                    symbol=order.symbol,
                ))

        self.event_log.extend(events)
        return events
