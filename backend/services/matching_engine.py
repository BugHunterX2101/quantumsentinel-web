"""QuantumSentinel — Queue-Aware Matching Engine (paper simulation only).

Routes paper orders against an ``OrderBook`` and processes the L2 feed.

Guarantees
----------
* **Every execution is its own event.** Each fill of a paper order emits
  exactly one ``ORDER_FILLED`` / ``ORDER_PARTIALLY_FILLED`` event whose
  details carry ``fill_price``, ``fill_quantity``, ``side`` and
  ``liquidity`` (taker/maker). Consumers (the paper exchange, analytics)
  settle fills from these fields and nothing else, so no fill can be lost.
* Market orders sweep the book; an unfilled remainder is cancelled.
* Limit orders take liquidity up to their limit, then rest (FIFO).
* IOC cancels any unfilled remainder; FOK checks available liquidity
  *before* executing and fills completely or not at all.
* Stop / stop-limit orders trigger on trade prints and then execute as
  market / limit orders.
* Paper orders never match other paper orders (self-trade prevention).
* Prices must lie on the book's tick grid.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from backend.services.market_microstructure import (
    BookEvent,
    BookEventType,
    TradeSide,
)
from backend.services.order_book import (
    Fill,
    Order,
    OrderBook,
    OrderStatus,
    OrderType,
    Owner,
    TimeInForce,
)

_EPS = 1e-9


class ExchangeEventType(str, Enum):
    """Events emitted by the matching engine."""
    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    ORDER_ACCEPTED = "ORDER_ACCEPTED"
    ORDER_QUEUED = "ORDER_QUEUED"
    ORDER_TRIGGERED = "ORDER_TRIGGERED"
    ORDER_PARTIALLY_FILLED = "ORDER_PARTIALLY_FILLED"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_CANCELLED = "ORDER_CANCELLED"
    ORDER_REJECTED = "ORDER_REJECTED"
    ORDER_EXPIRED = "ORDER_EXPIRED"
    QUOTE_UPDATE = "QUOTE_UPDATE"
    TRADE_EVENT = "TRADE_EVENT"
    BOOK_UPDATE = "BOOK_UPDATE"


FILL_EVENTS = (ExchangeEventType.ORDER_FILLED, ExchangeEventType.ORDER_PARTIALLY_FILLED)


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


class MatchingEngine:
    """Queue-position-aware matching engine for paper trading."""

    def __init__(self, book: OrderBook):
        self.book = book
        self.event_log: list[ExchangeEvent] = []
        self._pending_stops: list[Order] = []

    # ---- helpers --------------------------------------------------------

    def _event(self, now: float, etype: ExchangeEventType, order: Order, **details) -> ExchangeEvent:
        return ExchangeEvent(timestamp=now, event_type=etype, order_id=order.order_id,
                             symbol=order.symbol, details=details)

    def _fill_event(self, order: Order, price: float, qty: float, now: float, aggressor: bool) -> ExchangeEvent:
        etype = ExchangeEventType.ORDER_FILLED if order.status == OrderStatus.FILLED \
            else ExchangeEventType.ORDER_PARTIALLY_FILLED
        return self._event(
            now, etype, order,
            fill_price=price, fill_quantity=qty, side=order.side.value,
            liquidity="taker" if aggressor else "maker",
            avg_price=round(order.avg_fill_price, 8), quantity=order.quantity,
            cumulative_filled=order.filled_quantity, remaining=order.remaining_quantity,
        )

    def _reject(self, order: Order, now: float, reason: str) -> list[ExchangeEvent]:
        order.status = OrderStatus.REJECTED
        return [self._event(now, ExchangeEventType.ORDER_REJECTED, order, reason=reason)]

    def _take(self, order: Order, now: float, limit_price: float | None) -> list[ExchangeEvent]:
        """Execute ``order`` as an aggressor; one event per execution."""
        events = []
        for resting_fill in self.book.consume_liquidity(order.side, order.remaining_quantity, now,
                                                        limit_price=limit_price):
            order.record_fill(resting_fill.fill_price, resting_fill.fill_quantity, now)
            events.append(self._fill_event(order, resting_fill.fill_price,
                                           resting_fill.fill_quantity, now, aggressor=True))
        return events

    def _available(self, side: TradeSide, limit_price: float | None) -> float:
        """Market liquidity an aggressor of ``side`` could take up to ``limit_price``."""
        levels = self.book.ask_levels if side == TradeSide.BUY else self.book.bid_levels
        total = 0.0
        for level in levels:
            if limit_price is not None:
                if side == TradeSide.BUY and level.price > limit_price + _EPS:
                    break
                if side == TradeSide.SELL and level.price < limit_price - _EPS:
                    break
            total += level.market_size
        return total

    # ---- Order submission -----------------------------------------------

    def submit_order(self, order: Order, timestamp: float | None = None) -> list[ExchangeEvent]:
        """Process an incoming paper order; returns the events it produced.

        ``timestamp`` is the caller's simulation clock (wall clock if omitted).
        """
        now = timestamp if timestamp is not None else time.time()
        order.owner = Owner.PAPER
        order.symbol = order.symbol or self.book.symbol
        events = [self._event(now, ExchangeEventType.ORDER_SUBMITTED, order, side=order.side.value,
                              type=order.order_type.value, quantity=order.quantity)]

        if order.quantity <= 0:
            events += self._reject(order, now, "quantity must be positive")
        elif order.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT) and order.limit_price is None:
            events += self._reject(order, now, "limit order requires limit_price")
        elif order.order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and order.stop_price is None:
            events += self._reject(order, now, "stop order requires stop_price")
        elif any(p is not None and (p <= 0 or not self.book.is_on_grid(p))
                 for p in (order.limit_price, order.stop_price)):
            events += self._reject(order, now, f"price must be positive and a multiple of the "
                                               f"tick size {self.book.tick_size}")
        else:
            order.status = OrderStatus.ACCEPTED
            events.append(self._event(now, ExchangeEventType.ORDER_ACCEPTED, order))
            if order.order_type == OrderType.MARKET:
                events += self._execute_market_order(order, now)
            elif order.order_type == OrderType.LIMIT:
                events += self._execute_limit_order(order, now)
            else:
                self._pending_stops.append(order)

        self.event_log.extend(events)
        return events

    def cancel_order(self, order_id: str, timestamp: float | None = None) -> list[ExchangeEvent]:
        """Cancel a resting order or an untriggered stop."""
        now = timestamp if timestamp is not None else time.time()
        events: list[ExchangeEvent] = []
        for i, o in enumerate(self._pending_stops):
            if o.order_id == order_id:
                o.status = OrderStatus.CANCELLED
                self._pending_stops.pop(i)
                events.append(self._event(now, ExchangeEventType.ORDER_CANCELLED, o))
                break
        else:
            order = self.book.cancel_order(order_id)
            if order:
                events.append(self._event(now, ExchangeEventType.ORDER_CANCELLED, order,
                                          filled_quantity=order.filled_quantity))
        self.event_log.extend(events)
        return events

    # ---- Execution ------------------------------------------------------

    def _execute_market_order(self, order: Order, now: float) -> list[ExchangeEvent]:
        events = self._take(order, now, limit_price=None)
        if order.filled_quantity <= _EPS:
            return events + self._reject(order, now, "insufficient liquidity")
        if order.remaining_quantity > _EPS:
            # A market order never rests: the unfilled remainder is cancelled.
            order.status = OrderStatus.CANCELLED
            events.append(self._event(now, ExchangeEventType.ORDER_CANCELLED, order,
                                      reason="unfilled market remainder cancelled",
                                      filled_quantity=order.filled_quantity))
        return events

    def _execute_limit_order(self, order: Order, now: float) -> list[ExchangeEvent]:
        if order.time_in_force == TimeInForce.FOK:
            if self._available(order.side, order.limit_price) + _EPS < order.quantity:
                return self._reject(order, now, "FOK - insufficient liquidity for full fill")

        events = self._take(order, now, order.limit_price) if self._is_marketable(order) else []
        if order.remaining_quantity <= _EPS:
            return events

        if order.time_in_force in (TimeInForce.IOC, TimeInForce.FOK):
            order.status = OrderStatus.CANCELLED
            events.append(self._event(now, ExchangeEventType.ORDER_CANCELLED, order,
                                      reason=f"{order.time_in_force.value} remainder cancelled",
                                      filled_quantity=order.filled_quantity))
            return events

        self.book.add_order(order, timestamp=now)
        events.append(self._event(now, ExchangeEventType.ORDER_QUEUED, order,
                                  price=order.limit_price, quantity=order.remaining_quantity,
                                  queue_ahead=order.queue_ahead))
        return events

    def _is_marketable(self, order: Order) -> bool:
        if order.limit_price is None:
            return False
        if order.side == TradeSide.BUY:
            best_ask = self.book.best_ask
            return best_ask is not None and order.limit_price >= best_ask - _EPS
        best_bid = self.book.best_bid
        return best_bid is not None and order.limit_price <= best_bid + _EPS

    # ---- Market data ----------------------------------------------------

    def on_market_event(self, event: BookEvent) -> list[ExchangeEvent]:
        """Apply one L2 feed event: book update, maker fills, stop triggers."""
        now = event.timestamp
        events: list[ExchangeEvent] = []
        for fill in self.book.apply_market_event(event):
            order = self.book.get_order(fill.order_id)
            if order is not None:
                events.append(self._fill_event(order, fill.fill_price, fill.fill_quantity, now,
                                               aggressor=False))
        if event.event_type == BookEventType.TRADE:
            events.extend(self._check_stop_triggers(event.price, now))
        self.event_log.extend(events)
        return events

    def _check_stop_triggers(self, trade_price: float, now: float) -> list[ExchangeEvent]:
        events: list[ExchangeEvent] = []
        still_pending = []
        for order in self._pending_stops:
            triggered = (trade_price >= order.stop_price - _EPS if order.side == TradeSide.BUY
                         else trade_price <= order.stop_price + _EPS)
            if not triggered:
                still_pending.append(order)
                continue
            events.append(self._event(now, ExchangeEventType.ORDER_TRIGGERED, order,
                                      trigger_price=trade_price))
            if order.order_type == OrderType.STOP:
                order.order_type = OrderType.MARKET
                events += self._execute_market_order(order, now)
            else:
                order.order_type = OrderType.LIMIT
                events += self._execute_limit_order(order, now)
        self._pending_stops = still_pending
        return events

    # ---- Expiry ---------------------------------------------------------

    def expire_orders(self, current_time: float | None = None) -> list[ExchangeEvent]:
        """Expire resting orders and untriggered stops whose expires_at has passed."""
        now = current_time if current_time is not None else time.time()
        events: list[ExchangeEvent] = []
        for oid, order in list(self.book._orders.items()):
            if order.owner == Owner.PAPER and order.is_active and order.expires_at and now >= order.expires_at:
                if self.book.cancel_order(oid):
                    order.status = OrderStatus.EXPIRED
                    events.append(self._event(now, ExchangeEventType.ORDER_EXPIRED, order))
        kept = []
        for order in self._pending_stops:
            if order.expires_at and now >= order.expires_at:
                order.status = OrderStatus.EXPIRED
                events.append(self._event(now, ExchangeEventType.ORDER_EXPIRED, order))
            else:
                kept.append(order)
        self._pending_stops = kept
        self.event_log.extend(events)
        return events
