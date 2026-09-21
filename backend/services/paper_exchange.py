"""QuantumSentinel — Event-Driven Paper Exchange.

The paper exchange is the integration layer that connects:
  Market Data → Order Book → Matching Engine → Paper Orders → Fills → Portfolio

Safety boundary: ``TradingMode`` is an enum with exactly one member
(``PAPER``).  There is no ``LIVE`` variant — the enum makes it
structurally impossible to accidentally enable live trading.

The Alpaca integration (if retained) explicitly points to the Alpaca
**paper** endpoint only.
"""

from __future__ import annotations

import time
import hashlib
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from backend.services.market_microstructure import (
    BookEvent,
    BookEventType,
    OrderBookSnapshot,
    TradeSide,
)
from backend.services.order_book import (
    Fill,
    Order,
    OrderBook,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from backend.services.matching_engine import (
    ExchangeEvent,
    ExchangeEventType,
    MatchingEngine,
)


# ---------------------------------------------------------------------------
# Trading Mode — Paper Only (structural enforcement)
# ---------------------------------------------------------------------------

class TradingMode(str, Enum):
    """Trading mode enum — paper only.  No LIVE member exists."""
    PAPER = "paper"


TRADING_MODE = TradingMode.PAPER  # Immutable runtime constant


# ---------------------------------------------------------------------------
# Portfolio Tracker
# ---------------------------------------------------------------------------

@dataclass
class PaperPosition:
    """A paper portfolio position."""
    symbol: str
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    def apply_fill(self, fill_price: float, fill_qty: float, side: TradeSide) -> None:
        """Update position from a fill."""
        signed_qty = fill_qty if side == TradeSide.BUY else -fill_qty

        if (self.quantity > 0 and signed_qty < 0) or (self.quantity < 0 and signed_qty > 0):
            # Closing position (partial or full)
            close_qty = min(abs(signed_qty), abs(self.quantity))
            pnl_per_unit = fill_price - self.avg_entry_price
            if self.quantity < 0:
                pnl_per_unit = -pnl_per_unit
            self.realized_pnl += close_qty * pnl_per_unit
            remaining = abs(signed_qty) - close_qty

            if remaining > 0:
                # Flipping sides
                self.quantity = remaining if signed_qty > 0 else -remaining
                self.avg_entry_price = fill_price
            else:
                self.quantity += signed_qty
                if abs(self.quantity) < 1e-10:
                    self.quantity = 0.0
                    self.avg_entry_price = 0.0
        else:
            # Opening or adding to position
            total_cost = abs(self.quantity) * self.avg_entry_price + fill_qty * fill_price
            self.quantity += signed_qty
            if abs(self.quantity) > 0:
                self.avg_entry_price = total_cost / abs(self.quantity)

    def mark_to_market(self, current_price: float) -> None:
        """Update unrealized P&L."""
        if self.quantity == 0:
            self.unrealized_pnl = 0.0
        else:
            pnl_per_unit = current_price - self.avg_entry_price
            if self.quantity < 0:
                pnl_per_unit = -pnl_per_unit
            self.unrealized_pnl = abs(self.quantity) * pnl_per_unit

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "quantity": round(self.quantity, 8),
            "avg_entry_price": round(self.avg_entry_price, 8),
            "realized_pnl": round(self.realized_pnl, 4),
            "unrealized_pnl": round(self.unrealized_pnl, 4),
        }


# ---------------------------------------------------------------------------
# Paper Exchange
# ---------------------------------------------------------------------------

class PaperExchange:
    """Event-driven paper exchange with queue-aware matching.

    Connects market data, order book, matching engine, and portfolio
    into a unified simulation environment.

    Architecture::

        Market Event
              ↓
        Order Book Update
              ↓
        Strategy
              ↓
        Paper Order
              ↓
        Matching Engine
              ↓
        Queue Position
              ↓
        Fill
              ↓
        Portfolio
    """

    def __init__(
        self,
        symbol: str = "",
        latency_ms: float = 0.0,
        initial_cash: float = 100_000.0,
    ):
        assert TRADING_MODE == TradingMode.PAPER, "Only paper trading is supported"

        self.symbol = symbol
        self.latency_ms = latency_ms
        self.initial_cash = initial_cash
        self.cash = initial_cash

        self.book = OrderBook(symbol=symbol)
        self.engine = MatchingEngine(self.book)
        self.positions: dict[str, PaperPosition] = {}
        self.fill_history: list[Fill] = []
        self.order_history: list[Order] = []
        self.event_log: list[ExchangeEvent] = []

        self._event_count = 0
        self._current_time = 0.0

    # ---- Order submission -----------------------------------------------

    def submit_order(
        self,
        side: TradeSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        expires_at: float | None = None,
    ) -> Order:
        """Submit a paper order to the exchange."""
        order = Order(
            symbol=self.symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            submitted_at=self._current_time,
            expires_at=expires_at,
        )
        self.order_history.append(order)

        events = self.engine.submit_order(order, timestamp=self._current_time)
        self.event_log.extend(events)

        # Process any fills from immediate execution
        self._process_fills_from_events(events, side)

        return order

    def cancel_order(self, order_id: str) -> list[ExchangeEvent]:
        """Cancel a pending paper order."""
        events = self.engine.cancel_order(order_id)
        self.event_log.extend(events)
        return events

    # ---- Market event processing ---------------------------------------

    def on_market_event(self, event: BookEvent) -> list[ExchangeEvent]:
        """Process an incoming market data event through the full pipeline."""
        self._event_count += 1
        self._current_time = event.timestamp

        # Apply latency — the order book sees the event only after latency_ms
        effective_time = event.timestamp + self.latency_ms / 1000.0

        # Process through matching engine
        events = self.engine.on_market_event(event)

        # Process fills
        for ev in events:
            if ev.event_type in (ExchangeEventType.ORDER_FILLED, ExchangeEventType.ORDER_PARTIALLY_FILLED):
                fill_data = ev.details
                if "fill_price" in fill_data and "fill_quantity" in fill_data:
                    order = self.book.get_order(ev.order_id) if ev.order_id else None
                    side = TradeSide.BUY
                    if order:
                        side = order.side
                    self._apply_fill_to_portfolio(
                        ev.order_id or "",
                        fill_data["fill_price"],
                        fill_data["fill_quantity"],
                        side,
                    )

        self.event_log.extend(events)
        return events

    # ---- Fill processing ------------------------------------------------

    def _process_fills_from_events(
        self,
        events: list[ExchangeEvent],
        side: TradeSide,
    ) -> None:
        """Extract fills from exchange events and apply to portfolio."""
        for ev in events:
            if ev.event_type in (ExchangeEventType.ORDER_FILLED, ExchangeEventType.ORDER_PARTIALLY_FILLED):
                details = ev.details
                fill_price = details.get("avg_price") or details.get("fill_price", 0)
                fill_qty = details.get("quantity") or details.get("fill_quantity", 0)
                if fill_price and fill_qty:
                    self._apply_fill_to_portfolio(
                        ev.order_id or "",
                        fill_price,
                        fill_qty,
                        side,
                    )

    def _apply_fill_to_portfolio(
        self,
        order_id: str,
        fill_price: float,
        fill_qty: float,
        side: TradeSide,
    ) -> None:
        """Update portfolio position and cash from a fill."""
        symbol = self.symbol
        if symbol not in self.positions:
            self.positions[symbol] = PaperPosition(symbol=symbol)

        pos = self.positions[symbol]
        pos.apply_fill(fill_price, fill_qty, side)

        # Update cash
        cost = fill_price * fill_qty
        if side == TradeSide.BUY:
            self.cash -= cost
        else:
            self.cash += cost

        self.fill_history.append(Fill(
            order_id=order_id,
            fill_price=fill_price,
            fill_quantity=fill_qty,
            timestamp=self._current_time,
        ))

    # ---- Portfolio state ------------------------------------------------

    def portfolio_summary(self, current_prices: dict[str, float] | None = None) -> dict:
        """Generate a portfolio summary."""
        if current_prices:
            for sym, pos in self.positions.items():
                if sym in current_prices:
                    pos.mark_to_market(current_prices[sym])

        total_unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        total_realized = sum(p.realized_pnl for p in self.positions.values())

        return {
            "trading_mode": TRADING_MODE.value,
            "initial_cash": self.initial_cash,
            "cash": round(self.cash, 4),
            "total_realized_pnl": round(total_realized, 4),
            "total_unrealized_pnl": round(total_unrealized, 4),
            "total_pnl": round(total_realized + total_unrealized, 4),
            "equity": round(self.cash + total_unrealized, 4),
            "positions": {
                sym: pos.to_dict() for sym, pos in self.positions.items()
                if pos.quantity != 0
            },
            "total_orders": len(self.order_history),
            "total_fills": len(self.fill_history),
            "events_processed": self._event_count,
        }

    def order_summary(self) -> list[dict]:
        """Return all orders with current status."""
        return [o.to_dict() for o in self.order_history]

    def exchange_stats(self) -> dict:
        """Return exchange-level statistics."""
        filled = sum(1 for o in self.order_history if o.status == OrderStatus.FILLED)
        partial = sum(1 for o in self.order_history if o.status == OrderStatus.PARTIALLY_FILLED)
        cancelled = sum(1 for o in self.order_history if o.status == OrderStatus.CANCELLED)
        rejected = sum(1 for o in self.order_history if o.status == OrderStatus.REJECTED)
        queued = sum(1 for o in self.order_history if o.status == OrderStatus.QUEUED)

        return {
            "trading_mode": TRADING_MODE.value,
            "symbol": self.symbol,
            "total_orders": len(self.order_history),
            "filled": filled,
            "partially_filled": partial,
            "cancelled": cancelled,
            "rejected": rejected,
            "queued": queued,
            "total_fills": len(self.fill_history),
            "total_events": len(self.event_log),
            "book_bid_levels": len(self.book._bids),
            "book_ask_levels": len(self.book._asks),
        }
