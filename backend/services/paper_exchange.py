"""QuantumSentinel — Event-Driven Paper Exchange (research simulator).

Connects: Market Data → Order Book → Matching Engine → Paper Orders →
Fills → Portfolio → Execution Analytics, on one simulation clock.

Safety boundary: ``TradingMode`` is an enum with exactly one member
(``PAPER``). There is no ``LIVE`` variant, and the codebase contains no
external broker client.

Simulation semantics
--------------------
* **Clock.** The exchange clock is the timestamp of the last market event
  processed (or ``advance_to``). Orders are stamped with it.
* **Latency.** An order submitted at ``t`` reaches the matching engine at
  ``t + latency_ms``: it cannot trade, rest or be seen by the book before
  then. In-flight orders are released just before the first market event
  at or after their arrival time.
* **Cash account, long only.** A buy must be covered by available cash
  (cash minus what working buy orders have committed); a sell must be
  covered by the position minus shares committed to working sells.
* **Fills.** Cash and positions change only from fill events emitted by
  the matching engine, each carrying its own price, quantity and side.
"""

from __future__ import annotations

import bisect
import heapq
from dataclasses import dataclass
from enum import Enum

from backend.services.market_microstructure import BookEvent, TradeSide
from backend.services.order_book import (
    Fill,
    Order,
    OrderBook,
    OrderStatus,
    OrderType,
    Owner,
    TimeInForce,
)
from backend.services.matching_engine import (
    FILL_EVENTS,
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

# Starting capital of a simulation. Fixed server-side: API callers cannot
# choose their own buying power.
SIM_INITIAL_CASH = 100_000.0

_EPS = 1e-9
_WORKING = (OrderStatus.SUBMITTED, OrderStatus.ACCEPTED, OrderStatus.QUEUED,
            OrderStatus.PARTIALLY_FILLED)


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
            close_qty = min(abs(signed_qty), abs(self.quantity))
            pnl_per_unit = fill_price - self.avg_entry_price
            if self.quantity < 0:
                pnl_per_unit = -pnl_per_unit
            self.realized_pnl += close_qty * pnl_per_unit
            remaining = abs(signed_qty) - close_qty

            if remaining > 0:
                self.quantity = remaining if signed_qty > 0 else -remaining
                self.avg_entry_price = fill_price
            else:
                self.quantity += signed_qty
                if abs(self.quantity) < 1e-10:
                    self.quantity = 0.0
                    self.avg_entry_price = 0.0
        else:
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
    """Event-driven, latency-aware paper exchange with queue-aware matching."""

    def __init__(
        self,
        symbol: str = "",
        latency_ms: float = 0.0,
        initial_cash: float = SIM_INITIAL_CASH,
        tick_size: float = 0.01,
    ):
        assert TRADING_MODE == TradingMode.PAPER, "Only paper trading is supported"
        if latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")

        self.symbol = symbol
        self.latency_ms = latency_ms
        self.initial_cash = initial_cash
        self.cash = initial_cash

        self.book = OrderBook(symbol=symbol, tick_size=tick_size)
        self.engine = MatchingEngine(self.book)
        self.positions: dict[str, PaperPosition] = {}
        self.fill_history: list[Fill] = []
        self.order_history: list[Order] = []
        self.event_log: list[ExchangeEvent] = []

        # Execution analytics inputs, all on the simulation clock.
        self.decision_mid: dict[str, float | None] = {}
        self.arrival_mid: dict[str, float | None] = {}
        self.mid_history: list[tuple[float, float]] = []

        self._by_id: dict[str, Order] = {}
        self._order_seq = 0
        self._inflight: list[tuple[float, int, Order]] = []
        self._market_cost_estimate: dict[str, float] = {}
        self._seq = 0
        self._event_count = 0
        self._current_time = 0.0
        self._has_expiring = False

    # ---- State helpers --------------------------------------------------

    @property
    def current_time(self) -> float:
        return self._current_time

    def position_quantity(self) -> float:
        pos = self.positions.get(self.symbol)
        return pos.quantity if pos else 0.0

    def _working(self, side: TradeSide) -> list[Order]:
        return [o for o in self.order_history if o.side == side and o.status in _WORKING]

    @property
    def committed_cash(self) -> float:
        """Cash committed to working buy orders at their worst-case price."""
        total = 0.0
        for o in self._working(TradeSide.BUY):
            if o.order_type == OrderType.MARKET:
                total += self._market_cost_estimate.get(o.order_id, 0.0)
            else:
                price = o.limit_price if o.limit_price is not None else o.stop_price
                total += o.remaining_quantity * (price or 0.0)
        return total

    @property
    def available_cash(self) -> float:
        return self.cash - self.committed_cash

    @property
    def sellable_quantity(self) -> float:
        return self.position_quantity() - sum(o.remaining_quantity for o in self._working(TradeSide.SELL))

    def _estimate_market_buy_cost(self, quantity: float) -> float | None:
        remaining, cost, last_price = quantity, 0.0, None
        for level in self.book.ask_levels:
            take = min(remaining, level.market_size)
            cost += take * level.price
            remaining -= take
            last_price = level.price
            if remaining <= _EPS:
                break
        if last_price is None:
            return None     # no liquidity: the engine will reject it anyway
        return cost + max(0.0, remaining) * last_price

    def _pretrade_reject_reason(self, order: Order) -> str | None:
        if order.side == TradeSide.BUY:
            if order.order_type == OrderType.MARKET:
                cost = self._estimate_market_buy_cost(order.quantity)
                if cost is None:
                    return None
                self._market_cost_estimate[order.order_id] = cost
            else:
                price = order.limit_price if order.limit_price is not None else order.stop_price
                cost = order.quantity * (price or 0.0)
            if cost > self.available_cash + 1e-6:
                return (f"insufficient available cash: order needs {cost:.2f}, "
                        f"{self.available_cash:.2f} available")
        elif order.quantity > self.sellable_quantity + _EPS:
            return "sell quantity exceeds the sellable position (no short selling)"
        return None

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
        """Submit a paper order at the current simulation time.

        With ``latency_ms > 0`` the order is in flight (status SUBMITTED)
        until the clock reaches its arrival time.
        """
        # Sequential ids make identical replays produce identical artifacts.
        self._order_seq += 1
        order = Order(
            order_id=f"P{self._order_seq:08d}",
            symbol=self.symbol, side=side, order_type=order_type, quantity=quantity,
            limit_price=limit_price, stop_price=stop_price, time_in_force=time_in_force,
            submitted_at=self._current_time, expires_at=expires_at, owner=Owner.PAPER,
        )
        # Check before recording: a working order must not count against itself.
        reason = self._pretrade_reject_reason(order)
        self.order_history.append(order)
        self._by_id[order.order_id] = order
        self.decision_mid[order.order_id] = self.book.mid_price
        if reason:
            order.status = OrderStatus.REJECTED
            self.event_log.append(ExchangeEvent(self._current_time, ExchangeEventType.ORDER_REJECTED,
                                                order.order_id, self.symbol, {"reason": reason}))
            return order
        self._has_expiring = self._has_expiring or expires_at is not None
        order.active_at = self._current_time + self.latency_ms / 1000.0
        if self.latency_ms > 0:
            heapq.heappush(self._inflight, (order.active_at, self._seq, order))
            self._seq += 1
            self.event_log.append(ExchangeEvent(self._current_time, ExchangeEventType.ORDER_SUBMITTED,
                                                order.order_id, self.symbol,
                                                {"in_flight_until": order.active_at}))
        else:
            self._activate(order)
        return order

    def _activate(self, order: Order) -> None:
        self.arrival_mid[order.order_id] = self.book.mid_price
        self._settle(self.engine.submit_order(order, timestamp=order.active_at))
        self._market_cost_estimate.pop(order.order_id, None)

    def cancel_order(self, order_id: str) -> list[ExchangeEvent]:
        """Cancel a working order (including one still in flight)."""
        for i, (_t, _s, order) in enumerate(self._inflight):
            if order.order_id == order_id:
                self._inflight.pop(i)
                heapq.heapify(self._inflight)
                order.status = OrderStatus.CANCELLED
                self._market_cost_estimate.pop(order_id, None)
                event = ExchangeEvent(self._current_time, ExchangeEventType.ORDER_CANCELLED,
                                      order_id, self.symbol, {"reason": "cancelled in flight"})
                self.event_log.append(event)
                return [event]
        events = self.engine.cancel_order(order_id, timestamp=self._current_time)
        self.event_log.extend(events)
        return events

    # ---- Clock and market events ----------------------------------------

    def advance_to(self, timestamp: float) -> None:
        """Move the clock forward, releasing orders whose arrival time has come."""
        while self._inflight and self._inflight[0][0] <= timestamp + _EPS:
            arrival, _seq, order = heapq.heappop(self._inflight)
            self._current_time = max(self._current_time, arrival)
            self._activate(order)
        self._current_time = max(self._current_time, timestamp)
        if self._has_expiring:
            self._settle(self.engine.expire_orders(self._current_time))

    def on_market_event(self, event: BookEvent) -> list[ExchangeEvent]:
        """Process one market data event through the full pipeline."""
        self.advance_to(event.timestamp)
        self._event_count += 1
        events = self.engine.on_market_event(event)
        self._settle(events)
        mid = self.book.mid_price
        if mid is not None and (not self.mid_history or self.mid_history[-1][1] != mid):
            self.mid_history.append((event.timestamp, mid))
        return events

    # ---- Fill settlement ------------------------------------------------

    def _settle(self, events: list[ExchangeEvent]) -> None:
        for ev in events:
            self.event_log.append(ev)
            if ev.event_type in FILL_EVENTS:
                d = ev.details
                self._apply_fill(ev.order_id or "", d["fill_price"], d["fill_quantity"],
                                 TradeSide(d["side"]), ev.timestamp, d.get("liquidity") == "taker")

    def _apply_fill(self, order_id: str, price: float, qty: float, side: TradeSide,
                    timestamp: float, aggressor: bool) -> None:
        pos = self.positions.setdefault(self.symbol, PaperPosition(symbol=self.symbol))
        pos.apply_fill(price, qty, side)
        self.cash += -price * qty if side == TradeSide.BUY else price * qty
        order = self._by_id.get(order_id)
        self.fill_history.append(Fill(
            order_id=order_id, fill_price=price, fill_quantity=qty, timestamp=timestamp,
            is_partial=bool(order and order.status != OrderStatus.FILLED),
            aggressor=aggressor, side=side,
        ))

    # ---- Reporting ------------------------------------------------------

    def mid_at(self, timestamp: float) -> float | None:
        """Last observed mid at or before ``timestamp``."""
        idx = bisect.bisect_right(self.mid_history, (timestamp, float("inf"))) - 1
        return self.mid_history[idx][1] if idx >= 0 else None

    def portfolio_summary(self, current_prices: dict[str, float] | None = None) -> dict:
        """Portfolio marked at ``current_prices`` or, by default, the book mid."""
        mark = (current_prices or {}).get(self.symbol) or self.book.mid_price or self.book._last_trade_price
        positions_value = 0.0
        for sym, pos in self.positions.items():
            if mark is not None:
                pos.mark_to_market(mark)
                positions_value += pos.quantity * mark
            else:
                positions_value += pos.quantity * pos.avg_entry_price

        total_unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        total_realized = sum(p.realized_pnl for p in self.positions.values())
        equity = self.cash + positions_value
        return {
            "trading_mode": TRADING_MODE.value,
            "initial_cash": self.initial_cash,
            "cash": round(self.cash, 4),
            "committed_cash": round(self.committed_cash, 4),
            "mark_price": mark,
            "total_realized_pnl": round(total_realized, 4),
            "total_unrealized_pnl": round(total_unrealized, 4),
            "total_pnl": round(equity - self.initial_cash, 4),
            "equity": round(equity, 4),
            "positions": {
                sym: pos.to_dict() for sym, pos in self.positions.items()
                if pos.quantity != 0
            },
            "total_orders": len(self.order_history),
            "total_fills": len(self.fill_history),
            "events_processed": self._event_count,
        }

    def order_summary(self) -> list[dict]:
        return [o.to_dict() for o in self.order_history]

    def exchange_stats(self) -> dict:
        count = lambda status: sum(1 for o in self.order_history if o.status == status)  # noqa: E731
        return {
            "trading_mode": TRADING_MODE.value,
            "symbol": self.symbol,
            "latency_ms": self.latency_ms,
            "tick_size": self.book.tick_size,
            "total_orders": len(self.order_history),
            "filled": count(OrderStatus.FILLED),
            "partially_filled": count(OrderStatus.PARTIALLY_FILLED),
            "cancelled": count(OrderStatus.CANCELLED),
            "rejected": count(OrderStatus.REJECTED),
            "expired": count(OrderStatus.EXPIRED),
            "queued": count(OrderStatus.QUEUED),
            "in_flight": len(self._inflight),
            "total_fills": len(self.fill_history),
            "total_events": len(self.event_log),
            "book_bid_levels": len(self.book._bids),
            "book_ask_levels": len(self.book._asks),
        }

    def execution_report(self, adverse_selection_horizons_s: tuple[float, ...] = (1.0, 10.0, 60.0)) -> dict:
        """Execution analytics derived automatically from this exchange's own fills."""
        from backend.services import execution_analytics as ea
        return ea.build_execution_report(self, adverse_selection_horizons_s)
