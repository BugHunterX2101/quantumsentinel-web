"""Tests for order_book.py, matching_engine.py, and paper_exchange.py.

Covers:
- Order book FIFO queue and price-time priority
- Queue-aware matching: partial fills, cancels, expiry
- Market/Limit/Stop/Stop-Limit order types
- IOC/FOK time-in-force semantics
- Paper exchange integration (end-to-end order lifecycle)
- Paper-trading invariant (no live mode)
"""

import pytest
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
    TimeInForce,
)
from backend.services.matching_engine import (
    ExchangeEvent,
    ExchangeEventType,
    MatchingEngine,
)
from backend.services.paper_exchange import (
    PaperExchange,
    PaperPosition,
    TradingMode,
    TRADING_MODE,
)


# ---------------------------------------------------------------------------
# Order Book
# ---------------------------------------------------------------------------

class TestOrderBook:
    def test_add_order_and_snapshot(self):
        book = OrderBook("AAPL")
        order = Order(
            symbol="AAPL", side=TradeSide.BUY, order_type=OrderType.LIMIT,
            quantity=100, limit_price=150.0,
        )
        book.add_order(order)
        assert book.best_bid == 150.0
        snap = book.snapshot()
        assert len(snap.bids) == 1
        assert snap.bids[0].price == 150.0
        assert snap.bids[0].size == 100

    def test_best_bid_ask(self):
        book = OrderBook("AAPL")
        book.add_order(Order(symbol="AAPL", side=TradeSide.BUY, order_type=OrderType.LIMIT, quantity=100, limit_price=149.0))
        book.add_order(Order(symbol="AAPL", side=TradeSide.BUY, order_type=OrderType.LIMIT, quantity=50, limit_price=150.0))
        book.add_order(Order(symbol="AAPL", side=TradeSide.SELL, order_type=OrderType.LIMIT, quantity=80, limit_price=151.0))
        assert book.best_bid == 150.0
        assert book.best_ask == 151.0
        assert book.spread == pytest.approx(1.0)

    def test_cancel_order(self):
        book = OrderBook("AAPL")
        order = Order(symbol="AAPL", side=TradeSide.BUY, order_type=OrderType.LIMIT, quantity=100, limit_price=150.0)
        book.add_order(order)
        assert book.best_bid == 150.0
        book.cancel_order(order.order_id)
        assert book.best_bid is None

    def test_consume_liquidity(self):
        book = OrderBook("AAPL")
        book.add_order(Order(symbol="AAPL", side=TradeSide.SELL, order_type=OrderType.LIMIT, quantity=30, limit_price=100.0))
        book.add_order(Order(symbol="AAPL", side=TradeSide.SELL, order_type=OrderType.LIMIT, quantity=50, limit_price=100.5))
        # Buy 40 shares: 30 @ 100.0, 10 @ 100.5
        fills = book.consume_liquidity(TradeSide.BUY, 40, timestamp=1.0)
        assert len(fills) == 2
        assert fills[0].fill_price == 100.0
        assert fills[0].fill_quantity == 30
        assert fills[1].fill_price == 100.5
        assert fills[1].fill_quantity == 10

    def test_queue_ahead_tracking(self):
        book = OrderBook("AAPL")
        o1 = Order(symbol="AAPL", side=TradeSide.BUY, order_type=OrderType.LIMIT, quantity=100, limit_price=150.0)
        o2 = Order(symbol="AAPL", side=TradeSide.BUY, order_type=OrderType.LIMIT, quantity=50, limit_price=150.0)
        book.add_order(o1)
        book.add_order(o2)
        # o2 should have 100 shares ahead of it
        assert o2.queue_ahead_at_entry == 100

    def test_fifo_priority(self):
        book = OrderBook("AAPL")
        o1 = Order(symbol="AAPL", side=TradeSide.SELL, order_type=OrderType.LIMIT, quantity=50, limit_price=100.0)
        o2 = Order(symbol="AAPL", side=TradeSide.SELL, order_type=OrderType.LIMIT, quantity=50, limit_price=100.0)
        book.add_order(o1)
        book.add_order(o2)
        # Buy 50 — should fill o1 first (FIFO)
        fills = book.consume_liquidity(TradeSide.BUY, 50, timestamp=1.0)
        assert len(fills) == 1
        assert fills[0].order_id == o1.order_id
        assert o1.status == OrderStatus.FILLED
        assert o2.status == OrderStatus.QUEUED


# ---------------------------------------------------------------------------
# Matching Engine
# ---------------------------------------------------------------------------

class TestMatchingEngine:
    def _make_engine(self) -> MatchingEngine:
        book = OrderBook("AAPL")
        # Seed book with some liquidity
        book.add_order(Order(symbol="AAPL", side=TradeSide.SELL, order_type=OrderType.LIMIT, quantity=100, limit_price=100.0))
        book.add_order(Order(symbol="AAPL", side=TradeSide.SELL, order_type=OrderType.LIMIT, quantity=200, limit_price=100.5))
        book.add_order(Order(symbol="AAPL", side=TradeSide.BUY, order_type=OrderType.LIMIT, quantity=100, limit_price=99.0))
        return MatchingEngine(book)

    def test_market_order_fills(self):
        engine = self._make_engine()
        order = Order(symbol="AAPL", side=TradeSide.BUY, order_type=OrderType.MARKET, quantity=50)
        events = engine.submit_order(order)
        assert order.status == OrderStatus.FILLED
        assert any(e.event_type == ExchangeEventType.ORDER_FILLED for e in events)

    def test_limit_order_rests_on_book(self):
        engine = self._make_engine()
        order = Order(symbol="AAPL", side=TradeSide.BUY, order_type=OrderType.LIMIT, quantity=50, limit_price=98.0)
        events = engine.submit_order(order)
        assert order.status == OrderStatus.QUEUED
        assert any(e.event_type == ExchangeEventType.ORDER_QUEUED for e in events)

    def test_marketable_limit_fills_immediately(self):
        engine = self._make_engine()
        order = Order(symbol="AAPL", side=TradeSide.BUY, order_type=OrderType.LIMIT, quantity=50, limit_price=101.0)
        events = engine.submit_order(order)
        assert order.status == OrderStatus.FILLED

    def test_cancel_resting_order(self):
        engine = self._make_engine()
        order = Order(symbol="AAPL", side=TradeSide.BUY, order_type=OrderType.LIMIT, quantity=50, limit_price=98.0)
        engine.submit_order(order)
        events = engine.cancel_order(order.order_id)
        assert order.status == OrderStatus.CANCELLED
        assert any(e.event_type == ExchangeEventType.ORDER_CANCELLED for e in events)

    def test_ioc_no_fill_cancels(self):
        engine = self._make_engine()
        order = Order(
            symbol="AAPL", side=TradeSide.BUY, order_type=OrderType.LIMIT,
            quantity=50, limit_price=98.0, time_in_force=TimeInForce.IOC,
        )
        events = engine.submit_order(order)
        assert order.status == OrderStatus.CANCELLED

    def test_reject_invalid_order(self):
        engine = self._make_engine()
        order = Order(symbol="AAPL", side=TradeSide.BUY, order_type=OrderType.MARKET, quantity=0)
        events = engine.submit_order(order)
        assert order.status == OrderStatus.REJECTED

    def test_reject_limit_without_price(self):
        engine = self._make_engine()
        order = Order(symbol="AAPL", side=TradeSide.BUY, order_type=OrderType.LIMIT, quantity=50)
        events = engine.submit_order(order)
        assert order.status == OrderStatus.REJECTED

    def test_stop_order_triggers(self):
        engine = self._make_engine()
        order = Order(
            symbol="AAPL", side=TradeSide.BUY, order_type=OrderType.STOP,
            quantity=50, stop_price=100.0,
        )
        engine.submit_order(order)
        assert order.status == OrderStatus.ACCEPTED  # Pending trigger

        # Simulate a trade that triggers the stop
        event = BookEvent(timestamp=2.0, event_type=BookEventType.TRADE, side=TradeSide.BUY, price=100.0, size=10)
        exchange_events = engine.on_market_event(event)
        # Stop should have triggered and filled as market order
        assert order.order_type == OrderType.MARKET


# ---------------------------------------------------------------------------
# Paper Exchange (end-to-end)
# ---------------------------------------------------------------------------

class TestPaperExchange:
    def test_market_order_lifecycle(self):
        exchange = PaperExchange(symbol="AAPL", initial_cash=100_000)
        # Seed the book with asks
        exchange.book.add_order(Order(symbol="AAPL", side=TradeSide.SELL, order_type=OrderType.LIMIT, quantity=100, limit_price=150.0))

        order = exchange.submit_order(TradeSide.BUY, quantity=50, order_type=OrderType.MARKET)
        assert order.status == OrderStatus.FILLED
        assert exchange.cash < 100_000  # Spent money

    def test_portfolio_tracking(self):
        exchange = PaperExchange(symbol="AAPL", initial_cash=100_000)
        exchange.book.add_order(Order(symbol="AAPL", side=TradeSide.SELL, order_type=OrderType.LIMIT, quantity=100, limit_price=150.0))

        exchange.submit_order(TradeSide.BUY, quantity=10, order_type=OrderType.MARKET)
        summary = exchange.portfolio_summary()
        assert summary["trading_mode"] == "paper"
        assert "AAPL" in summary["positions"]
        assert summary["positions"]["AAPL"]["quantity"] == 10

    def test_exchange_stats(self):
        exchange = PaperExchange(symbol="TEST")
        stats = exchange.exchange_stats()
        assert stats["trading_mode"] == "paper"
        assert stats["total_orders"] == 0

    def test_position_close_realized_pnl(self):
        pos = PaperPosition(symbol="AAPL")
        pos.apply_fill(100.0, 10, TradeSide.BUY)  # Buy 10 @ 100
        assert pos.quantity == 10
        assert pos.avg_entry_price == 100.0

        pos.apply_fill(110.0, 10, TradeSide.SELL)  # Sell 10 @ 110
        assert pos.quantity == 0
        assert pos.realized_pnl == pytest.approx(100.0)  # 10 * (110-100)


# ---------------------------------------------------------------------------
# Paper-Trading Invariant
# ---------------------------------------------------------------------------

class TestPaperTradingInvariant:
    def test_only_paper_mode_exists(self):
        """Verify TradingMode enum has exactly one member: PAPER."""
        members = list(TradingMode)
        assert len(members) == 1
        assert members[0] == TradingMode.PAPER
        assert members[0].value == "paper"

    def test_runtime_constant_is_paper(self):
        assert TRADING_MODE == TradingMode.PAPER
        assert TRADING_MODE.value == "paper"

    def test_no_live_mode(self):
        """Ensure LIVE mode cannot be created."""
        with pytest.raises((ValueError, KeyError)):
            TradingMode("live")

    def test_exchange_asserts_paper_mode(self):
        """PaperExchange should work since mode is PAPER."""
        exchange = PaperExchange(symbol="TEST")
        assert exchange.exchange_stats()["trading_mode"] == "paper"
