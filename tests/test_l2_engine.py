"""L2 engine correctness: order book, matching, paper exchange, replay, analytics.

Each class pins a behaviour that was verified broken before the rewrite
(lost fills, stale queue positions, ignored book events, unapplied latency,
crossed synthetic books, cash-only equity) or a guarantee the rewrite adds.
"""
from dataclasses import replace

import pytest

from backend.services.execution_analytics import compute_execution_metrics
from backend.services.l2_event_replay import (
    L2EventStream, ReplayConfig, obi_momentum_strategy, replay_session,
)
from backend.services.market_microstructure import (
    BookEvent, BookEventType as E, L2DataError, TradeSide, build_snapshot_from_events,
    generate_synthetic_l2, validate_events,
)
from backend.services.order_book import Order, OrderBook, OrderStatus, OrderType, TimeInForce
from backend.services.paper_exchange import PaperExchange

BUY, SELL = TradeSide.BUY, TradeSide.SELL


def ev(t, kind, side, price, size, oid=None):
    return BookEvent(float(t), kind, side, price, size, oid)


def seeded(latency_ms=0.0, ask=101.0, bid=99.0, size=100):
    ex = PaperExchange(symbol="X", latency_ms=latency_ms)
    ex.on_market_event(ev(0, E.ADD, SELL, ask, size, "a0"))
    ex.on_market_event(ev(0, E.ADD, BUY, bid, size, "b0"))
    return ex


def bars(n=10, start=100.0, step=1.0):
    return [{"timestamp": 1_700_000_000 + i * 86400, "open": start + i * step, "high": start + 3 + i * step,
             "low": start - 2 + i * step, "close": start + 1 + i * step, "volume": 100_000} for i in range(n)]


class TestFillSettlement:
    def test_marketable_limit_fill_settles_cash_and_position(self):
        ex = seeded()
        o = ex.submit_order(BUY, 10, OrderType.LIMIT, limit_price=102.0)
        assert o.status == OrderStatus.FILLED
        assert ex.cash == pytest.approx(100_000 - 10 * 101.0)
        assert ex.position_quantity() == 10

    def test_partial_market_fill_settles_and_cancels_remainder(self):
        ex = seeded()
        o = ex.submit_order(BUY, 150, OrderType.MARKET)
        assert o.filled_quantity == 100 and o.status == OrderStatus.CANCELLED
        assert ex.cash == pytest.approx(100_000 - 100 * 101.0)
        assert ex.position_quantity() == 100

    def test_triggered_stop_settles(self):
        ex = seeded()
        o = ex.submit_order(BUY, 5, OrderType.STOP, stop_price=101.0)
        ex.on_market_event(ev(1, E.TRADE, BUY, 101.0, 1))
        assert o.status == OrderStatus.FILLED
        assert ex.position_quantity() == 5
        assert ex.cash == pytest.approx(100_000 - 5 * 101.0)

    def test_every_fill_is_reconcilable_from_the_fill_log(self):
        ex = seeded()
        ex.submit_order(BUY, 30, OrderType.MARKET)
        ex.submit_order(SELL, 10, OrderType.LIMIT, limit_price=105.0)
        # Price priority: the print exhausts the 70 left at 101 before reaching our 105.
        ex.on_market_event(ev(2, E.TRADE, BUY, 105.0, 80))
        cash = ex.initial_cash + sum((-1 if f.side == BUY else 1) * f.fill_price * f.fill_quantity
                                    for f in ex.fill_history)
        assert cash == pytest.approx(ex.cash)
        assert ex.position_quantity() == 20

    def test_equity_is_cash_plus_marked_position(self):
        ex = seeded(ask=150.0, bid=149.0)
        ex.submit_order(BUY, 10, OrderType.MARKET)
        summary = ex.portfolio_summary(current_prices={"X": 150.0})
        assert summary["equity"] == pytest.approx(100_000.0)
        assert summary["total_pnl"] == pytest.approx(0.0)


class TestQueueModel:
    def test_queue_ahead_advances_when_orders_in_front_fill(self):
        ex = PaperExchange(symbol="X")
        a = ex.submit_order(BUY, 10, OrderType.LIMIT, limit_price=100.0)
        b = ex.submit_order(BUY, 10, OrderType.LIMIT, limit_price=100.0)
        assert (a.queue_ahead, b.queue_ahead) == (0, 10)
        ex.on_market_event(ev(1, E.TRADE, SELL, 100.0, 10))
        assert a.status == OrderStatus.FILLED
        assert b.queue_ahead == 0

    def test_market_liquidity_ahead_is_consumed_before_our_order(self):
        ex = PaperExchange(symbol="X")
        ex.on_market_event(ev(0, E.ADD, BUY, 100.0, 30, "m1"))
        o = ex.submit_order(BUY, 10, OrderType.LIMIT, limit_price=100.0)
        assert o.queue_ahead == 30
        ex.on_market_event(ev(1, E.TRADE, SELL, 100.0, 25))
        assert o.filled_quantity == 0 and o.queue_ahead == 5
        ex.on_market_event(ev(2, E.TRADE, SELL, 100.0, 8))
        assert o.filled_quantity == 3 and o.status == OrderStatus.PARTIALLY_FILLED

    def test_aggregated_cancels_advance_the_queue_proportionally(self):
        ex = PaperExchange(symbol="X")
        ex.on_market_event(ev(0, E.ADD, BUY, 100.0, 40, "m1"))
        o = ex.submit_order(BUY, 10, OrderType.LIMIT, limit_price=100.0)
        ex.on_market_event(ev(0, E.ADD, BUY, 100.0, 40, "m2"))
        ex.on_market_event(ev(1, E.CANCEL, BUY, 100.0, 40))          # no id: half ahead, half behind
        assert o.queue_ahead == pytest.approx(20)

    def test_trade_through_fills_a_better_priced_resting_order(self):
        ex = seeded()
        ex.submit_order(BUY, 20, OrderType.MARKET)
        s = ex.submit_order(SELL, 10, OrderType.LIMIT, limit_price=100.5)
        ex.on_market_event(ev(1, E.TRADE, BUY, 101.0, 15))
        assert s.status == OrderStatus.FILLED and s.avg_fill_price == 100.5

    def test_paper_orders_never_trade_with_each_other(self):
        ex = seeded()
        ex.submit_order(BUY, 20, OrderType.MARKET)
        ex.submit_order(SELL, 10, OrderType.LIMIT, limit_price=100.0)
        buy = ex.submit_order(BUY, 5, OrderType.LIMIT, limit_price=100.0)
        assert buy.status == OrderStatus.QUEUED


class TestEventModel:
    def test_add_cancel_modify_trade_all_change_the_book(self):
        book = OrderBook()
        book.apply_market_event(ev(0, E.ADD, SELL, 101.0, 50, "s1"))
        book.apply_market_event(ev(0, E.ADD, BUY, 99.0, 40, "b1"))
        assert (book.best_bid, book.best_ask) == (99.0, 101.0)
        book.apply_market_event(ev(1, E.MODIFY, SELL, 101.0, 70))
        assert book.market_depth(SELL, 101.0) == pytest.approx(70)
        book.apply_market_event(ev(2, E.CANCEL, BUY, 99.0, 15, "b1"))
        assert book.market_depth(BUY, 99.0) == pytest.approx(25)
        book.apply_market_event(ev(3, E.TRADE, BUY, 101.0, 70))
        assert book.best_ask is None

    def test_prices_are_integer_ticks(self):
        book = OrderBook(tick_size=0.01)
        book.add_order(Order(side=BUY, quantity=1, limit_price=100.1))
        book.add_order(Order(side=BUY, quantity=1, limit_price=100.10000000001))
        assert len(book._bids) == 1 and book.best_bid == 100.1

    def test_off_grid_paper_order_is_rejected(self):
        ex = seeded()
        o = ex.submit_order(BUY, 1, OrderType.LIMIT, limit_price=100.005)
        assert o.status == OrderStatus.REJECTED

    def test_fok_never_partially_executes(self):
        ex = seeded()
        o = ex.submit_order(BUY, 150, OrderType.LIMIT, limit_price=101.0, time_in_force=TimeInForce.FOK)
        assert o.status == OrderStatus.REJECTED
        assert o.filled_quantity == 0 and ex.position_quantity() == 0
        assert ex.book.market_depth(SELL, 101.0) == pytest.approx(100)


class TestLatency:
    def test_order_is_in_flight_until_its_arrival_time(self):
        ex = seeded(latency_ms=5_000)
        o = ex.submit_order(BUY, 10, OrderType.LIMIT, limit_price=100.0)
        ex.on_market_event(ev(1, E.TRADE, SELL, 100.0, 10))
        assert o.status == OrderStatus.SUBMITTED and o.filled_quantity == 0
        ex.on_market_event(ev(6, E.TRADE, SELL, 100.0, 10))
        assert o.status == OrderStatus.FILLED
        assert o.active_at == pytest.approx(5.0)

    def test_arrival_price_is_the_book_when_the_order_arrives(self):
        ex = seeded(latency_ms=1_000)
        o = ex.submit_order(BUY, 10, OrderType.MARKET)
        ex.on_market_event(ev(0.5, E.TRADE, BUY, 101.0, 100))          # clears the ask
        ex.on_market_event(ev(0.9, E.ADD, SELL, 102.0, 100, "a1"))
        ex.on_market_event(ev(2.0, E.ADD, BUY, 98.0, 1, "b9"))
        assert o.avg_fill_price == 102.0


class TestPreTradeChecks:
    def test_no_naked_short(self):
        ex = seeded()
        assert ex.submit_order(SELL, 1, OrderType.MARKET).status == OrderStatus.REJECTED

    def test_an_order_does_not_count_against_itself(self):
        ex = seeded()
        ex.submit_order(BUY, 20, OrderType.MARKET)
        sell = ex.submit_order(SELL, 20, OrderType.MARKET)
        assert sell.status == OrderStatus.FILLED

    def test_working_buys_commit_cash(self):
        ex = seeded(ask=1_000.0, bid=999.0, size=1_000)
        ex.submit_order(BUY, 60, OrderType.LIMIT, limit_price=990.0)
        second = ex.submit_order(BUY, 60, OrderType.LIMIT, limit_price=990.0)
        assert second.status == OrderStatus.REJECTED


class TestSyntheticData:
    @pytest.mark.parametrize("seed", range(8))
    def test_generated_stream_is_a_valid_book_history(self, seed):
        events = generate_synthetic_l2(bars(), seed=seed, events_per_bar=100)
        assert len(events) == 1_000
        assert validate_events(events) == []
        assert all(abs(e.price / 0.01 - round(e.price / 0.01)) < 1e-6 for e in events)

    def test_book_mid_tracks_the_bar_path(self):
        b = bars()
        events = generate_synthetic_l2(b, seed=3, events_per_bar=200)
        book = OrderBook()
        for k, e in enumerate(events):
            book.apply_market_event(e)
            if (k + 1) % 200 == 0:
                bar = b[k // 200]
                assert abs(book.mid_price - bar["close"]) <= 0.3 * (bar["high"] - bar["low"])

    def test_validation_detects_corrupt_feeds(self):
        crossed = [ev(0, E.ADD, SELL, 100.0, 1, "a"), ev(1, E.ADD, BUY, 100.5, 1, "b")]
        backwards = [ev(2, E.ADD, SELL, 101.0, 1, "a"), ev(1, E.ADD, BUY, 99.0, 1, "b")]
        dup = [ev(0, E.ADD, SELL, 101.0, 1, "a"), ev(1, E.ADD, SELL, 102.0, 1, "a")]
        bad = [ev(0, E.ADD, SELL, -1.0, 1, "a")]
        for stream in (crossed, backwards, dup, bad):
            assert validate_events(stream)

    def test_external_loaders_reject_invalid_data(self):
        with pytest.raises(L2DataError):
            L2EventStream.from_records([{"timestamp": 0, "event_type": "ADD", "side": "SELL", "price": 100, "size": 1},
                                        {"timestamp": 1, "event_type": "ADD", "side": "BUY", "price": 101, "size": 1}])
        stream = L2EventStream.from_csv("timestamp,event_type,side,price,size,order_id\n"
                                        "1,ADD,BUY,99.5,10,b1\n2,ADD,SELL,100.5,10,a1\n")
        assert len(stream) == 2 and stream.source == "external"


class TestReplay:
    def test_incremental_book_matches_a_fresh_rebuild(self):
        stream = L2EventStream.from_synthetic(bars(3), seed=5, events_per_bar=100)
        result = replay_session(stream, config=ReplayConfig(snapshot_interval=50))
        rebuilt = build_snapshot_from_events(stream.events)
        last = result.snapshot_history[-1]
        assert [lvl["price"] for lvl in last["bids"]] == [lvl.price for lvl in rebuilt.bids]
        assert [lvl["size"] for lvl in last["asks"]] == pytest.approx([lvl.size for lvl in rebuilt.asks])
        assert result.data_source == "synthetic"

    def test_closed_loop_replay_trades_and_reconciles(self):
        stream = L2EventStream.from_synthetic(bars(20, step=0.5), seed=7, events_per_bar=200)
        ex = PaperExchange(symbol="SYN")
        result = replay_session(stream, lambda s, t: obi_momentum_strategy(s, t, obi_threshold=0.2),
                                ReplayConfig(snapshot_interval=10, warmup_events=50), exchange=ex)
        assert ex.fill_history, "strategy signals must become fills"
        cash = ex.initial_cash + sum((-1 if f.side == BUY else 1) * f.fill_price * f.fill_quantity
                                    for f in ex.fill_history)
        assert cash == pytest.approx(ex.cash)
        report = result.execution["analytics"]
        assert report["execution_metrics"]["total_orders"] == len(ex.order_history)
        assert report["implementation_shortfall"]["orders"]

    def test_replay_is_deterministic(self):
        stream = L2EventStream.from_synthetic(bars(10), seed=11, events_per_bar=150)

        def run(latency):
            ex = PaperExchange(symbol="SYN", latency_ms=latency)
            return replay_session(stream, lambda s, t: obi_momentum_strategy(s, t, 0.2),
                                  ReplayConfig(snapshot_interval=10, warmup_events=50,
                                               order_style="passive", cancel_after_events=20),
                                  exchange=ex).to_dict()["execution"]

        assert run(0.0) == run(0.0)

    def test_latency_changes_outcomes_when_events_are_dense(self):
        base = L2EventStream.from_synthetic(bars(10), seed=7, events_per_bar=200)
        dense = L2EventStream.from_events([replace(e, timestamp=i * 0.01) for i, e in enumerate(base.events)])

        def fills(latency):
            ex = PaperExchange(symbol="SYN", latency_ms=latency)
            replay_session(dense, lambda s, t: obi_momentum_strategy(s, t, 0.2),
                           ReplayConfig(snapshot_interval=10, warmup_events=50), exchange=ex)
            return [(round(f.fill_price, 4), f.fill_quantity) for f in ex.fill_history]

        assert fills(0.0) != fills(50.0)


class TestExecutionMetrics:
    def test_outcomes_are_counted_separately(self):
        full = Order(side=BUY, quantity=10, limit_price=100.0, status=OrderStatus.FILLED, avg_fill_price=99.5)
        partial_then_cancelled = Order(side=BUY, quantity=10, limit_price=100.0, status=OrderStatus.CANCELLED,
                                       filled_quantity=4, avg_fill_price=100.0)
        cancelled = Order(side=BUY, quantity=10, limit_price=100.0, status=OrderStatus.CANCELLED)
        m = compute_execution_metrics([full, partial_then_cancelled, cancelled], [])
        assert m["full_fill_rate"] == pytest.approx(1 / 3, abs=1e-4)
        assert m["partial_fill_rate"] == pytest.approx(1 / 3, abs=1e-4)
        assert m["any_fill_rate"] == pytest.approx(2 / 3, abs=1e-4)
        assert m["cancel_rate"] == pytest.approx(1 / 3, abs=1e-4)
        assert m["fill_ratio"] == pytest.approx(14 / 30, abs=1e-4)


class TestEndpoints:
    @pytest.fixture
    def live_price(self, monkeypatch):
        from backend import main
        monkeypatch.setattr(main.signal_engine, "get_live_price", lambda _t: {"price": 100.0})

    def test_exchange_simulation_is_synthetic_deterministic_and_server_funded(self, live_price):
        from backend import main, schemas
        req = schemas.ExchangeSimulationRequest(symbol="aapl", side="buy", quantity=5, order_type="limit",
                                                limit_price=99.9, seed=3, flow_events=300)
        first = main.exchange_submit_order(req, user=None)
        second = main.exchange_submit_order(req, user=None)
        assert first["data_source"] == "synthetic"
        assert first["portfolio"]["initial_cash"] == 100_000.0
        assert first == second
        with pytest.raises(Exception):
            schemas.ExchangeSimulationRequest(initial_cash=1_000_000_000)

    def test_sell_simulation_uses_an_explicit_starting_position(self, live_price):
        from backend import main, schemas
        req = schemas.ExchangeSimulationRequest(symbol="AAPL", side="SELL", quantity=5, initial_position=5)
        assert main.exchange_submit_order(req, user=None)["order"]["status"] == "FILLED"

    def test_replay_endpoint_executes_through_the_exchange(self):
        from backend import main, schemas
        req = schemas.MicrostructureReplayRequest(bars=bars(5), events_per_bar=100, execute=True,
                                                  warmup_events=20, obi_threshold=0.2,
                                                  latency_preset="retail")
        result = main.microstructure_replay(req, user=None)
        assert result["data_source"] == "synthetic"
        assert "analytics" in result["execution"]
        assert result["execution"]["exchange_stats"]["latency_ms"] > 0
