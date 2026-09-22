"""Tests for market_microstructure.py and l2_event_replay.py.

Covers:
- OrderBookSnapshot construction and analytics
- OBI, microprice, spread, depth, price impact calculations
- Synthetic L2 generation from OHLCV
- L2 event replay engine
"""

import pytest
from backend.services.market_microstructure import (
    BookEvent,
    BookEventType,
    OrderBookSnapshot,
    PriceLevel,
    TradeEvent,
    TradeSide,
    build_snapshot_from_events,
    compute_depth,
    compute_effective_spread,
    compute_microprice,
    compute_microprice_deviation,
    compute_mid_price,
    compute_order_book_imbalance,
    compute_price_impact,
    compute_realized_volatility,
    compute_spread,
    compute_spread_bps,
    compute_trade_imbalance,
    generate_synthetic_l2,
    snapshot_to_dict,
)
from backend.services.l2_event_replay import (
    L2EventStream,
    ReplayConfig,
    ReplayResult,
    obi_momentum_strategy,
    replay_session,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_snapshot(
    bids=((100.0, 50), (99.9, 100), (99.8, 200)),
    asks=((100.1, 40), (100.2, 80), (100.3, 150)),
) -> OrderBookSnapshot:
    return OrderBookSnapshot(
        timestamp=1000.0,
        bids=[PriceLevel(p, s) for p, s in bids],
        asks=[PriceLevel(p, s) for p, s in asks],
        last_trade_price=100.05,
        last_trade_size=10,
    )


def _make_ohlcv_bars(n=5, base_price=100.0) -> list[dict]:
    bars = []
    for i in range(n):
        o = base_price + i * 0.5
        bars.append({
            "timestamp": 1_000_000 + i * 86400,
            "open": o,
            "high": o + 1.0,
            "low": o - 0.5,
            "close": o + 0.3,
            "volume": 10_000,
        })
    return bars


# ---------------------------------------------------------------------------
# Mid-price, spread, microprice
# ---------------------------------------------------------------------------

class TestCoreAnalytics:
    def test_mid_price(self):
        snap = _make_snapshot()
        mid = compute_mid_price(snap)
        assert mid == pytest.approx((100.0 + 100.1) / 2)

    def test_spread(self):
        snap = _make_snapshot()
        assert compute_spread(snap) == pytest.approx(0.1)

    def test_spread_bps(self):
        snap = _make_snapshot()
        bps = compute_spread_bps(snap)
        assert bps is not None
        assert bps > 0

    def test_microprice(self):
        snap = _make_snapshot()
        micro = compute_microprice(snap)
        assert micro is not None
        # Microprice should be between bid and ask
        assert 100.0 <= micro <= 100.1

    def test_microprice_deviation(self):
        snap = _make_snapshot()
        dev = compute_microprice_deviation(snap)
        assert dev is not None
        assert isinstance(dev, float)

    def test_empty_book_returns_none(self):
        snap = OrderBookSnapshot(timestamp=1.0, bids=[], asks=[])
        assert compute_mid_price(snap) is None
        assert compute_spread(snap) is None
        assert compute_microprice(snap) is None
        assert compute_order_book_imbalance(snap) is None


# ---------------------------------------------------------------------------
# OBI
# ---------------------------------------------------------------------------

class TestOBI:
    def test_balanced_book(self):
        snap = _make_snapshot(
            bids=((100.0, 100),),
            asks=((100.1, 100),),
        )
        obi = compute_order_book_imbalance(snap, levels=1)
        assert obi == pytest.approx(0.0)

    def test_bid_heavy_positive(self):
        snap = _make_snapshot(
            bids=((100.0, 200),),
            asks=((100.1, 50),),
        )
        obi = compute_order_book_imbalance(snap, levels=1)
        assert obi is not None
        assert obi > 0  # Bid side dominates

    def test_ask_heavy_negative(self):
        snap = _make_snapshot(
            bids=((100.0, 50),),
            asks=((100.1, 200),),
        )
        obi = compute_order_book_imbalance(snap, levels=1)
        assert obi is not None
        assert obi < 0  # Ask side dominates

    def test_obi_bounded(self):
        snap = _make_snapshot()
        obi = compute_order_book_imbalance(snap, levels=5)
        assert obi is not None
        assert -1.0 <= obi <= 1.0


# ---------------------------------------------------------------------------
# Depth & price impact
# ---------------------------------------------------------------------------

class TestDepthAndImpact:
    def test_depth(self):
        snap = _make_snapshot()
        d = compute_depth(snap, levels=3)
        assert d["bid_depth"] == 50 + 100 + 200
        assert d["ask_depth"] == 40 + 80 + 150

    def test_price_impact_small_order(self):
        snap = _make_snapshot()
        result = compute_price_impact(10, snap, TradeSide.BUY)
        assert result["fully_filled"] is True
        assert result["vwap"] == pytest.approx(100.1)
        assert result["levels_consumed"] == 1

    def test_price_impact_multi_level(self):
        snap = _make_snapshot()
        # Buy 50 shares: 40 @ 100.1 + 10 @ 100.2
        result = compute_price_impact(50, snap, TradeSide.BUY)
        assert result["fully_filled"] is True
        assert result["levels_consumed"] == 2
        expected_vwap = (40 * 100.1 + 10 * 100.2) / 50
        assert result["vwap"] == pytest.approx(expected_vwap, abs=0.01)


# ---------------------------------------------------------------------------
# Trade imbalance & realized volatility
# ---------------------------------------------------------------------------

class TestTradeAnalytics:
    def test_trade_imbalance(self):
        trades = [
            TradeEvent(1.0, 100.0, 50, TradeSide.BUY),
            TradeEvent(2.0, 100.1, 30, TradeSide.SELL),
            TradeEvent(3.0, 100.0, 20, TradeSide.BUY),
        ]
        ti = compute_trade_imbalance(trades, window_seconds=10)
        assert ti is not None
        assert ti > 0  # More buy volume

    def test_effective_spread(self):
        es = compute_effective_spread(100.05, 100.0, TradeSide.BUY)
        assert es == pytest.approx(0.1)

    def test_realized_volatility(self):
        mids = [100.0, 100.5, 101.0, 100.8, 101.2]
        rv = compute_realized_volatility(mids)
        assert rv is not None
        assert rv > 0


# ---------------------------------------------------------------------------
# Synthetic L2 generation
# ---------------------------------------------------------------------------

class TestSyntheticL2:
    def test_generates_events(self):
        bars = _make_ohlcv_bars(3)
        events = generate_synthetic_l2(bars, events_per_bar=10, seed=42)
        assert len(events) == 30
        assert all(isinstance(e, BookEvent) for e in events)

    def test_deterministic_with_seed(self):
        bars = _make_ohlcv_bars(3)
        e1 = generate_synthetic_l2(bars, events_per_bar=10, seed=42)
        e2 = generate_synthetic_l2(bars, events_per_bar=10, seed=42)
        assert len(e1) == len(e2)
        for a, b in zip(e1, e2):
            assert a.price == b.price
            assert a.size == b.size

    def test_events_sorted_by_time(self):
        bars = _make_ohlcv_bars(5)
        events = generate_synthetic_l2(bars, seed=1)
        for i in range(1, len(events)):
            assert events[i].timestamp >= events[i - 1].timestamp

    def test_build_snapshot_from_events(self):
        bars = _make_ohlcv_bars(3)
        events = generate_synthetic_l2(bars, seed=42)
        snap = build_snapshot_from_events(events, levels=5)
        assert isinstance(snap, OrderBookSnapshot)
        # Should have some bids and asks
        assert len(snap.bids) > 0 or len(snap.asks) > 0

    def test_snapshot_to_dict(self):
        snap = _make_snapshot()
        d = snapshot_to_dict(snap)
        assert "mid_price" in d
        assert "spread" in d
        assert "microprice" in d
        assert "obi_5" in d


# ---------------------------------------------------------------------------
# L2 Event Replay
# ---------------------------------------------------------------------------

class TestL2Replay:
    def test_stream_from_synthetic(self):
        bars = _make_ohlcv_bars(3)
        stream = L2EventStream.from_synthetic(bars, seed=42, events_per_bar=10)
        assert len(stream) == 30
        assert stream.dataset_hash  # non-empty

    def test_stream_deterministic_hash(self):
        bars = _make_ohlcv_bars(3)
        s1 = L2EventStream.from_synthetic(bars, seed=42)
        s2 = L2EventStream.from_synthetic(bars, seed=42)
        assert s1.dataset_hash == s2.dataset_hash

    def test_dataset_hash_covers_every_event_and_all_replay_fields(self):
        """A provenance hash must change for any event that changes replay."""
        events = [
            BookEvent(float(i), BookEventType.ADD, TradeSide.BUY, 100.0, 1.0,
                      order_id=f"order-{i}")
            for i in range(1_001)
        ]
        changed_tail = list(events)
        changed_tail[-1] = BookEvent(1000.0, BookEventType.ADD, TradeSide.BUY,
                                      101.0, 1.0, order_id="order-1000")
        changed_side = list(events)
        changed_side[0] = BookEvent(0.0, BookEventType.ADD, TradeSide.SELL,
                                     100.0, 1.0, order_id="order-0")
        changed_order_id = list(events)
        changed_order_id[0] = BookEvent(0.0, BookEventType.ADD, TradeSide.BUY,
                                         100.0, 1.0, order_id="different-order")

        stream = L2EventStream.from_events(events)
        assert stream.dataset_hash != L2EventStream.from_events(changed_tail).dataset_hash
        assert stream.dataset_hash != L2EventStream.from_events(changed_side).dataset_hash
        assert stream.dataset_hash != L2EventStream.from_events(changed_order_id).dataset_hash

    def test_replay_session_basic(self):
        bars = _make_ohlcv_bars(3)
        stream = L2EventStream.from_synthetic(bars, seed=42, events_per_bar=20)
        result = replay_session(stream, config=ReplayConfig(snapshot_interval=5))
        assert result.total_events == 60
        assert result.snapshots_generated > 0

    def test_replay_with_strategy(self):
        bars = _make_ohlcv_bars(5)
        stream = L2EventStream.from_synthetic(bars, seed=42, events_per_bar=20)
        result = replay_session(
            stream,
            strategy_fn=obi_momentum_strategy,
            config=ReplayConfig(snapshot_interval=5, warmup_events=10),
        )
        assert result.total_events == 100
        # Strategy may or may not fire depending on OBI
        assert isinstance(result.signal_history, list)

    def test_replay_result_to_dict(self):
        bars = _make_ohlcv_bars(2)
        stream = L2EventStream.from_synthetic(bars, seed=1)
        result = replay_session(stream)
        d = result.to_dict()
        assert "total_events" in d
        assert "dataset_hash" in d

    def test_stream_from_records(self):
        records = [
            {"timestamp": 1.0, "event_type": "ADD", "side": "BUY", "price": 100.0, "size": 10},
            {"timestamp": 2.0, "event_type": "TRADE", "side": "SELL", "price": 100.1, "size": 5},
        ]
        stream = L2EventStream.from_records(records)
        assert len(stream) == 2
