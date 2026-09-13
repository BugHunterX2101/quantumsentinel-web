"""Tests for execution_analytics.py and latency_model.py.

Covers:
- Implementation shortfall decomposition
- Queue-position analytics
- Execution quality metrics
- Capacity analysis
- Latency model presets and sensitivity analysis
"""

import pytest
from backend.services.execution_analytics import (
    compute_adverse_selection_summary,
    compute_capacity_analysis,
    compute_execution_metrics,
    compute_implementation_shortfall,
    compute_queue_analytics,
)
from backend.services.latency_model import (
    LATENCY_PRESETS,
    LatencyModel,
    get_preset,
    latency_sensitivity_analysis,
)
from backend.services.order_book import Fill, Order, OrderStatus, OrderType, TimeInForce
from backend.services.market_microstructure import TradeSide


# ---------------------------------------------------------------------------
# Implementation Shortfall
# ---------------------------------------------------------------------------

class TestImplementationShortfall:
    def test_buy_order_shortfall(self):
        result = compute_implementation_shortfall(
            decision_price=100.0,
            arrival_price=100.05,
            execution_vwap=100.10,
            side="BUY",
            quantity=100,
            spread=0.02,
            fees=1.0,
        )
        assert result["delay_cost"] == pytest.approx(5.0)  # (100.05-100)*100
        assert result["market_impact"] == pytest.approx(5.0)  # (100.10-100.05)*100
        assert result["spread_cost"] == pytest.approx(2.0)  # 0.02*100
        assert result["fees"] == pytest.approx(1.0)
        assert result["total_is"] == pytest.approx(13.0)
        assert result["total_is_bps"] > 0

    def test_sell_order_shortfall(self):
        result = compute_implementation_shortfall(
            decision_price=100.0,
            arrival_price=99.95,
            execution_vwap=99.90,
            side="SELL",
            quantity=100,
        )
        # For sells, delay cost = -(arrival - decision) * qty
        assert result["delay_cost"] == pytest.approx(5.0)  # -(99.95-100)*100
        assert isinstance(result["total_is_bps"], float)

    def test_zero_shortfall(self):
        result = compute_implementation_shortfall(
            decision_price=100.0,
            arrival_price=100.0,
            execution_vwap=100.0,
            side="BUY",
            quantity=100,
        )
        assert result["total_is"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Queue Analytics
# ---------------------------------------------------------------------------

class TestQueueAnalytics:
    def test_empty_orders(self):
        result = compute_queue_analytics([])
        assert result["total_orders"] == 0

    def test_filled_orders(self):
        o1 = Order(symbol="AAPL", side=TradeSide.BUY, order_type=OrderType.LIMIT,
                    quantity=100, limit_price=150.0, status=OrderStatus.FILLED)
        o1.queue_ahead_at_entry = 200
        o1.queue_ahead_peak = 250
        o1.entered_book_at = 1000.0

        o2 = Order(symbol="AAPL", side=TradeSide.BUY, order_type=OrderType.LIMIT,
                    quantity=50, limit_price=150.0, status=OrderStatus.QUEUED)
        o2.queue_ahead_at_entry = 100
        o2.queue_ahead_peak = 100

        result = compute_queue_analytics([o1, o2])
        assert result["total_orders"] == 2
        assert result["filled"] == 1
        assert result["resting"] == 1
        assert result["fill_rate"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Execution Metrics
# ---------------------------------------------------------------------------

class TestExecutionMetrics:
    def test_basic_metrics(self):
        orders = [
            Order(symbol="AAPL", side=TradeSide.BUY, order_type=OrderType.LIMIT,
                  quantity=100, limit_price=150.0, status=OrderStatus.FILLED,
                  avg_fill_price=149.5),
            Order(symbol="AAPL", side=TradeSide.BUY, order_type=OrderType.LIMIT,
                  quantity=50, limit_price=150.0, status=OrderStatus.CANCELLED),
        ]
        fills = [
            Fill(order_id=orders[0].order_id, fill_price=149.5, fill_quantity=100, timestamp=1.0),
        ]
        result = compute_execution_metrics(orders, fills)
        assert result["total_orders"] == 2
        assert result["filled"] == 1
        assert result["cancelled"] == 1
        assert result["fill_rate"] == pytest.approx(0.5)
        assert result["avg_price_improvement"] == pytest.approx(0.5)  # 150 - 149.5


# ---------------------------------------------------------------------------
# Capacity Analysis
# ---------------------------------------------------------------------------

class TestCapacityAnalysis:
    def test_capacity_degradation(self):
        results = {
            10_000: {"sharpe": 2.1, "total_return": 0.21, "max_drawdown": -0.05, "avg_slippage_bps": 2, "fill_rate": 0.99},
            100_000: {"sharpe": 1.8, "total_return": 0.18, "max_drawdown": -0.07, "avg_slippage_bps": 5, "fill_rate": 0.95},
            1_000_000: {"sharpe": 0.7, "total_return": 0.07, "max_drawdown": -0.15, "avg_slippage_bps": 20, "fill_rate": 0.80},
        }
        analysis = compute_capacity_analysis(results)
        assert analysis["baseline_sharpe"] == pytest.approx(2.1)
        assert len(analysis["profiles"]) == 3
        # Last profile should show significant degradation
        assert analysis["profiles"][-1]["sharpe_degradation_pct"] > 50

    def test_empty_results(self):
        result = compute_capacity_analysis({})
        assert "error" in result


# ---------------------------------------------------------------------------
# Latency Model
# ---------------------------------------------------------------------------

class TestLatencyModel:
    def test_zero_latency(self):
        model = LatencyModel()
        assert model.total_latency_ms == 0

    def test_custom_latency(self):
        model = LatencyModel(signal_ms=0.2, network_ms=0.5, processing_ms=0.15, exchange_ms=0.2)
        assert model.total_latency_ms == pytest.approx(1.05)

    def test_apply_to_timestamp(self):
        model = LatencyModel(signal_ms=1.0)
        assert model.apply_to_timestamp(100.0) == pytest.approx(100.001)

    def test_presets_exist(self):
        assert "zero" in LATENCY_PRESETS
        assert "colocated" in LATENCY_PRESETS
        assert "retail" in LATENCY_PRESETS

    def test_get_preset(self):
        model = get_preset("retail")
        assert model.total_latency_ms == pytest.approx(50.0)

    def test_invalid_preset_raises(self):
        with pytest.raises(ValueError):
            get_preset("nonexistent")

    def test_to_dict(self):
        model = LatencyModel(signal_ms=1, network_ms=2, processing_ms=3, exchange_ms=4)
        d = model.to_dict()
        assert d["total_latency_ms"] == pytest.approx(10.0)

    def test_sensitivity_analysis(self):
        results = {
            "zero": {"sharpe": 2.0, "total_return": 0.2, "max_drawdown": -0.05, "fill_rate": 1.0},
            "retail": {"sharpe": 1.0, "total_return": 0.1, "max_drawdown": -0.10, "fill_rate": 0.8},
        }
        analysis = latency_sensitivity_analysis(results)
        assert len(analysis["profiles"]) == 2
        assert analysis["profiles"][1]["sharpe_degradation_pct"] == pytest.approx(50.0)
