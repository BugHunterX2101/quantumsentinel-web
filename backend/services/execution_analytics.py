"""QuantumSentinel — Execution Analytics.

Post-trade analytics for the paper exchange, covering:
- Adverse selection analysis
- Implementation shortfall decomposition
- Queue-position analytics
- Execution quality metrics
- Capacity analysis

All functions return plain Python dicts for JSON serialisation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from backend.services.market_microstructure import (
    OrderBookSnapshot,
    TradeSide,
    compute_mid_price,
)
from backend.services.order_book import Fill, Order, OrderStatus


# ---------------------------------------------------------------------------
# Adverse Selection
# ---------------------------------------------------------------------------

def compute_adverse_selection(
    fills: Sequence[dict],
    mid_prices_after: dict[str, list[tuple[float, float]]],
    horizons_ms: Sequence[float] = (5, 50, 500, 5000),
) -> list[dict]:
    """Measure adverse selection for each fill.

    For each fill, computes Mid_{t+Δ} − FillPrice at multiple horizons.
    For buys: negative values indicate adverse selection (price fell after buy).
    For sells: negative values indicate adverse selection (price rose after sell).

    Parameters
    ----------
    fills : list of dicts
        Each must have ``fill_price``, ``timestamp``, ``side``, ``order_id``.
    mid_prices_after : dict
        order_id → list of (horizon_ms, mid_price_at_horizon).
    horizons_ms : sequence of float
        Horizons to measure at (in milliseconds).

    Returns
    -------
    list of dicts
        Per-fill adverse selection at each horizon.
    """
    results = []
    for fill in fills:
        oid = fill.get("order_id", "")
        fill_price = fill.get("fill_price", 0)
        side = fill.get("side", "BUY")
        horizon_data = mid_prices_after.get(oid, [])

        row = {
            "order_id": oid,
            "fill_price": fill_price,
            "side": side,
            "horizons": {},
        }

        for h_ms, mid_after in horizon_data:
            if side == "BUY":
                adv_sel = mid_after - fill_price
            else:
                adv_sel = fill_price - mid_after
            row["horizons"][f"{h_ms}ms"] = round(adv_sel, 6)

        results.append(row)

    return results


def compute_adverse_selection_summary(
    per_fill_results: list[dict],
) -> dict:
    """Aggregate adverse-selection metrics across all fills.

    Returns mean adverse selection at each horizon and the fraction of
    fills experiencing adverse selection (negative value).
    """
    if not per_fill_results:
        return {}

    all_horizons: set[str] = set()
    for r in per_fill_results:
        all_horizons.update(r.get("horizons", {}).keys())

    summary: dict[str, dict] = {}
    for h in sorted(all_horizons):
        vals = [r["horizons"][h] for r in per_fill_results if h in r.get("horizons", {})]
        if vals:
            mean_as = sum(vals) / len(vals)
            adverse_count = sum(1 for v in vals if v < 0)
            summary[h] = {
                "mean_adverse_selection": round(mean_as, 6),
                "adverse_fraction": round(adverse_count / len(vals), 4),
                "count": len(vals),
            }

    return summary


# ---------------------------------------------------------------------------
# Implementation Shortfall Decomposition
# ---------------------------------------------------------------------------

def compute_implementation_shortfall(
    decision_price: float,
    arrival_price: float,
    execution_vwap: float,
    side: str,
    quantity: float,
    spread: float = 0.0,
    fees: float = 0.0,
) -> dict:
    """Decompose implementation shortfall into components.

    IS = Spread + MarketImpact + Delay + Fees

    Parameters
    ----------
    decision_price : float
        Price at signal generation time.
    arrival_price : float
        Price when order reaches the exchange.
    execution_vwap : float
        Volume-weighted average execution price.
    side : str
        "BUY" or "SELL".
    quantity : float
        Order quantity.
    spread : float
        Half-spread cost (one-way).
    fees : float
        Total transaction fees.

    Returns
    -------
    dict
        Decomposition: delay_cost, spread_cost, market_impact, fees,
        total_is, total_is_bps.
    """
    sign = 1.0 if side == "BUY" else -1.0

    # Delay cost: price moved between decision and arrival
    delay_cost = sign * (arrival_price - decision_price) * quantity

    # Spread cost
    spread_cost = spread * quantity

    # Market impact: execution price vs arrival price
    market_impact = sign * (execution_vwap - arrival_price) * quantity

    total_is = delay_cost + spread_cost + market_impact + fees

    # Express in basis points relative to notional
    notional = decision_price * quantity
    total_is_bps = (total_is / notional * 10_000) if notional > 0 else 0

    return {
        "decision_price": round(decision_price, 6),
        "arrival_price": round(arrival_price, 6),
        "execution_vwap": round(execution_vwap, 6),
        "side": side,
        "quantity": quantity,
        "delay_cost": round(delay_cost, 4),
        "spread_cost": round(spread_cost, 4),
        "market_impact": round(market_impact, 4),
        "fees": round(fees, 4),
        "total_is": round(total_is, 4),
        "total_is_bps": round(total_is_bps, 2),
    }


# ---------------------------------------------------------------------------
# Queue-Position Analytics
# ---------------------------------------------------------------------------

def compute_queue_analytics(orders: Sequence[Order]) -> dict:
    """Compute queue-position analytics for a set of paper orders.

    Returns aggregate statistics on queue position, fill probability,
    and time-in-queue metrics.
    """
    filled = [o for o in orders if o.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED)]
    resting = [o for o in orders if o.status == OrderStatus.QUEUED]
    all_active = filled + resting

    if not all_active:
        return {"total_orders": 0}

    queue_at_entry = [o.queue_ahead_at_entry for o in all_active]
    queue_at_peak = [o.queue_ahead_peak for o in all_active]

    fill_times = []
    for o in filled:
        if o.entered_book_at is not None and o.filled_at is not None:
            ft = o.filled_at - o.entered_book_at
            if ft >= 0:
                fill_times.append(ft)

    partial_fills = [o for o in filled if o.status == OrderStatus.PARTIALLY_FILLED]

    fill_rate = len(filled) / len(all_active) if all_active else 0
    partial_rate = len(partial_fills) / len(filled) if filled else 0

    return {
        "total_orders": len(all_active),
        "filled": len(filled),
        "partially_filled": len(partial_fills),
        "resting": len(resting),
        "fill_rate": round(fill_rate, 4),
        "partial_fill_rate": round(partial_rate, 4),
        "avg_queue_ahead_at_entry": round(sum(queue_at_entry) / len(queue_at_entry), 2) if queue_at_entry else 0,
        "avg_queue_ahead_peak": round(sum(queue_at_peak) / len(queue_at_peak), 2) if queue_at_peak else 0,
        "avg_fill_time_seconds": round(sum(fill_times) / len(fill_times), 4) if fill_times else None,
    }


# ---------------------------------------------------------------------------
# Execution Quality Metrics
# ---------------------------------------------------------------------------

def compute_execution_metrics(
    orders: Sequence[Order],
    fills: Sequence[Fill],
    mid_prices: dict[str, float] | None = None,
) -> dict:
    """Compute comprehensive execution quality metrics.

    Returns fill rate, partial fill rate, cancel rate, average queue time,
    spread paid, price improvement, and more.
    """
    total = len(orders)
    if total == 0:
        return {"total_orders": 0}

    filled_orders = [o for o in orders if o.status == OrderStatus.FILLED]
    partial_orders = [o for o in orders if o.status == OrderStatus.PARTIALLY_FILLED]
    cancelled_orders = [o for o in orders if o.status == OrderStatus.CANCELLED]
    rejected_orders = [o for o in orders if o.status == OrderStatus.REJECTED]

    # Fill rate
    fill_rate = (len(filled_orders) + len(partial_orders)) / total
    cancel_rate = len(cancelled_orders) / total

    # Average fill price vs limit price (price improvement)
    price_improvements = []
    for o in filled_orders:
        if o.limit_price and o.avg_fill_price:
            if o.side == TradeSide.BUY:
                improvement = o.limit_price - o.avg_fill_price
            else:
                improvement = o.avg_fill_price - o.limit_price
            price_improvements.append(improvement)

    # Turnover (total filled notional)
    total_notional = sum(
        f.fill_price * f.fill_quantity for f in fills
    )

    return {
        "total_orders": total,
        "filled": len(filled_orders),
        "partially_filled": len(partial_orders),
        "cancelled": len(cancelled_orders),
        "rejected": len(rejected_orders),
        "fill_rate": round(fill_rate, 4),
        "cancel_rate": round(cancel_rate, 4),
        "avg_price_improvement": round(
            sum(price_improvements) / len(price_improvements), 6
        ) if price_improvements else 0,
        "total_notional": round(total_notional, 2),
        "total_fills": len(fills),
    }


# ---------------------------------------------------------------------------
# Capacity Analysis
# ---------------------------------------------------------------------------

def compute_capacity_analysis(
    strategy_results_by_capital: dict[float, dict],
) -> dict:
    """Analyze strategy capacity by running at different capital levels.

    Parameters
    ----------
    strategy_results_by_capital : dict
        Keys are capital amounts (e.g. 10000, 50000, 100000),
        values are dicts with ``sharpe``, ``total_return``,
        ``max_drawdown``, ``avg_slippage_bps``, ``fill_rate``.

    Returns
    -------
    dict
        Capacity profile with degradation analysis.
    """
    if not strategy_results_by_capital:
        return {"error": "no results provided"}

    capitals = sorted(strategy_results_by_capital.keys())
    baseline_sharpe = strategy_results_by_capital[capitals[0]].get("sharpe", 0)

    rows = []
    for cap in capitals:
        r = strategy_results_by_capital[cap]
        sharpe = r.get("sharpe", 0)
        # Normalize by |baseline_sharpe|, not baseline_sharpe itself — a
        # negative baseline would otherwise flip the sign, reporting a
        # genuine degradation (sharpe getting more negative) as a
        # "negative degradation" (i.e. an apparent improvement). E.g.
        # baseline -1.0 -> -2.0 is real degradation but (−1−−2)/−1*100 = −100%.
        degradation = (
            (baseline_sharpe - sharpe) / abs(baseline_sharpe) * 100
            if baseline_sharpe != 0 else 0
        )
        rows.append({
            "capital": cap,
            "sharpe": round(sharpe, 4),
            "total_return": round(r.get("total_return", 0), 4),
            "max_drawdown": round(r.get("max_drawdown", 0), 4),
            "avg_slippage_bps": round(r.get("avg_slippage_bps", 0), 2),
            "fill_rate": round(r.get("fill_rate", 0), 4),
            "sharpe_degradation_pct": round(degradation, 2),
        })

    # Estimate capacity as capital where Sharpe drops below 50% of baseline
    capacity_estimate = None
    for row in rows:
        if row["sharpe_degradation_pct"] > 50:
            capacity_estimate = row["capital"]
            break

    return {
        "baseline_capital": capitals[0],
        "baseline_sharpe": round(baseline_sharpe, 4),
        "capacity_estimate": capacity_estimate,
        "profiles": rows,
    }
