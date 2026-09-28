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

def _filled_quantity(o: Order) -> float:
    # An order constructed directly as FILLED (e.g. from stored history)
    # may not carry filled_quantity; its status is authoritative.
    if o.status == OrderStatus.FILLED and o.filled_quantity <= 0:
        return o.quantity
    return o.filled_quantity


def _is_partial(o: Order) -> bool:
    """Some quantity executed but not all — including partials later cancelled/expired."""
    return o.status == OrderStatus.PARTIALLY_FILLED or (
        o.status != OrderStatus.FILLED and _filled_quantity(o) > 0)


def _queue_time(o: Order) -> float | None:
    if o.entered_book_at is None or o.filled_at is None:
        return None
    t = o.filled_at - o.entered_book_at
    return t if t >= 0 else None


def compute_queue_analytics(orders: Sequence[Order]) -> dict:
    """Queue-position analytics for orders that rested on the book.

    ``fill_rate`` is the share of rested orders with any execution (kept for
    compatibility); ``full_fill_rate`` counts only complete fills and
    ``fill_ratio`` is executed quantity / submitted quantity.
    """
    filled = [o for o in orders if o.status == OrderStatus.FILLED or _is_partial(o)]
    resting = [o for o in orders if o.status == OrderStatus.QUEUED and _filled_quantity(o) <= 0]
    all_active = filled + resting

    if not all_active:
        return {"total_orders": 0}

    full = [o for o in filled if o.status == OrderStatus.FILLED]
    partial = [o for o in filled if _is_partial(o)]
    queue_at_entry = [o.queue_ahead_at_entry for o in all_active]
    queue_at_peak = [o.queue_ahead_peak for o in all_active]
    fill_times = [t for t in (_queue_time(o) for o in filled) if t is not None]
    submitted = sum(o.quantity for o in all_active)
    executed = sum(_filled_quantity(o) for o in all_active)

    return {
        "total_orders": len(all_active),
        "filled": len(filled),
        "fully_filled": len(full),
        "partially_filled": len(partial),
        "resting": len(resting),
        "fill_rate": round(len(filled) / len(all_active), 4),
        "full_fill_rate": round(len(full) / len(all_active), 4),
        "partial_fill_rate": round(len(partial) / len(filled), 4) if filled else 0,
        "fill_ratio": round(executed / submitted, 4) if submitted else 0,
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
    """Execution quality metrics with each outcome counted separately.

    * ``any_fill_rate``   — orders with any execution / orders
      (``fill_rate`` is kept as an alias for compatibility)
    * ``full_fill_rate``  — completely filled orders / orders
    * ``partial_fill_rate`` — partially executed orders / orders
    * ``fill_ratio``      — executed quantity / submitted quantity
    * ``cancel_rate`` / ``expiry_rate`` / ``reject_rate`` — orders that ended
      in that state *without* any execution / orders
    """
    total = len(orders)
    if total == 0:
        return {"total_orders": 0}

    full = [o for o in orders if o.status == OrderStatus.FILLED]
    partial = [o for o in orders if _is_partial(o)]
    unfilled = [o for o in orders if _filled_quantity(o) <= 0]
    cancelled = [o for o in unfilled if o.status == OrderStatus.CANCELLED]
    expired = [o for o in unfilled if o.status == OrderStatus.EXPIRED]
    rejected = [o for o in unfilled if o.status == OrderStatus.REJECTED]
    submitted_qty = sum(o.quantity for o in orders if o.status != OrderStatus.REJECTED)
    executed_qty = sum(_filled_quantity(o) for o in orders)

    price_improvements = []
    for o in full:
        if o.limit_price and o.avg_fill_price:
            improvement = (o.limit_price - o.avg_fill_price if o.side == TradeSide.BUY
                           else o.avg_fill_price - o.limit_price)
            price_improvements.append(improvement)
    queue_times = [t for t in (_queue_time(o) for o in orders if _filled_quantity(o) > 0) if t is not None]
    any_fill = len(full) + len(partial)

    return {
        "total_orders": total,
        "filled": len(full),
        "partially_filled": len(partial),
        "cancelled": len(cancelled),
        "expired": len(expired),
        "rejected": len(rejected),
        "fill_rate": round(any_fill / total, 4),
        "any_fill_rate": round(any_fill / total, 4),
        "full_fill_rate": round(len(full) / total, 4),
        "partial_fill_rate": round(len(partial) / total, 4),
        "fill_ratio": round(executed_qty / submitted_qty, 4) if submitted_qty else 0,
        "cancel_rate": round(len(cancelled) / total, 4),
        "expiry_rate": round(len(expired) / total, 4),
        "reject_rate": round(len(rejected) / total, 4),
        "avg_price_improvement": round(
            sum(price_improvements) / len(price_improvements), 6
        ) if price_improvements else 0,
        "avg_queue_time_seconds": round(sum(queue_times) / len(queue_times), 4) if queue_times else None,
        "total_notional": round(sum(f.fill_price * f.fill_quantity for f in fills), 2),
        "total_fills": len(fills),
    }


def build_execution_report(exchange, adverse_selection_horizons_s: Sequence[float] = (1.0, 10.0, 60.0)) -> dict:
    """Execution analytics derived from a PaperExchange's own orders and fills.

    Implementation shortfall uses the book mid when the decision was made
    (order submission) and when the order reached the exchange (after
    latency), so delay cost is the latency cost and market impact includes
    the spread paid. Adverse selection compares each fill with the mid
    observed ``h`` seconds later; horizons beyond the replayed data are
    omitted rather than extrapolated.
    """
    orders = exchange.order_history
    fills = exchange.fill_history
    rows, total_is, notional = [], 0.0, 0.0
    for o in orders:
        qty = o.filled_quantity
        decision = exchange.decision_mid.get(o.order_id)
        arrival = exchange.arrival_mid.get(o.order_id)
        if qty <= 0 or decision is None or arrival is None:
            continue
        row = compute_implementation_shortfall(decision, arrival, o.avg_fill_price, o.side.value, qty)
        row["order_id"] = o.order_id
        rows.append(row)
        total_is += row["total_is"]
        notional += decision * qty

    last_seen = exchange.mid_history[-1][0] if exchange.mid_history else None
    fill_rows, mids_after = [], {}
    for i, f in enumerate(fills):
        key = f"{f.order_id}#{i}"
        fill_rows.append({"order_id": key, "fill_price": f.fill_price,
                          "side": f.side.value if f.side else "BUY"})
        horizon_mids = []
        for h in adverse_selection_horizons_s:
            if last_seen is not None and f.timestamp + h <= last_seen:
                mid = exchange.mid_at(f.timestamp + h)
                if mid is not None:
                    label = int(h * 1000) if float(h * 1000).is_integer() else h * 1000
                    horizon_mids.append((label, mid))
        mids_after[key] = horizon_mids
    per_fill = compute_adverse_selection(fill_rows, mids_after)

    return {
        "execution_metrics": compute_execution_metrics(orders, fills),
        "queue_analytics": compute_queue_analytics([o for o in orders if o.entered_book_at is not None]),
        "implementation_shortfall": {
            "orders": rows,
            "total_is": round(total_is, 4),
            "total_is_bps": round(total_is / notional * 10_000, 2) if notional else 0,
        },
        "adverse_selection": {
            "horizons_seconds": list(adverse_selection_horizons_s),
            "summary": compute_adverse_selection_summary(per_fill),
        },
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
