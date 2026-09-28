"""QuantumSentinel — L2 Event Replay Engine.

Replays time-stamped Level-2 events through one incrementally maintained
order book and, optionally, a strategy trading through the paper exchange:

    L2 event → order-book update → snapshot → strategy → signal →
    paper order (latency) → queue-aware matching → fill → portfolio →
    execution analytics

Data sources (``L2EventStream``):

* ``from_synthetic`` — SYNTHETIC L2 generated from OHLCV bars. It is a
  consistent book history for research, not historical venue data.
* ``from_records`` / ``from_csv`` — external L2 (e.g. vendor exports),
  validated against market-data invariants on load.
* ``from_events`` — a pre-built event list, used as given.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass, field
from typing import Callable, Iterator

from backend.services.market_microstructure import (
    BookEvent,
    BookEventType,
    L2DataError,
    OrderBookSnapshot,
    TradeEvent,
    TradeSide,
    generate_synthetic_l2,
    snapshot_to_dict,
    validate_events,
)


# ---------------------------------------------------------------------------
# L2 Event Stream
# ---------------------------------------------------------------------------

@dataclass
class L2EventStream:
    """Time-sorted sequence of L2 book events with a provenance hash."""
    events: list[BookEvent]
    dataset_id: str = ""
    dataset_hash: str = ""
    source: str = "external"          # "synthetic" | "external"

    def __post_init__(self):
        if not self.dataset_hash:
            # Hash the complete, canonical replay input: every event and every
            # field that affects book state and matching, JSON-framed.
            canonical_events = [
                {
                    "timestamp": event.timestamp,
                    "event_type": event.event_type.value,
                    "side": event.side.value,
                    "price": event.price,
                    "size": event.size,
                    "order_id": event.order_id,
                }
                for event in self.events
            ]
            raw = json.dumps(canonical_events, separators=(",", ":"), ensure_ascii=False)
            self.dataset_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def __iter__(self) -> Iterator[BookEvent]:
        return iter(self.events)

    def __len__(self) -> int:
        return len(self.events)

    @classmethod
    def from_synthetic(
        cls,
        ohlcv_bars: list[dict],
        levels: int = 10,
        base_spread_bps: float = 10.0,
        base_depth: float = 100.0,
        events_per_bar: int = 50,
        seed: int = 42,
        dataset_id: str = "synthetic",
        tick_size: float = 0.01,
    ) -> L2EventStream:
        """Synthetic L2 generated from OHLCV bars (not historical venue data)."""
        events = generate_synthetic_l2(
            ohlcv_bars, levels=levels, base_spread_bps=base_spread_bps, base_depth=base_depth,
            events_per_bar=events_per_bar, seed=seed, tick_size=tick_size,
        )
        return cls(events=events, dataset_id=dataset_id, source="synthetic")

    @classmethod
    def from_events(cls, events: list[BookEvent], dataset_id: str = "raw") -> L2EventStream:
        """Wrap a pre-built event list as given."""
        return cls(events=events, dataset_id=dataset_id)

    @classmethod
    def from_records(cls, records: list[dict], dataset_id: str = "csv",
                     tick_size: float = 0.01, validate: bool = True) -> L2EventStream:
        """Build from dicts with keys timestamp, event_type, side, price, size[, order_id].

        Records are stably sorted by timestamp and, when ``validate`` is set,
        rejected with ``L2DataError`` if they violate a market-data invariant.
        """
        events = []
        for i, r in enumerate(records):
            try:
                events.append(BookEvent(
                    timestamp=float(r["timestamp"]),
                    event_type=BookEventType(str(r["event_type"]).upper()),
                    side=TradeSide(str(r["side"]).upper()),
                    price=float(r["price"]),
                    size=float(r["size"]),
                    order_id=(str(r["order_id"]) if r.get("order_id") not in (None, "") else None),
                ))
            except (KeyError, ValueError, TypeError) as exc:
                raise L2DataError(f"record {i}: {exc}") from exc
        events.sort(key=lambda e: e.timestamp)
        if validate:
            issues = validate_events(events, tick_size=tick_size)
            if issues:
                raise L2DataError("; ".join(issues))
        return cls(events=events, dataset_id=dataset_id)

    @classmethod
    def from_csv(cls, text: str, dataset_id: str = "csv", tick_size: float = 0.01,
                 validate: bool = True) -> L2EventStream:
        """Parse CSV text with a header row naming the ``from_records`` fields."""
        return cls.from_records(list(csv.DictReader(io.StringIO(text))), dataset_id=dataset_id,
                                tick_size=tick_size, validate=validate)


# ---------------------------------------------------------------------------
# Replay Session
# ---------------------------------------------------------------------------

@dataclass
class ReplayConfig:
    """Configuration for an L2 replay session."""
    snapshot_interval: int = 10      # Snapshot (and consult the strategy) every N events
    warmup_events: int = 100         # Events before the strategy is consulted
    max_events: int | None = None    # Cap for debugging / partial replays
    snapshot_levels: int = 10
    tick_size: float = 0.01
    # Execution — used when replay_session is given a PaperExchange
    order_quantity: float = 100.0
    order_style: str = "aggressive"  # "aggressive": market orders | "passive": join the best quote
    cancel_after_events: int | None = None   # cancel a passive order unfilled after N events
    adverse_selection_horizons_s: tuple[float, ...] = (1.0, 10.0, 60.0)


@dataclass
class ReplayResult:
    """Result of an L2 replay session."""
    total_events: int = 0
    snapshots_generated: int = 0
    strategy_signals: int = 0
    trades_observed: int = 0
    dataset_id: str = ""
    dataset_hash: str = ""
    data_source: str = ""
    snapshot_history: list[dict] = field(default_factory=list)
    signal_history: list[dict] = field(default_factory=list)
    execution: dict | None = None

    def to_dict(self) -> dict:
        d = {
            "total_events": self.total_events,
            "snapshots_generated": self.snapshots_generated,
            "strategy_signals": self.strategy_signals,
            "trades_observed": self.trades_observed,
            "dataset_id": self.dataset_id,
            "dataset_hash": self.dataset_hash,
            "data_source": self.data_source,
            "snapshot_count": len(self.snapshot_history),
            "signal_count": len(self.signal_history),
        }
        if self.execution is not None:
            d["execution"] = self.execution
        return d


_WORKING_STATUSES = ("SUBMITTED", "ACCEPTED", "QUEUED", "PARTIALLY_FILLED")


def replay_session(
    stream: L2EventStream,
    strategy_fn: Callable[[OrderBookSnapshot, list[TradeEvent]], dict | None] | None = None,
    config: ReplayConfig | None = None,
    exchange=None,
) -> ReplayResult:
    """Replay an L2 stream through one incrementally updated book.

    Parameters
    ----------
    stream : L2EventStream
    strategy_fn : callable, optional
        ``(snapshot, recent_trades) -> {"signal": "BUY"|"SELL", ...} | None``,
        consulted every ``snapshot_interval`` events after warm-up.
    config : ReplayConfig, optional
    exchange : PaperExchange, optional
        When given, events flow through the exchange and signals become
        paper orders: BUY opens ``order_quantity`` when flat, SELL closes
        the position (long only). Orders are subject to the exchange's
        latency and queue model, and the result carries its fills,
        portfolio and execution analytics.
    """
    from backend.services.order_book import OrderBook, OrderType

    config = config or ReplayConfig()
    result = ReplayResult(dataset_id=stream.dataset_id, dataset_hash=stream.dataset_hash,
                          data_source=stream.source)
    book = exchange.book if exchange is not None else OrderBook(tick_size=config.tick_size)
    recent_trades: list[TradeEvent] = []
    working: dict[str, int] = {}          # passive order id -> event index at submission

    for i, event in enumerate(stream):
        if config.max_events is not None and i >= config.max_events:
            break
        result.total_events += 1
        if exchange is not None:
            exchange.on_market_event(event)
        else:
            book.apply_market_event(event)

        if event.event_type == BookEventType.TRADE:
            result.trades_observed += 1
            recent_trades.append(TradeEvent(timestamp=event.timestamp, price=event.price,
                                            size=event.size, aggressor_side=event.side))
            if len(recent_trades) > 100:
                recent_trades = recent_trades[-100:]

        if exchange is not None and config.cancel_after_events is not None:
            for oid, submitted_at in list(working.items()):
                order = exchange._by_id.get(oid)
                if order is None or order.status.value not in _WORKING_STATUSES:
                    working.pop(oid)
                elif i - submitted_at >= config.cancel_after_events:
                    exchange.cancel_order(oid)
                    working.pop(oid)

        if (i + 1) % config.snapshot_interval != 0:
            continue
        snap = book.snapshot(config.snapshot_levels)
        result.snapshots_generated += 1
        result.snapshot_history.append(snapshot_to_dict(snap))
        if len(result.snapshot_history) > 500:
            result.snapshot_history = result.snapshot_history[-250:]

        if not strategy_fn or i < config.warmup_events:
            continue
        signal = strategy_fn(snap, recent_trades)
        if signal is None:
            continue
        signal["timestamp"] = event.timestamp
        result.strategy_signals += 1
        if exchange is not None:
            order = _order_for_signal(exchange, signal.get("signal"), config, OrderType)
            if order is not None:
                signal["order_id"] = order.order_id
                signal["order_status"] = order.status.value
                if config.order_style == "passive":
                    working[order.order_id] = i
        result.signal_history.append(signal)

    if exchange is not None:
        result.execution = {
            "portfolio": exchange.portfolio_summary(),
            "exchange_stats": exchange.exchange_stats(),
            "orders": exchange.order_summary(),
            "fills": [f.to_dict() for f in exchange.fill_history],
            "analytics": exchange.execution_report(config.adverse_selection_horizons_s),
        }
    return result


def _order_for_signal(exchange, signal: str | None, config: ReplayConfig, OrderType):
    """Translate a BUY/SELL signal into a long-only paper order, or None."""
    if any(o.status.value in _WORKING_STATUSES for o in exchange.order_history):
        return None
    held = exchange.position_quantity()
    if signal == "BUY" and held <= 0:
        side, qty = TradeSide.BUY, config.order_quantity
    elif signal == "SELL" and held > 0:
        side, qty = TradeSide.SELL, held
    else:
        return None
    if config.order_style == "passive":
        price = exchange.book.best_bid if side == TradeSide.BUY else exchange.book.best_ask
        if price is None:
            return None
        return exchange.submit_order(side, qty, OrderType.LIMIT, limit_price=price)
    return exchange.submit_order(side, qty, OrderType.MARKET)


# ---------------------------------------------------------------------------
# Built-in Research Strategies
# ---------------------------------------------------------------------------

def obi_momentum_strategy(
    snap: OrderBookSnapshot,
    recent_trades: list[TradeEvent],
    obi_threshold: float = 0.3,
) -> dict | None:
    """Simple OBI-momentum strategy for research / demonstration.

    Generates a BUY signal when OBI > threshold (strong bid pressure)
    and a SELL signal when OBI < -threshold (strong ask pressure).
    """
    from backend.services.market_microstructure import (
        compute_order_book_imbalance,
        compute_microprice,
        compute_spread_bps,
        compute_trade_imbalance,
    )

    obi = compute_order_book_imbalance(snap, levels=5)
    if obi is None:
        return None

    microprice = compute_microprice(snap)
    spread_bps = compute_spread_bps(snap)
    trade_imb = compute_trade_imbalance(recent_trades, window_seconds=30.0)

    if abs(obi) < obi_threshold:
        return None

    return {
        "signal": "BUY" if obi > 0 else "SELL",
        "strength": round(abs(obi), 4),
        "obi": round(obi, 4),
        "microprice": round(microprice, 4) if microprice else None,
        "spread_bps": round(spread_bps, 2) if spread_bps else None,
        "trade_imbalance": round(trade_imb, 4) if trade_imb is not None else None,
    }
