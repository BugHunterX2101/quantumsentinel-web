"""QuantumSentinel — L2 Event Replay Engine.

Replays time-stamped Level-2 order-book events through strategies and
the paper exchange, producing fills that respect queue position and
realistic latency.

The replay engine is the bridge between:
  - Historical or synthetic L2 data
  - The paper exchange / matching engine
  - Research strategies that consume microstructure signals

Usage
-----
>>> from backend.services.l2_event_replay import L2EventStream, replay_session
>>> stream = L2EventStream.from_synthetic(ohlcv_bars, seed=42)
>>> results = replay_session(stream, strategy_fn, exchange)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable, Iterator, Sequence

from backend.services.market_microstructure import (
    BookEvent,
    BookEventType,
    OrderBookSnapshot,
    PriceLevel,
    TradeEvent,
    TradeSide,
    build_snapshot_from_events,
    generate_synthetic_l2,
    snapshot_to_dict,
)


# ---------------------------------------------------------------------------
# L2 Event Stream
# ---------------------------------------------------------------------------

@dataclass
class L2EventStream:
    """Iterable container for a sequence of time-sorted L2 book events.

    The stream can be created from:
    - Synthetic generation (``from_synthetic``)
    - Raw event lists (``from_events``)
    - CSV/dict data (``from_records``)
    """
    events: list[BookEvent]
    dataset_id: str = ""
    dataset_hash: str = ""

    def __post_init__(self):
        if not self.dataset_hash:
            # Hash the complete, canonical replay input. A prefix hash could
            # attest two streams that diverge after event 1,000, while a
            # concatenated string omitted fields (side/order identity) that
            # affect book state and matching. JSON framing also avoids field
            # boundary ambiguities inherent in simple string concatenation.
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
    ) -> L2EventStream:
        """Generate a stream from OHLCV daily bars."""
        events = generate_synthetic_l2(
            ohlcv_bars,
            levels=levels,
            base_spread_bps=base_spread_bps,
            base_depth=base_depth,
            events_per_bar=events_per_bar,
            seed=seed,
        )
        return cls(events=events, dataset_id=dataset_id)

    @classmethod
    def from_events(cls, events: list[BookEvent], dataset_id: str = "raw") -> L2EventStream:
        """Wrap a pre-built event list."""
        return cls(events=events, dataset_id=dataset_id)

    @classmethod
    def from_records(cls, records: list[dict], dataset_id: str = "csv") -> L2EventStream:
        """Build from dicts with keys: timestamp, event_type, side, price, size."""
        events = []
        for r in records:
            events.append(BookEvent(
                timestamp=float(r["timestamp"]),
                event_type=BookEventType(r["event_type"]),
                side=TradeSide(r["side"]),
                price=float(r["price"]),
                size=float(r["size"]),
                order_id=r.get("order_id"),
            ))
        events.sort(key=lambda e: e.timestamp)
        return cls(events=events, dataset_id=dataset_id)


# ---------------------------------------------------------------------------
# Replay Session
# ---------------------------------------------------------------------------

@dataclass
class ReplayConfig:
    """Configuration for an L2 replay session."""
    snapshot_interval: int = 10      # Build snapshot every N events
    warmup_events: int = 100         # Events to skip before strategy starts
    max_events: int | None = None    # Cap for debugging / partial replays


@dataclass
class ReplayResult:
    """Result of an L2 replay session."""
    total_events: int = 0
    snapshots_generated: int = 0
    strategy_signals: int = 0
    trades_observed: int = 0
    dataset_id: str = ""
    dataset_hash: str = ""
    snapshot_history: list[dict] = field(default_factory=list)
    signal_history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_events": self.total_events,
            "snapshots_generated": self.snapshots_generated,
            "strategy_signals": self.strategy_signals,
            "trades_observed": self.trades_observed,
            "dataset_id": self.dataset_id,
            "dataset_hash": self.dataset_hash,
            "snapshot_count": len(self.snapshot_history),
            "signal_count": len(self.signal_history),
        }


def replay_session(
    stream: L2EventStream,
    strategy_fn: Callable[[OrderBookSnapshot, list[TradeEvent]], dict | None] | None = None,
    config: ReplayConfig | None = None,
) -> ReplayResult:
    """Replay an L2 event stream, building snapshots and invoking the strategy.

    Parameters
    ----------
    stream : L2EventStream
        The event data to replay.
    strategy_fn : callable, optional
        A function ``(snapshot, recent_trades) -> signal_dict | None``.
        Called at each snapshot interval after the warmup period.
        If None, only snapshots and analytics are computed.
    config : ReplayConfig, optional
        Replay parameters.

    Returns
    -------
    ReplayResult
        Summary of the replay including snapshot history and signals.
    """
    if config is None:
        config = ReplayConfig()

    result = ReplayResult(
        dataset_id=stream.dataset_id,
        dataset_hash=stream.dataset_hash,
    )

    event_buffer: list[BookEvent] = []
    recent_trades: list[TradeEvent] = []

    for i, event in enumerate(stream):
        if config.max_events is not None and i >= config.max_events:
            break

        result.total_events += 1
        event_buffer.append(event)

        # Track trade events
        if event.event_type == BookEventType.TRADE:
            result.trades_observed += 1
            recent_trades.append(TradeEvent(
                timestamp=event.timestamp,
                price=event.price,
                size=event.size,
                aggressor_side=event.side,
            ))
            # Keep last 100 trades
            if len(recent_trades) > 100:
                recent_trades = recent_trades[-100:]

        # Build snapshot at interval
        if len(event_buffer) % config.snapshot_interval == 0:
            snap = build_snapshot_from_events(event_buffer)
            result.snapshots_generated += 1

            snap_dict = snapshot_to_dict(snap)
            result.snapshot_history.append(snap_dict)
            # Keep snapshot history bounded for memory
            if len(result.snapshot_history) > 500:
                result.snapshot_history = result.snapshot_history[-250:]

            # Call strategy after warmup
            if strategy_fn and i >= config.warmup_events:
                signal = strategy_fn(snap, recent_trades)
                if signal is not None:
                    signal["timestamp"] = event.timestamp
                    result.signal_history.append(signal)
                    result.strategy_signals += 1

    return result


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
