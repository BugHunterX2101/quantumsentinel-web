"""QuantumSentinel — Latency Model.

Configurable latency simulation for realistic paper-trading execution.
Models the real-world delay chain:

    Signal calculation delay
    + Network delay
    + Order processing delay
    + Exchange simulation delay

Usage
-----
>>> model = LatencyModel(signal_ms=0.2, network_ms=0.5, processing_ms=0.15, exchange_ms=0.2)
>>> model.total_latency_ms  # 1.05
>>> model.apply_to_timestamp(100.0)  # 100.00105

The model is intentionally configurable so users can run sensitivity
analysis across latency profiles:

    0 ms   → ideal / no-latency baseline
    1 ms   → co-located / same-datacenter
    5 ms   → regional cloud
    10 ms  → cross-region
    50 ms  → retail / overseas
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(slots=True)
class LatencyModel:
    """Configurable latency model for paper-trading simulation.

    Each component models a distinct stage in the real-world order path.
    """
    signal_ms: float = 0.0       # Time to compute the trading signal
    network_ms: float = 0.0      # Network round-trip to exchange
    processing_ms: float = 0.0   # Order management system processing
    exchange_ms: float = 0.0     # Exchange's internal matching delay

    @property
    def total_latency_ms(self) -> float:
        """Total one-way latency in milliseconds."""
        return self.signal_ms + self.network_ms + self.processing_ms + self.exchange_ms

    @property
    def total_latency_seconds(self) -> float:
        """Total one-way latency in seconds."""
        return self.total_latency_ms / 1000.0

    def apply_to_timestamp(self, event_timestamp: float) -> float:
        """Return the effective timestamp after applying latency."""
        return event_timestamp + self.total_latency_seconds

    def to_dict(self) -> dict:
        return {
            "signal_ms": self.signal_ms,
            "network_ms": self.network_ms,
            "processing_ms": self.processing_ms,
            "exchange_ms": self.exchange_ms,
            "total_latency_ms": round(self.total_latency_ms, 4),
        }


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

LATENCY_PRESETS: dict[str, LatencyModel] = {
    "zero":        LatencyModel(0, 0, 0, 0),
    "colocated":   LatencyModel(0.05, 0.01, 0.02, 0.02),       # ~0.1 ms
    "datacenter":  LatencyModel(0.1, 0.2, 0.1, 0.1),            # ~0.5 ms
    "regional":    LatencyModel(0.2, 1.0, 0.3, 0.5),            # ~2 ms
    "cloud":       LatencyModel(0.5, 3.0, 0.5, 1.0),            # ~5 ms
    "cross_region": LatencyModel(0.5, 8.0, 0.5, 1.0),           # ~10 ms
    "retail":      LatencyModel(1.0, 30.0, 5.0, 14.0),          # ~50 ms
}


def get_preset(name: str) -> LatencyModel:
    """Get a named latency preset."""
    if name not in LATENCY_PRESETS:
        raise ValueError(
            f"Unknown latency preset '{name}'. "
            f"Available: {', '.join(LATENCY_PRESETS.keys())}"
        )
    preset = LATENCY_PRESETS[name]
    # Return a copy so presets are immutable
    return LatencyModel(
        signal_ms=preset.signal_ms,
        network_ms=preset.network_ms,
        processing_ms=preset.processing_ms,
        exchange_ms=preset.exchange_ms,
    )


# ---------------------------------------------------------------------------
# Sensitivity Analysis
# ---------------------------------------------------------------------------

def latency_sensitivity_analysis(
    strategy_results: dict[str, dict],
) -> dict:
    """Analyze how strategy performance varies across latency profiles.

    Parameters
    ----------
    strategy_results : dict
        Keys are preset names, values are dicts with at least
        ``sharpe``, ``total_return``, ``max_drawdown``, ``fill_rate``.

    Returns
    -------
    dict
        Summary table + degradation metrics.
    """
    presets = list(strategy_results.keys())
    if not presets:
        return {"error": "no results provided"}

    rows = []
    baseline = strategy_results.get("zero", strategy_results[presets[0]])
    baseline_sharpe = baseline.get("sharpe", 0)

    for name, result in strategy_results.items():
        sharpe = result.get("sharpe", 0)
        degradation = (
            (baseline_sharpe - sharpe) / baseline_sharpe * 100
            if baseline_sharpe != 0 else 0
        )
        rows.append({
            "preset": name,
            "latency_ms": LATENCY_PRESETS.get(name, LatencyModel()).total_latency_ms,
            "sharpe": round(sharpe, 4),
            "total_return": round(result.get("total_return", 0), 4),
            "max_drawdown": round(result.get("max_drawdown", 0), 4),
            "fill_rate": round(result.get("fill_rate", 0), 4),
            "sharpe_degradation_pct": round(degradation, 2),
        })

    return {
        "baseline_preset": presets[0],
        "baseline_sharpe": baseline_sharpe,
        "profiles": rows,
    }
