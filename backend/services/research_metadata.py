"""Reproducibility and point-in-time metadata for research experiments."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetMetadata:
    """Data lineage recorded with every experiment.

    The object deliberately captures known Yahoo Finance limitations rather
    than implying that a daily-bar source has a survivorship-free universe.
    """
    source: str
    frequency: str
    adjusted: bool
    retrieved_at: str
    timezone: str = "UTC"
    corporate_action_adjustment: str = "vendor_adjusted"
    missing_data_handling: str = "drop_incomplete_rows"
    survivorship_universe: str = "not_survivorship_free"
    calendar: str = "source_exchange_calendar"
    delisting_handling: str = "not_available_from_source"
    point_in_time_validated: bool = False

    @classmethod
    def yahoo_daily(cls) -> "DatasetMetadata":
        return cls(source="Yahoo Finance", frequency="1d", adjusted=True,
                   retrieved_at=datetime.now(timezone.utc).isoformat())

    def validate(self) -> list[str]:
        warnings: list[str] = []
        if not self.point_in_time_validated:
            warnings.append("Dataset has not been validated as point-in-time safe.")
        if self.survivorship_universe == "not_survivorship_free":
            warnings.append("Universe may contain survivorship bias.")
        if self.delisting_handling == "not_available_from_source":
            warnings.append("Delisting handling is unavailable from this source.")
        return warnings


def data_hash(data: Any) -> str:
    """Stable SHA-256 hash for serialisable research inputs/metadata."""
    if hasattr(data, "tolist"):
        data = data.tolist()
    raw = json.dumps(data, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def git_commit() -> str | None:
    try:
        root = Path(__file__).resolve().parents[2]
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.SubprocessError):
        return None


@dataclass
class ExperimentRecord:
    experiment_id: str
    dataset: DatasetMetadata
    assets: list[str]
    date_range: str
    strategy: str
    strategy_parameters: dict[str, Any]
    execution_model: str
    random_seed: int | None = None
    feature_version: str = "unknown"
    parameter_trials: int = 1
    git_commit_hash: str | None = field(default_factory=git_commit)
    data_hash: str | None = None
    in_sample_performance: dict[str, Any] = field(default_factory=dict)
    out_of_sample_performance: dict[str, Any] = field(default_factory=dict)
    statistical_results: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, *, dataset: DatasetMetadata, assets: list[str], date_range: str,
               strategy: str, strategy_parameters: dict[str, Any], execution_model: str,
               random_seed: int | None = None, feature_version: str = "unknown",
               parameter_trials: int = 1, **kwargs: Any) -> "ExperimentRecord":
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        digest = data_hash({"assets": assets, "strategy": strategy, "at": stamp})[:6]
        return cls(f"QS-{stamp}-{digest}", dataset, assets, date_range, strategy,
                   strategy_parameters, execution_model, random_seed, feature_version,
                   parameter_trials, data_hash=data_hash(asdict(dataset)), **kwargs)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["dataset"]["warnings"] = self.dataset.validate()
        return value
