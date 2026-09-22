"""QuantumSentinel — Experiment Registry & Signed Research Manifests.

Provides versioned experiment tracking with deterministic replay
guarantees and cryptographically signed research manifests.

The experiment chain:

    Dataset → Experiment → Strategy → Parameters → Paper Orders
    → Fills → Portfolio → Metrics → Signed Manifest

Every completed experiment produces a manifest that captures:
- dataset_hash (SHA-256 of input data)
- strategy_hash (SHA-256 of strategy code/version)
- parameter_hash (SHA-256 of serialised parameters)
- code_commit (git HEAD at experiment time)
- result_hash (SHA-256 of all outputs)

The manifest is then ML-DSA-65 signed by the server identity,
creating a provable link between:
    this P&L ↔ this exact dataset + strategy + config + code version + simulator
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
import uuid
import datetime as dt
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Experiment Status
# ---------------------------------------------------------------------------

class ExperimentStatus(str, Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    VALIDATED = "VALIDATED"      # Passed all validation gates
    APPROVED = "APPROVED"        # Approved for paper deployment


# ---------------------------------------------------------------------------
# Experiment Record
# ---------------------------------------------------------------------------

@dataclass
class Experiment:
    """A single research experiment with full provenance tracking."""
    experiment_id: str = field(default_factory=lambda: f"QS-{uuid.uuid4().hex[:8].upper()}")
    strategy_id: str = ""
    strategy_version: str = ""
    dataset_id: str = ""
    dataset_hash: str = ""
    parameter_hash: str = ""
    code_commit: str = ""
    random_seed: int = 42
    execution_model: str = "LOB_QUEUE_V2"
    latency_model: str = "zero"
    status: ExperimentStatus = ExperimentStatus.CREATED
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None

    # Results (populated after completion)
    result_hash: str = ""
    results: dict = field(default_factory=dict)
    validation_gates: dict = field(default_factory=dict)
    manifest_signature: str = ""

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "strategy_id": self.strategy_id,
            "strategy_version": self.strategy_version,
            "dataset_id": self.dataset_id,
            "dataset_hash": self.dataset_hash,
            "parameter_hash": self.parameter_hash,
            "code_commit": self.code_commit,
            "random_seed": self.random_seed,
            "execution_model": self.execution_model,
            "latency_model": self.latency_model,
            "status": self.status.value,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "result_hash": self.result_hash,
            "validation_gates": self.validation_gates,
            "has_signature": bool(self.manifest_signature),
        }


# ---------------------------------------------------------------------------
# Hashing Utilities
# ---------------------------------------------------------------------------

def hash_dataset(data: Any) -> str:
    """Produce a deterministic SHA-256 hash of a dataset."""
    serialised = json.dumps(data, sort_keys=True, default=str).encode()
    return hashlib.sha256(serialised).hexdigest()


def hash_parameters(params: dict) -> str:
    """Produce a deterministic SHA-256 hash of strategy parameters."""
    serialised = json.dumps(params, sort_keys=True, default=str).encode()
    return hashlib.sha256(serialised).hexdigest()


def hash_results(results: dict) -> str:
    """Produce a deterministic SHA-256 hash of experiment results."""
    serialised = json.dumps(results, sort_keys=True, default=str).encode()
    return hashlib.sha256(serialised).hexdigest()


def get_code_commit() -> str:
    """Get the current git HEAD commit hash, or 'unknown' if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Experiment Manifest
# ---------------------------------------------------------------------------

def build_manifest(experiment: Experiment) -> dict:
    """Build a canonical manifest dict for signing.

    The manifest captures the full provenance of an experiment:
    dataset, strategy, parameters, code version, execution model, and results.
    """
    return {
        "experiment_id": experiment.experiment_id,
        "dataset_hash": experiment.dataset_hash,
        "strategy_id": experiment.strategy_id,
        "strategy_version": experiment.strategy_version,
        "parameter_hash": experiment.parameter_hash,
        "code_commit": experiment.code_commit,
        "random_seed": experiment.random_seed,
        "execution_model": experiment.execution_model,
        "latency_model": experiment.latency_model,
        "result_hash": experiment.result_hash,
        "created_at": experiment.created_at,
        "completed_at": experiment.completed_at,
    }


def _fallback_hmac_key() -> bytes:
    """Derive the dev/test fallback signing key from the deployment's own
    CSRF_SECRET rather than a fixed literal — a hardcoded key would let
    anyone who has read the (public) source code forge a manifest signature,
    which defeats the entire point of a "provable link between this P&L and
    this exact dataset/strategy/code version". CSRF_SECRET is random per
    deployment (and required to be explicitly set in production, see
    backend/config.py), so this key can't be derived without server-side
    secret material even though the derivation formula itself is public.
    """
    from backend.config import CSRF_SECRET
    return hashlib.sha256(f"qs-experiment-manifest-hmac:{CSRF_SECRET}".encode()).digest()


def sign_manifest(manifest: dict) -> str:
    """Sign an experiment manifest with the server's ML-DSA-65 key.

    Falls back to HMAC-SHA256 if ML-DSA is unavailable (e.g. in tests).
    The returned string is always prefixed with the signing method so that
    ``verify_manifest_signature`` can select the correct verification path.
    """
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    try:
        from backend.services import security_service
        identity = security_service.server_identity
        sig = identity.sign(canonical.encode())
        sig_hex = sig.hex() if isinstance(sig, bytes) else str(sig)
        return f"ml-dsa:{sig_hex}"
    except Exception:
        pass
    # Fallback: HMAC-SHA256 keyed off the deployment's own secret (see
    # _fallback_hmac_key) for dev/test, or if ML-DSA signing ever fails.
    import hmac
    sig = hmac.new(
        _fallback_hmac_key(),
        canonical.encode(),
        hashlib.sha256,
    ).hexdigest()
    return f"hmac-sha256:{sig}"


def verify_manifest_signature(manifest: dict, signature: str) -> bool:
    """Verify an experiment manifest signature."""
    if not signature:
        return False

    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))

    if signature.startswith("hmac-sha256:"):
        import hmac
        expected = hmac.new(
            _fallback_hmac_key(),
            canonical.encode(),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(signature, f"hmac-sha256:{expected}")

    if signature.startswith("ml-dsa:"):
        try:
            from backend.crypto.pqc import dsa_verify
            from backend.services import security_service
            identity = security_service.server_identity
            sig_hex = signature[len("ml-dsa:"):]
            sig_bytes = bytes.fromhex(sig_hex)
            return dsa_verify(identity.dsa_pk, canonical.encode(), sig_bytes)
        except Exception:
            return False

    return False



# ---------------------------------------------------------------------------
# Validation Gates (Strategy Deployment Protection)
# ---------------------------------------------------------------------------

def validate_for_deployment(experiment: Experiment) -> dict:
    """Check whether an experiment passes all deployment gates.

    A strategy is eligible for paper execution only if:
    - Walk-forward OOS Sharpe > threshold
    - DSR passes (survives deflation)
    - Turnover within limits
    - Risk limits defined
    - Experiment manifest is signed

    Returns a dict of gate results.
    """
    results = experiment.results
    gates = {}

    # Gate 1: Walk-forward OOS Sharpe
    oos_sharpe = results.get("oos_sharpe", 0)
    gates["walk_forward"] = {
        "passed": oos_sharpe > 0.5,
        "value": oos_sharpe,
        "threshold": 0.5,
        "description": "OOS Sharpe ratio must exceed 0.5",
    }

    # Gate 2: Deflated Sharpe Ratio
    dsr_passes = results.get("survives_deflation", False)
    gates["dsr"] = {
        "passed": bool(dsr_passes),
        "value": dsr_passes,
        "description": "Strategy must survive Deflated Sharpe Ratio test",
    }

    # Gate 3: Turnover
    turnover = results.get("avg_daily_turnover", 1.0)
    gates["turnover"] = {
        "passed": turnover < 2.0,
        "value": turnover,
        "threshold": 2.0,
        "description": "Average daily turnover must be below 200%",
    }

    # Gate 4: Risk limits defined
    has_risk = bool(results.get("max_position_size") or results.get("max_drawdown_limit"))
    gates["risk_limits"] = {
        "passed": has_risk,
        "description": "Strategy must define position size and drawdown limits",
    }

    # Gate 5: Manifest signed
    gates["manifest_signed"] = {
        "passed": bool(experiment.manifest_signature),
        "description": "Experiment manifest must be cryptographically signed",
    }

    all_passed = all(g["passed"] for g in gates.values())

    return {
        "experiment_id": experiment.experiment_id,
        "all_gates_passed": all_passed,
        "eligible_for_paper_deployment": all_passed,
        "gates": gates,
    }


# ---------------------------------------------------------------------------
# Experiment Registry
# ---------------------------------------------------------------------------

class ExperimentRegistry:
    """In-memory experiment registry for research governance.

    In a production deployment this would be backed by PostgreSQL.
    For the paper-trading research platform, in-memory storage
    with signed manifests provides full provenance tracking.
    """

    def __init__(self):
        self._experiments: dict[str, Experiment] = {}

    def create(
        self,
        strategy_id: str,
        strategy_version: str,
        dataset_id: str,
        dataset: Any = None,
        parameters: dict | None = None,
        random_seed: int = 42,
        execution_model: str = "LOB_QUEUE_V2",
        latency_model: str = "zero",
    ) -> Experiment:
        """Create a new experiment with full provenance."""
        exp = Experiment(
            strategy_id=strategy_id,
            strategy_version=strategy_version,
            dataset_id=dataset_id,
            dataset_hash=hash_dataset(dataset) if dataset is not None else "",
            parameter_hash=hash_parameters(parameters or {}),
            code_commit=get_code_commit(),
            random_seed=random_seed,
            execution_model=execution_model,
            latency_model=latency_model,
        )
        self._experiments[exp.experiment_id] = exp
        return exp

    def start(self, experiment_id: str) -> Experiment | None:
        """Mark an experiment as running."""
        exp = self._experiments.get(experiment_id)
        if exp:
            exp.status = ExperimentStatus.RUNNING
        return exp

    def complete(
        self,
        experiment_id: str,
        results: dict,
        sign: bool = True,
    ) -> Experiment | None:
        """Complete an experiment with results and optional signing."""
        exp = self._experiments.get(experiment_id)
        if not exp:
            return None

        exp.results = results
        exp.result_hash = hash_results(results)
        exp.completed_at = time.time()
        exp.status = ExperimentStatus.COMPLETED

        if sign:
            manifest = build_manifest(exp)
            exp.manifest_signature = sign_manifest(manifest)

        return exp

    def validate(self, experiment_id: str) -> dict | None:
        """Run deployment validation gates on an experiment."""
        exp = self._experiments.get(experiment_id)
        if not exp:
            return None

        result = validate_for_deployment(exp)
        exp.validation_gates = result
        if result["all_gates_passed"]:
            exp.status = ExperimentStatus.VALIDATED
        return result

    def approve(self, experiment_id: str) -> Experiment | None:
        """Approve a validated experiment for paper deployment."""
        exp = self._experiments.get(experiment_id)
        if exp and exp.status == ExperimentStatus.VALIDATED:
            exp.status = ExperimentStatus.APPROVED
        return exp

    def get(self, experiment_id: str) -> Experiment | None:
        return self._experiments.get(experiment_id)

    def list_all(self) -> list[dict]:
        return [exp.to_dict() for exp in self._experiments.values()]

    def get_manifest(self, experiment_id: str) -> dict | None:
        exp = self._experiments.get(experiment_id)
        if not exp:
            return None
        manifest = build_manifest(exp)
        manifest["signature"] = exp.manifest_signature
        manifest["signature_valid"] = verify_manifest_signature(
            {k: v for k, v in manifest.items() if k not in ("signature", "signature_valid")},
            exp.manifest_signature,
        ) if exp.manifest_signature else False
        return manifest


# ---------------------------------------------------------------------------
# Durable experiment registry
# ---------------------------------------------------------------------------

class PersistentExperimentRegistry:
    """Database-backed experiment registry scoped to one authenticated user.

    The in-memory registry above remains useful for isolated unit experiments;
    API requests must use this registry so a create request is retrievable by
    a later request and survives an application restart.
    """

    def __init__(self, db, user_id: str):
        self.db = db
        self.user_id = user_id

    @staticmethod
    def _epoch(value: dt.datetime | None) -> float | None:
        return value.timestamp() if value is not None else None

    @classmethod
    def _to_experiment(cls, row) -> Experiment:
        return Experiment(
            experiment_id=row.id,
            strategy_id=row.strategy_id,
            strategy_version=row.strategy_version,
            dataset_id=row.dataset_id,
            dataset_hash=row.dataset_hash,
            parameter_hash=row.parameter_hash,
            code_commit=row.code_commit,
            random_seed=row.random_seed,
            execution_model=row.execution_model,
            latency_model=row.latency_model,
            status=ExperimentStatus(row.status),
            created_at=cls._epoch(row.created_at) or 0.0,
            completed_at=cls._epoch(row.completed_at),
            result_hash=row.result_hash,
            results=row.results_json or {},
            validation_gates=row.validation_gates_json or {},
            manifest_signature=row.manifest_signature or "",
        )

    def _row(self, experiment_id: str):
        from backend import models
        return self.db.get(models.ResearchExperiment, experiment_id)

    def create(self, strategy_id: str, strategy_version: str, dataset_id: str,
               dataset: Any = None, parameters: dict | None = None,
               random_seed: int = 42, execution_model: str = "LOB_QUEUE_V2",
               latency_model: str = "zero") -> Experiment:
        from backend import models
        exp = Experiment(
            strategy_id=strategy_id,
            strategy_version=strategy_version,
            dataset_id=dataset_id,
            dataset_hash=hash_dataset(dataset) if dataset is not None else "",
            parameter_hash=hash_parameters(parameters or {}),
            code_commit=get_code_commit(),
            random_seed=random_seed,
            execution_model=execution_model,
            latency_model=latency_model,
        )
        self.db.add(models.ResearchExperiment(
            id=exp.experiment_id, user_id=self.user_id,
            strategy_id=exp.strategy_id, strategy_version=exp.strategy_version,
            dataset_id=exp.dataset_id, dataset_json=dataset,
            dataset_hash=exp.dataset_hash, parameters_json=parameters or {},
            parameter_hash=exp.parameter_hash, code_commit=exp.code_commit,
            random_seed=exp.random_seed, execution_model=exp.execution_model,
            latency_model=exp.latency_model, status=exp.status.value,
        ))
        self.db.commit()
        return self.get(exp.experiment_id)  # type: ignore[return-value]

    def get(self, experiment_id: str) -> Experiment | None:
        row = self._row(experiment_id)
        if not row or row.user_id != self.user_id:
            return None
        return self._to_experiment(row)

    def inputs(self, experiment_id: str) -> tuple[Any, dict, int] | None:
        """Return the immutable stored inputs for deterministic verification."""
        row = self._row(experiment_id)
        if not row or row.user_id != self.user_id:
            return None
        return row.dataset_json, row.parameters_json or {}, row.random_seed

    def complete(self, experiment_id: str, results: dict, sign: bool = True) -> Experiment | None:
        row = self._row(experiment_id)
        if not row or row.user_id != self.user_id:
            return None
        exp = self._to_experiment(row)
        exp.results = results
        exp.result_hash = hash_results(results)
        exp.completed_at = time.time()
        exp.status = ExperimentStatus.COMPLETED
        if sign:
            exp.manifest_signature = sign_manifest(build_manifest(exp))
        row.results_json = results
        row.result_hash = exp.result_hash
        row.completed_at = dt.datetime.fromtimestamp(exp.completed_at, tz=dt.timezone.utc)
        row.status = exp.status.value
        row.manifest_signature = exp.manifest_signature
        self.db.commit()
        return self._to_experiment(row)

    def get_manifest(self, experiment_id: str) -> dict | None:
        exp = self.get(experiment_id)
        if not exp:
            return None
        manifest = build_manifest(exp)
        manifest["signature"] = exp.manifest_signature
        manifest["signature_valid"] = bool(exp.manifest_signature) and verify_manifest_signature(
            build_manifest(exp), exp.manifest_signature,
        )
        return manifest
