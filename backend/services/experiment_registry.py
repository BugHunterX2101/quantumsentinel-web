"""QuantumSentinel — Experiment Registry & Signed Research Manifests.

Versioned experiment tracking with deterministic replay and ML-DSA-65
signed manifests.

    Dataset → Experiment → Strategy → Parameters → Paper Orders
    → Fills → Portfolio → Metrics → Signed Manifest

A completed experiment's manifest (v2) binds the result to:

* ``dataset_hash`` / ``parameter_hash`` / ``random_seed`` — exact inputs
* ``strategy_hash`` — SHA-256 of the strategy's *source code* for
  platform-executable strategies (otherwise of its id and version)
* ``code_commit`` / ``dependency_lock_hash`` / ``execution_engine_version``
  — the software that produced it
* ``execution_model`` / ``latency_model`` — the simulator configuration
* ``result_hash`` — the outputs
* ``signing_key_id`` — which server key signed it, so manifests stay
  verifiable after key rotation

Platform-executable strategies (``EXECUTABLE_STRATEGIES``) can be run and
*re-executed* from their stored inputs; replay compares result hashes.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import inspect
import json
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# Bump whenever the simulator's semantics change, so old results are not
# silently compared against a different engine.
ENGINE_VERSION = "QS-L2-SIM-2"
MANIFEST_VERSION = 2
_REPO_ROOT = Path(__file__).resolve().parents[2]


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


class ExperimentError(ValueError):
    """An experiment operation is not permitted in the experiment's state."""


class ManifestSigningError(RuntimeError):
    """ML-DSA signing failed where a downgrade to HMAC is not acceptable."""


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
    strategy_hash: str = ""
    dependency_lock_hash: str = ""
    engine_version: str = ENGINE_VERSION
    signing_key_id: str | None = None

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
            "strategy_hash": self.strategy_hash,
            "executable": self.strategy_id in EXECUTABLE_STRATEGY_IDS,
            "dataset_id": self.dataset_id,
            "dataset_hash": self.dataset_hash,
            "parameter_hash": self.parameter_hash,
            "code_commit": self.code_commit,
            "dependency_lock_hash": self.dependency_lock_hash,
            "execution_engine_version": self.engine_version,
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

def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()


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
            capture_output=True, text=True, timeout=5, cwd=_REPO_ROOT,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def dependency_lock_hash() -> str:
    """SHA-256 of requirements.txt with line endings normalised (platform-independent)."""
    try:
        data = (_REPO_ROOT / "requirements.txt").read_bytes()
    except OSError:
        return "unknown"
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


# ---------------------------------------------------------------------------
# Platform-executable strategies
# ---------------------------------------------------------------------------

EXECUTABLE_STRATEGY_IDS = frozenset({"obi_momentum"})

_OBI_DEFAULTS = {
    "obi_threshold": 0.3, "events_per_bar": 50, "snapshot_interval": 10, "warmup_events": 50,
    "order_quantity": 100.0, "order_style": "aggressive", "cancel_after_events": None,
    "latency_preset": "zero", "tick_size": 0.01,
}


def _strategy_function(strategy_id: str):
    if strategy_id == "obi_momentum":
        from backend.services.l2_event_replay import obi_momentum_strategy
        return obi_momentum_strategy
    return None


def strategy_code_hash(strategy_id: str, strategy_version: str) -> str:
    """SHA-256 of the strategy's source code when the platform can execute it,
    otherwise of its declared identity (id + version)."""
    fn = _strategy_function(strategy_id)
    if fn is not None:
        return hashlib.sha256(b"source:" + inspect.getsource(fn).encode()).hexdigest()
    return hashlib.sha256(_canonical({"strategy_id": strategy_id, "strategy_version": strategy_version})).hexdigest()


def _obi_parameters(parameters: dict) -> dict:
    from backend.services.latency_model import LATENCY_PRESETS
    unknown = set(parameters) - set(_OBI_DEFAULTS)
    if unknown:
        raise ExperimentError(f"unknown obi_momentum parameters: {sorted(unknown)}")
    p = {**_OBI_DEFAULTS, **parameters}
    if not 0 < float(p["obi_threshold"]) < 1:
        raise ExperimentError("obi_threshold must be in (0, 1)")
    if not 2 <= int(p["events_per_bar"]) <= 500:
        raise ExperimentError("events_per_bar must be between 2 and 500")
    if float(p["order_quantity"]) <= 0 or float(p["tick_size"]) <= 0:
        raise ExperimentError("order_quantity and tick_size must be positive")
    if p["order_style"] not in ("aggressive", "passive"):
        raise ExperimentError("order_style must be aggressive or passive")
    if p["latency_preset"] not in LATENCY_PRESETS:
        raise ExperimentError(f"latency_preset must be one of {sorted(LATENCY_PRESETS)}")
    return p


def run_strategy(strategy_id: str, dataset: Any, parameters: dict, seed: int) -> dict:
    """Execute a platform strategy deterministically from its inputs.

    ``obi_momentum``: ``dataset`` is a list of OHLCV bars; synthetic L2 is
    generated from them with ``seed`` and replayed through the paper
    exchange (latency preset, queue-aware matching). Identical inputs give
    byte-identical results.
    """
    if strategy_id not in EXECUTABLE_STRATEGY_IDS:
        raise ExperimentError(f"strategy {strategy_id!r} is not executable by the platform")
    from backend.services.l2_event_replay import L2EventStream, ReplayConfig, replay_session, obi_momentum_strategy
    from backend.services.latency_model import LATENCY_PRESETS
    from backend.services.paper_exchange import PaperExchange

    p = _obi_parameters(parameters or {})
    if not isinstance(dataset, list) or not dataset or len(dataset) > 500:
        raise ExperimentError("obi_momentum needs a dataset of 1-500 OHLCV bars")
    required = {"timestamp", "open", "high", "low", "close"}
    if any(not isinstance(b, dict) or not required <= set(b) for b in dataset):
        raise ExperimentError(f"every bar needs {sorted(required)}")
    if len(dataset) * int(p["events_per_bar"]) > 100_000:
        raise ExperimentError("bars x events_per_bar must not exceed 100,000 events")

    stream = L2EventStream.from_synthetic(dataset, seed=int(seed), events_per_bar=int(p["events_per_bar"]),
                                          tick_size=float(p["tick_size"]))
    exchange = PaperExchange(symbol="SYNTHETIC", tick_size=float(p["tick_size"]),
                             latency_ms=LATENCY_PRESETS[p["latency_preset"]].total_latency_ms)
    config = ReplayConfig(snapshot_interval=int(p["snapshot_interval"]), warmup_events=int(p["warmup_events"]),
                          tick_size=float(p["tick_size"]), order_quantity=float(p["order_quantity"]),
                          order_style=p["order_style"], cancel_after_events=p["cancel_after_events"])
    threshold = float(p["obi_threshold"])
    result = replay_session(stream, lambda s, t: obi_momentum_strategy(s, t, obi_threshold=threshold),
                            config, exchange=exchange)
    execution = result.execution
    analytics = execution["analytics"]
    return {
        "l2_dataset_hash": stream.dataset_hash,
        "data_source": stream.source,
        "total_events": result.total_events,
        "strategy_signals": result.strategy_signals,
        "portfolio": execution["portfolio"],
        "execution_metrics": analytics["execution_metrics"],
        "implementation_shortfall": {
            "total_is": analytics["implementation_shortfall"]["total_is"],
            "total_is_bps": analytics["implementation_shortfall"]["total_is_bps"],
        },
        "fills_hash": hashlib.sha256(_canonical(execution["fills"])).hexdigest(),
        "orders_hash": hashlib.sha256(_canonical(execution["orders"])).hexdigest(),
        "max_position_size": float(p["order_quantity"]),
        "latency_ms": execution["exchange_stats"]["latency_ms"],
    }


# ---------------------------------------------------------------------------
# Experiment Manifest
# ---------------------------------------------------------------------------

_MANIFEST_BOUND_FIELDS = ("experiment_id", "strategy_id", "strategy_version", "strategy_hash",
                          "dataset_id", "dataset_hash", "parameter_hash", "code_commit",
                          "dependency_lock_hash", "random_seed", "execution_model", "latency_model",
                          "result_hash")


def build_manifest(experiment: Experiment) -> dict:
    """Canonical manifest dict for signing (v2)."""
    return {
        "manifest_version": MANIFEST_VERSION,
        "experiment_id": experiment.experiment_id,
        "strategy_id": experiment.strategy_id,
        "strategy_version": experiment.strategy_version,
        "strategy_hash": experiment.strategy_hash,
        "dataset_id": experiment.dataset_id,
        "dataset_hash": experiment.dataset_hash,
        "parameter_hash": experiment.parameter_hash,
        "code_commit": experiment.code_commit,
        "dependency_lock_hash": experiment.dependency_lock_hash,
        "execution_engine_version": experiment.engine_version,
        "random_seed": experiment.random_seed,
        "execution_model": experiment.execution_model,
        "latency_model": experiment.latency_model,
        "result_hash": experiment.result_hash,
        "created_at": experiment.created_at,
        "completed_at": experiment.completed_at,
        "signing_key_id": experiment.signing_key_id,
    }


def _fallback_hmac_key() -> bytes:
    """Development/test-only HMAC key, derived from the deployment secret."""
    from backend.config import CSRF_SECRET
    return hashlib.sha256(f"qs-experiment-manifest-hmac:{CSRF_SECRET}".encode()).digest()


def _environment() -> str:
    from backend.config import ENVIRONMENT
    return ENVIRONMENT


def sign_manifest(manifest: dict) -> str:
    """Sign a manifest with the server's ML-DSA-65 key.

    Outside production a failed ML-DSA signature falls back to an HMAC
    (clearly prefixed); in production that downgrade is refused.
    """
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    try:
        from backend.services import security_service
        sig = security_service.server_identity.sign(canonical.encode())
        return f"ml-dsa:{sig.hex() if isinstance(sig, bytes) else sig}"
    except Exception as exc:
        if _environment() == "production":
            raise ManifestSigningError("ML-DSA manifest signing failed") from exc
    import hmac
    sig = hmac.new(_fallback_hmac_key(), canonical.encode(), hashlib.sha256).hexdigest()
    return f"hmac-sha256:{sig}"


def _manifest_public_key(key_id: str | None, db=None) -> bytes | None:
    """Public key that signed a manifest: historical key by id, else current."""
    from backend.services import security_service
    identity = security_service.server_identity
    if key_id and db is not None:
        from backend import models
        from backend.crypto import pqc
        record = db.query(models.ServerSigningKey).filter(models.ServerSigningKey.key_id == key_id).first()
        if record is not None:
            return pqc.unb64(record.public_key)
    if not key_id or key_id == identity.key_id:
        return identity.dsa_pk
    return None


def verify_manifest_signature(manifest: dict, signature: str, db=None) -> bool:
    """Verify a manifest signature, using the key named by ``signing_key_id``."""
    if not signature:
        return False
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))

    if signature.startswith("hmac-sha256:"):
        if _environment() == "production":
            return False
        import hmac
        expected = hmac.new(_fallback_hmac_key(), canonical.encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(signature, f"hmac-sha256:{expected}")

    if signature.startswith("ml-dsa:"):
        try:
            from backend.crypto.pqc import dsa_verify
            public_key = _manifest_public_key(manifest.get("signing_key_id"), db)
            if public_key is None:
                return False
            return dsa_verify(public_key, canonical.encode(), bytes.fromhex(signature[len("ml-dsa:"):]))
        except Exception:
            return False

    return False


# ---------------------------------------------------------------------------
# Validation Gates (Strategy Deployment Protection)
# ---------------------------------------------------------------------------

def validate_for_deployment(experiment: Experiment) -> dict:
    """Deployment gates appropriate to the kind of result.

    Signal research (walk-forward results, ``oos_sharpe`` present):
    OOS Sharpe, deflated Sharpe, turnover, risk limits, signed manifest.

    Execution experiments (``execution_metrics`` present, e.g. a replay of
    a platform strategy): positive net P&L, fill ratio, implementation
    shortfall, risk limits, signed manifest.
    """
    results = experiment.results
    gates: dict[str, dict] = {}

    if "execution_metrics" in results:
        pnl = results.get("portfolio", {}).get("total_pnl", 0.0)
        fill_ratio = results.get("execution_metrics", {}).get("fill_ratio", 0.0)
        is_bps = results.get("implementation_shortfall", {}).get("total_is_bps", float("inf"))
        gates["net_pnl"] = {"passed": pnl > 0, "value": pnl, "threshold": 0.0,
                            "description": "Net P&L after execution costs must be positive"}
        gates["fill_ratio"] = {"passed": fill_ratio >= 0.5, "value": fill_ratio, "threshold": 0.5,
                               "description": "At least half of submitted quantity must execute"}
        gates["implementation_shortfall"] = {"passed": is_bps <= 25.0, "value": is_bps, "threshold": 25.0,
                                             "description": "Implementation shortfall must not exceed 25 bps"}
    else:
        oos_sharpe = results.get("oos_sharpe", 0)
        gates["walk_forward"] = {"passed": oos_sharpe > 0.5, "value": oos_sharpe, "threshold": 0.5,
                                 "description": "OOS Sharpe ratio must exceed 0.5"}
        dsr_passes = results.get("survives_deflation", False)
        gates["dsr"] = {"passed": bool(dsr_passes), "value": dsr_passes,
                        "description": "Strategy must survive Deflated Sharpe Ratio test"}
        turnover = results.get("avg_daily_turnover", 1.0)
        gates["turnover"] = {"passed": turnover < 2.0, "value": turnover, "threshold": 2.0,
                             "description": "Average daily turnover must be below 200%"}

    has_risk = bool(results.get("max_position_size") or results.get("max_drawdown_limit"))
    gates["risk_limits"] = {"passed": has_risk,
                            "description": "Strategy must define position size and drawdown limits"}
    gates["manifest_signed"] = {"passed": bool(experiment.manifest_signature),
                                "description": "Experiment manifest must be cryptographically signed"}

    all_passed = all(g["passed"] for g in gates.values())
    return {
        "experiment_id": experiment.experiment_id,
        "all_gates_passed": all_passed,
        "eligible_for_paper_deployment": all_passed,
        "gates": gates,
    }


# ---------------------------------------------------------------------------
# In-memory registry (unit experiments)
# ---------------------------------------------------------------------------

class ExperimentRegistry:
    """In-memory experiment registry for isolated research experiments.

    API requests use :class:`PersistentExperimentRegistry`.
    """

    def __init__(self):
        self._experiments: dict[str, Experiment] = {}

    def create(self, strategy_id: str, strategy_version: str, dataset_id: str, dataset: Any = None,
               parameters: dict | None = None, random_seed: int = 42,
               execution_model: str = "LOB_QUEUE_V2", latency_model: str = "zero") -> Experiment:
        exp = Experiment(
            strategy_id=strategy_id, strategy_version=strategy_version, dataset_id=dataset_id,
            dataset_hash=hash_dataset(dataset) if dataset is not None else "",
            parameter_hash=hash_parameters(parameters or {}),
            code_commit=get_code_commit(), random_seed=random_seed,
            execution_model=execution_model, latency_model=latency_model,
            strategy_hash=strategy_code_hash(strategy_id, strategy_version),
            dependency_lock_hash=dependency_lock_hash(),
        )
        self._experiments[exp.experiment_id] = exp
        return exp

    def start(self, experiment_id: str) -> Experiment | None:
        exp = self._experiments.get(experiment_id)
        if exp:
            exp.status = ExperimentStatus.RUNNING
        return exp

    def complete(self, experiment_id: str, results: dict, sign: bool = True) -> Experiment | None:
        exp = self._experiments.get(experiment_id)
        if not exp:
            return None
        exp.results = results
        exp.result_hash = hash_results(results)
        exp.completed_at = time.time()
        exp.status = ExperimentStatus.COMPLETED
        if sign:
            from backend.services import security_service
            exp.signing_key_id = security_service.server_identity.key_id
            exp.manifest_signature = sign_manifest(build_manifest(exp))
        return exp

    def validate(self, experiment_id: str) -> dict | None:
        exp = self._experiments.get(experiment_id)
        if not exp:
            return None
        result = validate_for_deployment(exp)
        exp.validation_gates = result
        if result["all_gates_passed"]:
            exp.status = ExperimentStatus.VALIDATED
        return result

    def approve(self, experiment_id: str) -> Experiment | None:
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
        signature_valid = verify_manifest_signature(manifest, exp.manifest_signature) \
            if exp.manifest_signature else False
        return {**manifest, "signature": exp.manifest_signature, "signature_valid": signature_valid}


# ---------------------------------------------------------------------------
# Durable experiment registry
# ---------------------------------------------------------------------------

class PersistentExperimentRegistry:
    """Database-backed experiment registry scoped to one authenticated user.

    Inputs are immutable once created; results can be recorded once; the
    exact signed manifest is stored and verified verbatim.
    """

    def __init__(self, db, user_id: str):
        self.db = db
        self.user_id = user_id

    @staticmethod
    def _epoch(value: dt.datetime | None) -> float | None:
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=dt.timezone.utc)
        return value.timestamp()

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
            strategy_hash=row.strategy_hash or "",
            dependency_lock_hash=row.dependency_lock_hash or "",
            engine_version=row.engine_version or "",
            signing_key_id=row.signing_key_id,
            result_hash=row.result_hash,
            results=row.results_json or {},
            validation_gates=row.validation_gates_json or {},
            manifest_signature=row.manifest_signature or "",
        )

    def _row(self, experiment_id: str):
        from backend import models
        row = self.db.get(models.ResearchExperiment, experiment_id)
        return row if row is not None and row.user_id == self.user_id else None

    def create(self, strategy_id: str, strategy_version: str, dataset_id: str,
               dataset: Any = None, parameters: dict | None = None,
               random_seed: int = 42, execution_model: str = "LOB_QUEUE_V2",
               latency_model: str = "zero") -> Experiment:
        from backend import models
        exp = Experiment(
            strategy_id=strategy_id, strategy_version=strategy_version, dataset_id=dataset_id,
            dataset_hash=hash_dataset(dataset) if dataset is not None else "",
            parameter_hash=hash_parameters(parameters or {}),
            code_commit=get_code_commit(), random_seed=random_seed,
            execution_model=execution_model, latency_model=latency_model,
            strategy_hash=strategy_code_hash(strategy_id, strategy_version),
            dependency_lock_hash=dependency_lock_hash(),
        )
        self.db.add(models.ResearchExperiment(
            id=exp.experiment_id, user_id=self.user_id,
            strategy_id=exp.strategy_id, strategy_version=exp.strategy_version,
            dataset_id=exp.dataset_id, dataset_json=dataset,
            dataset_hash=exp.dataset_hash, parameters_json=parameters or {},
            parameter_hash=exp.parameter_hash, code_commit=exp.code_commit,
            random_seed=exp.random_seed, execution_model=exp.execution_model,
            latency_model=exp.latency_model, status=exp.status.value,
            strategy_hash=exp.strategy_hash, dependency_lock_hash=exp.dependency_lock_hash,
            engine_version=exp.engine_version,
        ))
        self.db.commit()
        return self.get(exp.experiment_id)  # type: ignore[return-value]

    def get(self, experiment_id: str) -> Experiment | None:
        row = self._row(experiment_id)
        return self._to_experiment(row) if row else None

    def inputs(self, experiment_id: str) -> tuple[Any, dict, int] | None:
        """Return the immutable stored inputs for deterministic verification."""
        row = self._row(experiment_id)
        if not row:
            return None
        return row.dataset_json, row.parameters_json or {}, row.random_seed

    def complete(self, experiment_id: str, results: dict, sign: bool = True) -> Experiment | None:
        """Record results once and sign the manifest; results are then immutable."""
        row = self._row(experiment_id)
        if not row:
            return None
        if row.status != ExperimentStatus.CREATED.value:
            raise ExperimentError(f"results already recorded (status {row.status})")
        exp = self._to_experiment(row)
        exp.results = results
        exp.result_hash = hash_results(results)
        exp.completed_at = time.time()
        exp.status = ExperimentStatus.COMPLETED
        manifest = None
        if sign:
            from backend.services import security_service
            # Register in *this* database (idempotent by fingerprint) so the
            # key named in the manifest is resolvable wherever the manifest is.
            identity = security_service.server_identity
            if identity.dsa_sk is None:
                identity.sign(b"")
            identity.register_in_db(self.db)
            exp.signing_key_id = identity.key_id
            manifest = build_manifest(exp)
            exp.manifest_signature = sign_manifest(manifest)
        row.results_json = results
        row.result_hash = exp.result_hash
        row.completed_at = dt.datetime.fromtimestamp(exp.completed_at, tz=dt.timezone.utc)
        row.status = exp.status.value
        row.signing_key_id = exp.signing_key_id
        row.manifest_json = manifest
        row.manifest_signature = exp.manifest_signature
        self.db.commit()
        return self._to_experiment(row)

    def run(self, experiment_id: str) -> Experiment | None:
        """Execute a platform strategy on the stored inputs and record the result."""
        row = self._row(experiment_id)
        if not row:
            return None
        if row.strategy_id not in EXECUTABLE_STRATEGY_IDS:
            raise ExperimentError(f"strategy {row.strategy_id!r} is not executable by the platform")
        if row.status != ExperimentStatus.CREATED.value:
            raise ExperimentError(f"experiment already has results (status {row.status})")
        results = run_strategy(row.strategy_id, row.dataset_json, row.parameters_json or {}, row.random_seed)
        return self.complete(experiment_id, results)

    def reexecute(self, experiment_id: str) -> dict | None:
        """Re-run a completed executable experiment and compare result hashes."""
        row = self._row(experiment_id)
        if not row:
            return None
        if row.strategy_id not in EXECUTABLE_STRATEGY_IDS or not row.result_hash:
            return {"executed": False, "result_reproduced": None}
        results = run_strategy(row.strategy_id, row.dataset_json, row.parameters_json or {}, row.random_seed)
        replay_hash = hash_results(results)
        return {"executed": True, "result_hash": replay_hash,
                "result_reproduced": replay_hash == row.result_hash,
                "engine_version_matches": (row.engine_version or "") == ENGINE_VERSION}

    def get_manifest(self, experiment_id: str) -> dict | None:
        """The stored signed manifest, its signature validity and row consistency."""
        row = self._row(experiment_id)
        if not row:
            return None
        manifest = row.manifest_json
        if not manifest:
            return {"experiment_id": row.id, "signature": row.manifest_signature or "",
                    "signature_valid": False, "consistent_with_record": False}
        exp = self._to_experiment(row)
        consistent = all(manifest.get(k) == getattr(exp, k) for k in _MANIFEST_BOUND_FIELDS)
        return {**manifest, "signature": row.manifest_signature,
                "signature_valid": verify_manifest_signature(manifest, row.manifest_signature, self.db),
                "consistent_with_record": consistent}

    def validate(self, experiment_id: str) -> dict | None:
        """Integrity gates plus deployment gates; VALIDATED only if all pass."""
        row = self._row(experiment_id)
        if not row:
            return None
        if row.status not in (ExperimentStatus.COMPLETED.value, ExperimentStatus.VALIDATED.value):
            raise ExperimentError(f"only completed experiments can be validated (status {row.status})")
        exp = self._to_experiment(row)
        report = validate_for_deployment(exp)
        manifest = self.get_manifest(experiment_id)
        gates = report["gates"]
        gates["signature_valid"] = {"passed": bool(manifest["signature_valid"]),
                                    "description": "Manifest signature verifies with the key that signed it"}
        gates["manifest_consistent"] = {"passed": bool(manifest["consistent_with_record"]),
                                        "description": "Signed manifest matches the stored experiment"}
        inputs_ok = ((hash_dataset(row.dataset_json) if row.dataset_json is not None else "") == row.dataset_hash
                     and hash_parameters(row.parameters_json or {}) == row.parameter_hash)
        gates["inputs_unchanged"] = {"passed": inputs_ok,
                                     "description": "Stored dataset and parameters still hash to the recorded values"}
        replay = self.reexecute(experiment_id)
        gates["result_reproducible"] = {
            "passed": bool(replay.get("result_reproduced")),
            "value": replay.get("result_hash"),
            "description": "Re-executing the strategy from stored inputs reproduces the result hash"
                           if replay["executed"] else
                           "Strategy is not executable by the platform, so its result cannot be reproduced",
        }
        report["all_gates_passed"] = all(g["passed"] for g in gates.values())
        report["eligible_for_paper_deployment"] = report["all_gates_passed"]
        row.validation_gates_json = report
        row.status = (ExperimentStatus.VALIDATED if report["all_gates_passed"]
                      else ExperimentStatus.COMPLETED).value
        self.db.commit()
        return report


def approve_experiment(db, experiment_id: str, approver_id: str) -> Experiment:
    """Approve a VALIDATED experiment (operator action, four-eyes).

    Callers must check that the approver holds an operator role. The owner
    cannot approve their own experiment.
    """
    from backend import models
    row = db.get(models.ResearchExperiment, experiment_id)
    if row is None:
        raise LookupError(experiment_id)
    if row.user_id == approver_id:
        raise ExperimentError("an experiment cannot be approved by its owner (four-eyes rule)")
    if row.status != ExperimentStatus.VALIDATED.value:
        raise ExperimentError(f"only validated experiments can be approved (status {row.status})")
    row.status = ExperimentStatus.APPROVED.value
    row.approved_by = approver_id
    row.approved_at = dt.datetime.now(dt.timezone.utc)
    db.commit()
    return PersistentExperimentRegistry._to_experiment(row)
