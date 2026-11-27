"""QuantumSentinel — ORM models (mirrors the PRD PostgreSQL schema, simplified)."""
import uuid
import datetime as dt

from sqlalchemy import (
    Column, String, Boolean, Date, DateTime, Numeric, Integer, BigInteger, Float, ForeignKey, Text, JSON,
    LargeBinary, UniqueConstraint, Sequence, Index, ForeignKeyConstraint, text,
)
from sqlalchemy.orm import relationship

from .database import Base


def gen_uuid() -> str:
    return str(uuid.uuid4())


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, default=gen_uuid)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    tier = Column(String, default="free")  # free | pro | enterprise
    beginner_mode = Column(Boolean, default=True)
    is_active = Column(Boolean, default=True)
    watchlist = Column(JSON, nullable=True)  # list[str] of ticker symbols; None → use default
    preferred_exchanges = Column(JSON, nullable=True)  # list[str] e.g. ["US","NSE","CRYPTO"]
    user_timezone = Column(String(64), nullable=True)  # IANA tz e.g. "Asia/Kolkata"
    # Provisioned out of band (python -m backend.manage set-role), never by
    # registration: user | risk_admin | admin.
    role = Column(String(32), nullable=False, default="user", server_default="user")
    created_at = Column(DateTime(timezone=True), default=utcnow)

    key_pairs = relationship("KeyPair", back_populates="user")
    trades = relationship("Trade", back_populates="user")
    positions = relationship("Position", back_populates="user")


class KeyPair(Base):
    __tablename__ = "key_pairs"
    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    algorithm = Column(String(50), nullable=False)  # ML-KEM-768 | ML-DSA-65 | X25519
    public_key = Column(Text, nullable=False)  # base64
    private_key = Column(Text, nullable=True)  # base64 — demo only; a real client keeps this on-device
    is_active = Column(Boolean, default=True)
    rotation_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    revoked_at = Column(DateTime(timezone=True), nullable=True)

    user = relationship("User", back_populates="key_pairs")


class Strategy(Base):
    __tablename__ = "strategies"
    __table_args__ = (Index("ix_strategies_user_created", "user_id", "created_at"),)
    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    name = Column(String, nullable=False)
    assets = Column(JSON, default=list)
    config = Column(JSON, default=dict)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)


class Backtest(Base):
    __tablename__ = "backtests"
    id = Column(String, primary_key=True, default=gen_uuid)
    strategy_id = Column(String, ForeignKey("strategies.id"), nullable=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    start_date = Column(DateTime(timezone=True), nullable=True)
    end_date = Column(DateTime(timezone=True), nullable=True)
    initial_capital = Column(Numeric, default=100_000)
    final_capital = Column(Numeric, nullable=False)
    sharpe_ratio = Column(Numeric, default=0)
    max_drawdown = Column(Numeric, default=0)
    win_rate = Column(Numeric, default=0)
    total_trades = Column(Integer, default=0)
    status = Column(String, default="COMPLETED")
    result_json = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), default=utcnow)


class ApiKey(Base):
    __tablename__ = "api_keys"
    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String(80), nullable=False)
    key_prefix = Column(String(16), nullable=False)
    key_hash = Column(String(128), nullable=False, unique=True, index=True)
    hmac_secret_encrypted = Column(Text, nullable=True)  # Fernet-encrypted HMAC secret for signed requests
    scopes = Column(JSON, default=list)
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    is_revoked = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=utcnow)


class Webhook(Base):
    __tablename__ = "webhooks"
    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    url = Column(String(2048), nullable=False)
    # Fernet ciphertext of the signing secret (about 140 characters), not a hash.
    secret_hash = Column(Text, nullable=False)
    event_types = Column(JSON, default=list)
    is_active = Column(Boolean, default=True)
    last_delivery_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)


class Trade(Base):
    __tablename__ = "trades"
    # Every per-user read (orders, positions, exposure, holdings) filters on
    # user_id and usually status; the sweeper reads only resting orders, by asset.
    __table_args__ = (Index("ix_trades_user_status_filled", "user_id", "status", "filled_at"),
                      Index("ix_trades_accepted_asset", "asset", "submitted_at",
                            postgresql_where=text("status = 'ACCEPTED'")))
    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    asset = Column(String, nullable=False)
    side = Column(String, nullable=False)  # buy | sell
    quantity = Column(Numeric, nullable=False)
    order_type = Column(String, default="market")  # market | limit
    limit_price = Column(Numeric, nullable=True)
    stop_price = Column(Numeric, nullable=True)
    time_in_force = Column(String, default="day")
    status = Column(String, default="PENDING")
    alpaca_order_id = Column(String, nullable=True)
    filled_price = Column(Numeric, nullable=True)
    pqc_signature = Column(Text, nullable=True)  # ML-DSA-65 signature (base64)
    submitted_at = Column(DateTime(timezone=True), default=utcnow)
    filled_at = Column(DateTime(timezone=True), nullable=True)

    user = relationship("User", back_populates="trades")


class Position(Base):
    __tablename__ = "positions"
    __table_args__ = (Index("ix_positions_user_asset", "user_id", "asset"),)
    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    asset = Column(String, nullable=False)
    quantity = Column(Numeric, default=0)
    avg_entry_price = Column(Numeric, default=0)
    realized_pnl = Column(Numeric, default=0)
    updated_at = Column(DateTime(timezone=True), default=utcnow)

    user = relationship("User", back_populates="positions")


class PaperAccount(Base):
    """Server-side paper-trading cash ledger, one row per user.

    Money is stored as integer micro-dollars so that the atomic
    compare-and-swap reservations in paper_broker are exact on every backend
    (SQLite has no decimal type and would otherwise compare binary floats).
    """
    __tablename__ = "paper_accounts"
    user_id = Column(String, ForeignKey("users.id"), primary_key=True)
    cash_micros = Column(BigInteger, nullable=False)
    reserved_micros = Column(BigInteger, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class SignalRecord(Base):
    __tablename__ = "signals"
    id = Column(String, primary_key=True, default=gen_uuid)
    asset = Column(String, nullable=False)
    signal_type = Column(String, nullable=False)  # BUY | SELL | HOLD
    confidence = Column(Numeric, nullable=False)
    features = Column(JSON, default=dict)
    sba_iterations = Column(Integer, default=100)
    engine_version = Column(String, default="1.0.0-py")
    generated_at = Column(DateTime(timezone=True), default=utcnow)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    __table_args__ = (Index("ix_audit_logs_user_created", "user_id", "created_at"),)
    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    action = Column(String, nullable=False)
    resource_type = Column(String, nullable=True)
    resource_id = Column(String, nullable=True)
    metadata_json = Column(JSON, default=dict)
    pqc_signature = Column(Text, nullable=True)
    signing_key_id = Column(String, nullable=True)  # FK to server_signing_keys for historical verification
    created_at = Column(DateTime(timezone=True), default=utcnow)


class OrderSecurityRecord(Base):
    """Immutable security envelope retained separately from broker state."""
    __tablename__ = "order_security_records"
    __table_args__ = (UniqueConstraint("user_id", "nonce", name="uq_order_security_user_nonce"),)

    id = Column(String, primary_key=True, default=gen_uuid)
    trade_id = Column(String, ForeignKey("trades.id"), nullable=False, unique=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    canonical_order = Column(Text, nullable=False)
    request_hash = Column(String(64), nullable=False)
    nonce = Column(String(128), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    signer_key_id = Column(String, nullable=True)
    signature = Column(Text, nullable=False)
    signature_mode = Column(String(32), nullable=False)  # client_mldsa | development_server
    created_at = Column(DateTime(timezone=True), default=utcnow)

    # Declares the dependency on trades so a flush that adds an order and its
    # record together inserts the trade first; without it the record could be
    # inserted first and violate its foreign key.
    trade = relationship("Trade")


class IdempotencyRecord(Base):
    """Stores the first response for a caller supplied idempotency key."""
    __tablename__ = "idempotency_records"
    __table_args__ = (UniqueConstraint("user_id", "idempotency_key", name="uq_idempotency_user_key"),)

    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    idempotency_key = Column(String(128), nullable=False)
    request_hash = Column(String(64), nullable=False)
    response_json = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)


class ResearchExperiment(Base):
    """Durable, user-owned research experiment and its immutable inputs."""
    __tablename__ = "research_experiments"

    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    strategy_id = Column(String, nullable=False)
    strategy_version = Column(String, nullable=False)
    dataset_id = Column(String, nullable=False)
    dataset_json = Column(JSON, nullable=True)
    dataset_hash = Column(String(64), nullable=False)
    parameters_json = Column(JSON, nullable=False, default=dict)
    parameter_hash = Column(String(64), nullable=False)
    code_commit = Column(String, nullable=False)
    random_seed = Column(Integer, nullable=False)
    execution_model = Column(String, nullable=False)
    latency_model = Column(String, nullable=False)
    status = Column(String(32), nullable=False)
    result_hash = Column(String(64), nullable=False, default="")
    results_json = Column(JSON, nullable=False, default=dict)
    validation_gates_json = Column(JSON, nullable=False, default=dict)
    manifest_signature = Column(Text, nullable=False, default="")
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    # Provenance captured at creation / completion (manifest v2)
    strategy_hash = Column(String(64), nullable=True)
    dependency_lock_hash = Column(String(64), nullable=True)
    engine_version = Column(String(32), nullable=True)
    signing_key_id = Column(String, nullable=True)
    # The exact manifest that was signed, stored verbatim so verification
    # never depends on re-deriving it from lossy column types.
    manifest_json = Column(JSON, nullable=True)
    approved_by = Column(String, ForeignKey("users.id"), nullable=True)
    approved_at = Column(DateTime(timezone=True), nullable=True)


class ResearchTrial(Base):
    """One distinct strategy configuration a user has evaluated.

    A "family" is one research question (the same strategy on the same
    assets); every distinct parameter set tried for it is a trial. The
    Deflated Sharpe test must discount a result by at least this many
    trials, however few the user declares.
    """
    __tablename__ = "research_trials"
    __table_args__ = (UniqueConstraint("user_id", "family_hash", "config_hash",
                                       name="uq_research_trial_config"),)

    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    family_hash = Column(String(64), nullable=False, index=True)
    config_hash = Column(String(64), nullable=False)
    source = Column(String(32), nullable=False)  # backtest | walk_forward | event_backtest | strategy_backtest
    created_at = Column(DateTime(timezone=True), default=utcnow)


class ResearchTrialEvaluation(Base):
    """One accepted run of a strategy version: the trial it counted as, and
    the Sharpe ratio, number of returns and return-series hash of its result
    (research_trials.evaluation_metrics), with the datasets it read.

    One row per job, written with the job's result, so a result the worker
    refuses (a holdout set while it ran, data of another family) or loses
    (its lease lapsed) leaves the trial counted and no evaluation. ``job_id``
    names the job but has no foreign key: finished jobs are purged, and these
    rows stay.

    Immutable: migration 0007 adds triggers that refuse UPDATE, DELETE and
    TRUNCATE, and the CHECKs on the values (PostgreSQL SQL, so they live in
    the migration only).
    """
    __tablename__ = "research_trial_evaluations"
    __table_args__ = (
        ForeignKeyConstraint(["user_id", "family_hash", "config_hash"],
                             ["research_trials.user_id", "research_trials.family_hash",
                              "research_trials.config_hash"],
                             name="fk_research_trial_evaluations_trial"),
    )

    job_id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    family_hash = Column(String(64), nullable=False)
    config_hash = Column(String(64), nullable=False)
    strategy_version_id = Column(String(80), ForeignKey("strategy_versions.version_id"), nullable=False)
    sharpe = Column(Float, nullable=False)
    n_obs = Column(Integer, nullable=False)
    returns_sha256 = Column(String(96), nullable=False)
    dataset_ids = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)


class TrialMatrix(Base):
    """A trial matrix a strategy validation judged (services/trial_matrix.py),
    stored once under its id: ``canonical`` is the canonical JSON the id
    hashes. Immutable, and the id is held to the hash by a CHECK (migration
    0008, PostgreSQL SQL, so it lives in the migration only)."""
    __tablename__ = "trial_matrices"

    matrix_id = Column(String(96), primary_key=True)
    canonical = Column(LargeBinary, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)


class StrategyValidation(Base):
    """The validation policy's verdict on one strategy evaluation
    (services/strategy_validation.py), one row per job: the policy (its id
    and canonical JSON), the matrix it judged, whether every gate passed,
    and every gate's figures. ``job_id`` has no foreign key: finished jobs
    are purged, and verdicts stay.

    Immutable: migration 0008 adds triggers that refuse UPDATE, DELETE and
    TRUNCATE, and CHECKs that hold the policy id to its JSON and the
    verdict's passed, policy and candidate to the columns."""
    __tablename__ = "strategy_validations"

    job_id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    evaluation_job_id = Column(String, ForeignKey("research_trial_evaluations.job_id"), nullable=False)
    strategy_version_id = Column(String(80), ForeignKey("strategy_versions.version_id"), nullable=False)
    family_hash = Column(String(64), nullable=False)
    matrix_id = Column(String(96), ForeignKey("trial_matrices.matrix_id"), nullable=False)
    policy_id = Column(String(96), nullable=False)
    policy_canonical = Column(LargeBinary, nullable=False)
    passed = Column(Boolean, nullable=False)
    verdict = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)


class ValidationManifest(Base):
    """The signed manifest (v3) of one strategy validation's verdict
    (services/validation_manifest.py), stored once under its id: the hash
    of ``canonical``, the manifest's canonical JSON, which ``signature``
    signs with the key ``signing_key_id`` names (None for the HMAC that only
    a non-production deployment may use). Verdicts recorded before manifests
    existed have none.

    Immutable: migration 0009 adds triggers that refuse UPDATE, DELETE and
    TRUNCATE, and CHECKs that hold the id to the hash, the manifest's job and
    key to the row's, and the signature to its form (PostgreSQL SQL, so they
    live in the migration only)."""
    __tablename__ = "validation_manifests"

    manifest_id = Column(String(96), primary_key=True)
    job_id = Column(String, ForeignKey("strategy_validations.job_id"), nullable=False, unique=True)
    canonical = Column(LargeBinary, nullable=False)
    signature = Column(Text, nullable=False)
    signing_key_id = Column(String, ForeignKey("server_signing_keys.key_id"), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)


class HoldoutAttempt(Base):
    """One attempt at grading a validated strategy version on its dataset
    family's sealed holdout (services/holdout_evaluation.py). The research
    evaluator writes it when it claims the ``final_evaluation`` job, before it
    reads any data, so a run that then fails has still spent it. One per
    user, version, dataset family and holdout start. The rows of a research
    family count as trials of that family (trial_matrix.family_state).

    Immutable: migration 0011 adds triggers that refuse UPDATE, DELETE and
    TRUNCATE."""
    __tablename__ = "holdout_attempts"
    __table_args__ = (
        UniqueConstraint("user_id", "strategy_version_id", "dataset_family", "holdout_start",
                         name="uq_holdout_attempts_once_per_holdout"),
        Index("ix_holdout_attempts_family", "user_id", "family_hash"),
    )

    job_id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    validation_job_id = Column(String, ForeignKey("strategy_validations.job_id"), nullable=False)
    strategy_version_id = Column(String(80), ForeignKey("strategy_versions.version_id"), nullable=False)
    family_hash = Column(String(64), nullable=False)
    dataset_family = Column(String(64), nullable=False)
    holdout_start = Column(Date, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)


class HoldoutEvaluation(Base):
    """The grade of one holdout attempt, with every figure it rests on, and
    its signed manifest (services/holdout_evaluation.py). Nothing else holds
    the figures: the job's result and audit event carry ids and pass/fail
    only, and the research worker cannot read this table (db_roles.py).

    Immutable: migration 0011 adds triggers that refuse UPDATE, DELETE and
    TRUNCATE, and CHECKs that hold the policy id and manifest id to their
    hashes, ``passed`` to the grade and the manifest, and the manifest to its
    row (PostgreSQL SQL, so they live in the migration only)."""
    __tablename__ = "holdout_evaluations"

    job_id = Column(String, ForeignKey("holdout_attempts.job_id"), primary_key=True)
    policy_id = Column(String(96), nullable=False)
    policy_canonical = Column(LargeBinary, nullable=False)
    passed = Column(Boolean, nullable=False)
    grade = Column(JSON, nullable=False)
    manifest_id = Column(String(96), nullable=False, unique=True)
    canonical = Column(LargeBinary, nullable=False)
    signature = Column(Text, nullable=False)
    signing_key_id = Column(String, ForeignKey("server_signing_keys.key_id"), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)


class ResearchJob(Base):
    """A research computation queued by the API and run by a worker process.

    status: queued -> running -> succeeded | failed | cancelled. A running
    job is owned by ``worker_id`` only while ``lease_expires_at`` is in the
    future; every write a worker makes is conditional on still holding it.
    """
    __tablename__ = "research_jobs"
    __table_args__ = (Index("ix_research_jobs_status_created", "status", "created_at"),
                      Index("ix_research_jobs_user_status", "user_id", "status"),
                      Index("ix_research_jobs_pipeline", "pipeline_id"),
                      # A retry that names the same key gets the job the first
                      # request queued (research_jobs.enqueue), while the job
                      # is kept (RESEARCH_JOB_RETENTION_DAYS).
                      Index("uq_research_jobs_idempotency", "user_id", "idempotency_key", unique=True,
                            postgresql_where=text("idempotency_key IS NOT NULL")))

    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    kind = Column(String(64), nullable=False)
    params_json = Column(JSON, nullable=False)
    status = Column(String(16), nullable=False, default="queued")
    cancel_requested = Column(Boolean, nullable=False, default=False)
    attempts = Column(Integer, nullable=False, default=0)
    max_attempts = Column(Integer, nullable=False, default=2)
    worker_id = Column(String(128), nullable=True)
    lease_expires_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    result_json = Column(JSON, nullable=True)
    error_status = Column(Integer, nullable=True)   # HTTP-style status of a failure
    error_detail = Column(Text, nullable=True)      # user-safe message, never a traceback
    # The pipeline the job was queued in (its budget paid for it), the step
    # of that pipeline it runs, the caller's idempotency key, and the sha256
    # of the request queued under it (research_jobs.request_fingerprint),
    # which a retry under the key must match.
    pipeline_id = Column(String, ForeignKey("research_pipelines.id"), nullable=True)
    stage = Column(String(64), nullable=True)
    idempotency_key = Column(String(128), nullable=True)
    request_hash = Column(String(64), nullable=True)


class ResearchPipeline(Base):
    """A bounded run of research jobs: it may queue at most ``max_jobs``.

    ``jobs_queued`` counts every job ever queued in the pipeline, whatever
    became of it, and is never lowered: a finished job is purged after
    RESEARCH_JOB_RETENTION_DAYS, but the budget it spent stays spent.
    Migration 0018 adds CHECKs holding the counter within the budget and
    ``closed_at`` to the status.
    """
    __tablename__ = "research_pipelines"
    __table_args__ = (Index("ix_research_pipelines_user_created", "user_id", "created_at"),)

    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    name = Column(String(120), nullable=False)
    max_jobs = Column(Integer, nullable=False)
    jobs_queued = Column(Integer, nullable=False, default=0)
    status = Column(String(16), nullable=False, default="open")   # open | closed
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    closed_at = Column(DateTime(timezone=True), nullable=True)


class ResearchPipelineDay(Base):
    """How many pipeline jobs a user queued on one UTC day, for the daily
    budget (RESEARCH_PIPELINE_DAILY_JOBS). Like a pipeline's counter, it
    counts every job queued and is never lowered."""
    __tablename__ = "research_pipeline_days"

    user_id = Column(String, ForeignKey("users.id"), primary_key=True)
    day = Column(Date, primary_key=True)
    jobs_queued = Column(Integer, nullable=False, default=0)


class ResearchSweep(Base):
    """One strategy version expanded over a parameter grid in a pipeline
    (services/research_sweeps.py): the grid points it generated, how many of
    them were refused as specs, and the MinBTL cap the family was held to.

    Append-only (migration 0019's triggers): a refused candidate counts
    toward its family's trials, so ``n_rejected`` is never lowered. The
    CHECKs on the counts live in the migration.
    """
    __tablename__ = "research_sweeps"
    __table_args__ = (Index("ix_research_sweeps_user_family", "user_id", "family_hash"),
                      Index("ix_research_sweeps_pipeline", "pipeline_id"),
                      # A retried request gets the sweep the first one created.
                      Index("uq_research_sweeps_idempotency", "user_id", "idempotency_key", unique=True,
                            postgresql_where=text("idempotency_key IS NOT NULL")))

    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    pipeline_id = Column(String, ForeignKey("research_pipelines.id"), nullable=False)
    base_version_id = Column(String(80), ForeignKey("strategy_versions.version_id"), nullable=False)
    family_hash = Column(String(64), nullable=False)
    grid = Column(JSON, nullable=False)
    period = Column(String(8), nullable=False)
    initial_capital = Column(Float, nullable=False)
    dataset_ids = Column(JSON, nullable=False)
    n_candidates = Column(Integer, nullable=False)
    n_rejected = Column(Integer, nullable=False)
    # The family's trials once every candidate has run, and the most the
    # MinBTL bound allowed for the sweep's period.
    trials_after = Column(Integer, nullable=False)
    minbtl_cap = Column(Integer, nullable=False)
    idempotency_key = Column(String(128), nullable=True)
    request_hash = Column(String(64), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)


class ResearchSweepCandidate(Base):
    """One grid point of a sweep: the strategy version it registered and the
    trial config its backtest runs as, or why its spec was refused. Its
    backtest job is named once, when it is queued (migration 0019's trigger
    refuses any other change); the job is purged later, the name stays."""
    __tablename__ = "research_sweep_candidates"

    sweep_id = Column(String, ForeignKey("research_sweeps.id"), primary_key=True)
    ordinal = Column(Integer, primary_key=True)
    params = Column(JSON, nullable=False)
    strategy_version_id = Column(String(80), ForeignKey("strategy_versions.version_id"), nullable=True)
    config_hash = Column(String(64), nullable=True)
    rejection = Column(String(500), nullable=True)
    job_id = Column(String, nullable=True)
    queued_at = Column(DateTime(timezone=True), nullable=True)


class DatasetVersion(Base):
    """A frozen, content-addressed dataset (services/dataset_registry.py).

    Immutable: migration 0004 adds triggers that refuse UPDATE, DELETE and
    TRUNCATE, and CHECKs that the id is the hash of ``manifest_canonical``
    and the identity columns are the identity inside it. Those are
    PostgreSQL SQL, so they live in the migration only (alembic's schema
    comparison ignores CHECKs, and the SQLite import tests build this model).
    """
    __tablename__ = "dataset_versions"

    dataset_id = Column(String(80), primary_key=True)
    family = Column(String(64), nullable=False)
    provider = Column(String(64), nullable=False)
    licence_id = Column(String(64), nullable=False)
    frequency = Column(String(64), nullable=False)
    session_boundary = Column(String(64), nullable=False)
    adjustment = Column(String(64), nullable=False)
    survivorship = Column(String(64), nullable=False)
    personal_use_only = Column(Boolean, nullable=False)
    redistribution_allowed = Column(Boolean, nullable=False)
    point_in_time = Column(Boolean, nullable=False)
    coverage_start = Column(Date, nullable=False)
    coverage_end = Column(Date, nullable=False)
    symbols = Column(JSON, nullable=False)
    manifest_canonical = Column(LargeBinary, nullable=False)
    tables_json = Column(JSON, nullable=False)
    blob_format = Column(String(32), nullable=False)
    blob_sha256 = Column(String(64), nullable=False)
    blob_size = Column(BigInteger, nullable=False)
    provider_client = Column(String(64), nullable=False)
    request_json = Column(JSON, nullable=False)
    retrieved_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)


class UserDataset(Base):
    """A user's claim on a dataset they snapshotted; datasets under a
    personal-use licence are listed only to the users who fetched them."""
    __tablename__ = "user_datasets"

    user_id = Column(String, ForeignKey("users.id"), primary_key=True)
    dataset_id = Column(String(80), ForeignKey("dataset_versions.dataset_id"), primary_key=True)
    job_id = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)


class DatasetHoldout(Base):
    """Where a dataset family's sealed final holdout starts
    (services/dataset_holdouts.py): research sees the family's data only
    before the latest row's ``holdout_start``.

    Append-only: migration 0005 adds triggers that refuse UPDATE, DELETE and
    TRUNCATE, and an insert whose start is not later than the family's
    current one. Those are PostgreSQL SQL, so they live in the migration only.
    """
    __tablename__ = "dataset_holdouts"

    family = Column(String(64), primary_key=True)
    holdout_start = Column(Date, primary_key=True)
    set_by = Column(String(320), nullable=False)
    reason = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)


class StrategyVersion(Base):
    """A strategy spec, stored as its canonical JSON under its content hash
    (services/strategy_registry.py). One row serves every user who wrote the
    same spec.

    Immutable: migration 0006 adds triggers that refuse UPDATE, DELETE and
    TRUNCATE, and CHECKs that the id is the hash of ``spec_canonical`` and
    ``dsl``, ``template`` and ``family`` are the values inside it. Those are
    PostgreSQL SQL, so they live in the migration only.
    """
    __tablename__ = "strategy_versions"

    version_id = Column(String(80), primary_key=True)
    dsl = Column(String(32), nullable=False)
    template = Column(String(32), nullable=False)
    family = Column(String(64), nullable=False)
    spec_canonical = Column(LargeBinary, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)


class UserStrategyVersion(Base):
    """A user who registered a strategy version; append-only (migration 0006)."""
    __tablename__ = "user_strategy_versions"

    user_id = Column(String, ForeignKey("users.id"), primary_key=True)
    version_id = Column(String(80), ForeignKey("strategy_versions.version_id"), primary_key=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)


class UserStrategyTag(Base):
    """A user's own name for a version they hold. A tag is never moved or
    removed (migration 0006): it always names the version it was given to."""
    __tablename__ = "user_strategy_tags"
    __table_args__ = (
        ForeignKeyConstraint(["user_id", "version_id"],
                             ["user_strategy_versions.user_id", "user_strategy_versions.version_id"],
                             name="fk_user_strategy_tags_held_version"),
    )

    user_id = Column(String, primary_key=True)
    label = Column(String(64), primary_key=True)
    version_id = Column(String(80), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)


class ResearchWorker(Base):
    """Liveness record of a research worker process (heartbeat)."""
    __tablename__ = "research_workers"

    id = Column(String(128), primary_key=True)
    hostname = Column(String(255), nullable=False)
    pid = Column(Integer, nullable=False)
    started_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    last_seen_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    current_job_id = Column(String, nullable=True)
    # The QS_PROCESS_ROLE it runs as, which decides the job kinds it claims.
    role = Column(String(32), nullable=False, default="research-worker",
                  server_default="research-worker")


class AuditChainLink(Base):
    """Hash-chain link for tamper-evident audit history."""
    __tablename__ = "audit_chain_links"

    id = Column(String, primary_key=True, default=gen_uuid)
    sequence = Column(BigInteger, nullable=False, unique=True)
    audit_log_id = Column(String, ForeignKey("audit_logs.id"), nullable=False, unique=True, index=True)
    previous_hash = Column(String(64), nullable=False)
    entry_hash = Column(String(64), nullable=False, unique=True)
    checkpoint_signature = Column(Text, nullable=False)
    signing_key_id = Column(String, nullable=True)  # FK to server_signing_keys for historical verification
    created_at = Column(DateTime(timezone=True), default=utcnow)


# --- Refresh Token (Item 1: cookie-based auth with rotation) ------------------

class RefreshToken(Base):
    """Refresh tokens for cookie-based session management with family rotation."""
    __tablename__ = "refresh_tokens"
    __table_args__ = (UniqueConstraint("token_hash", name="uq_refresh_token_hash"),)

    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    token_hash = Column(String(64), nullable=False)  # SHA-256 of the opaque token (unique: uq_refresh_token_hash)
    family_id = Column(String, nullable=False, index=True)       # rotation chain family
    is_used = Column(Boolean, default=False)                     # set True on rotation
    is_revoked = Column(Boolean, default=False)                  # set True on reuse-detection
    created_at = Column(DateTime(timezone=True), default=utcnow)
    expires_at = Column(DateTime(timezone=True), nullable=False)


# --- Server Signing Key History (Item 8: audit key rotation) ------------------

class ServerSigningKey(Base):
    """Historical record of server ML-DSA signing keys for audit verification."""
    __tablename__ = "server_signing_keys"

    key_id = Column(String, primary_key=True, default=gen_uuid)
    algorithm = Column(String(32), nullable=False, default="ML-DSA-65")
    public_key = Column(Text, nullable=False)      # base64 public key
    fingerprint = Column(String(64), nullable=False, unique=True)  # SHA-256 of public key bytes
    status = Column(String(16), nullable=False, default="active")  # active | retired | revoked
    created_at = Column(DateTime(timezone=True), default=utcnow)
    activated_at = Column(DateTime(timezone=True), default=utcnow)
    retired_at = Column(DateTime(timezone=True), nullable=True)
