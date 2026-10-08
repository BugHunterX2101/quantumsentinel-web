import os
import secrets
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

def _setting(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    file_path = os.getenv(f"{name}_FILE")
    if file_path:
        return Path(file_path).read_text(encoding="utf-8").strip()
    return default


ENVIRONMENT = _setting("ENVIRONMENT", "development").lower()
# Which process this is. In production each requires only the secrets it
# uses: the API all of them; a research worker its database URL and the
# server signing key; a research job process none, and its worker starts it
# with none (backend/worker.py).
PROCESS_ROLE = _setting("QS_PROCESS_ROLE", "api")
if PROCESS_ROLE not in ("api", "research-worker", "research-job"):
    raise RuntimeError("QS_PROCESS_ROLE must be one of: api, research-worker, research-job")
JWT_ALGORITHM = "RS256"
_jwt_private_pem = _setting("JWT_PRIVATE_KEY")
_jwt_public_pem = _setting("JWT_PUBLIC_KEY")
if _jwt_private_pem and _jwt_public_pem:
    JWT_SIGNING_KEY = _jwt_private_pem.replace("\\n", "\n").encode()
    JWT_VERIFY_KEY = _jwt_public_pem.replace("\\n", "\n").encode()
else:
    _jwt_private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    JWT_SIGNING_KEY = _jwt_private.private_bytes(serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    JWT_VERIFY_KEY = _jwt_private.public_key().public_bytes(serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo)
JWT_EXPIRE_SECONDS = int(os.getenv("JWT_EXPIRE_SECONDS", "900"))
JWT_ISSUER = _setting("JWT_ISSUER", "quantumsentinel")
JWT_AUDIENCE = _setting("JWT_AUDIENCE", "quantumsentinel-api")
REFRESH_SESSION_SECONDS = int(os.getenv("REFRESH_SESSION_SECONDS", "3600"))

# --- Refresh-token rotation ---------------------------------------------------
REFRESH_TOKEN_SECRET = _setting("REFRESH_TOKEN_SECRET") or secrets.token_hex(32)
REFRESH_TOKEN_SECONDS = int(os.getenv("REFRESH_TOKEN_SECONDS", "86400"))
# Absolute cap on a refresh-token family's total lifetime, independent of the
# sliding REFRESH_TOKEN_SECONDS window each rotation grants. Without this, a
# session that keeps getting silently rotated (a browser tab left open, or a
# stolen token an attacker keeps refreshing) never expires and never forces
# re-authentication with the password.
REFRESH_ABSOLUTE_SESSION_SECONDS = int(os.getenv("REFRESH_ABSOLUTE_SESSION_SECONDS", str(30 * 86400)))

# --- HttpOnly cookie settings -------------------------------------------------
COOKIE_DOMAIN = _setting("COOKIE_DOMAIN")          # None = browser infers from origin
COOKIE_SECURE = ENVIRONMENT == "production"         # always True in prod
COOKIE_SAMESITE = _setting("COOKIE_SAMESITE", "strict")

# --- CSRF double-submit -------------------------------------------------------
CSRF_SECRET = _setting("CSRF_SECRET") or secrets.token_hex(32)

WEBHOOK_ENCRYPTION_KEY = _setting("WEBHOOK_ENCRYPTION_KEY")
PRIVATE_KEY_ENCRYPTION_KEY = _setting("PRIVATE_KEY_ENCRYPTION_KEY")
SERVER_DSA_PRIVATE_KEY = _setting("SERVER_DSA_PRIVATE_KEY")
SERVER_DSA_PUBLIC_KEY = _setting("SERVER_DSA_PUBLIC_KEY")
SERVER_DSA_CREATED_AT = _setting("SERVER_DSA_CREATED_AT")

# --- Server identity pinning (Item 4) ----------------------------------------
TRUSTED_SERVER_DSA_FINGERPRINT = _setting("TRUSTED_SERVER_DSA_FINGERPRINT")

# Operator privileges come from users.role, provisioned with
# `python -m backend.manage set-role <email> <role>` — never from anything a
# self-registering user controls (such as registering a configured email).
OPERATOR_ROLES = frozenset({"admin", "risk_admin"})

# --- Paper-trading ledger -------------------------------------------------------
# Starting capital is a server-side constant: clients never choose their own
# buying power.
PAPER_INITIAL_CASH = float(os.getenv("PAPER_INITIAL_CASH", "100000"))
# Per-asset concentration cap as a fraction of account equity.
PAPER_MAX_POSITION_FRACTION = float(os.getenv("PAPER_MAX_POSITION_FRACTION", "0.05"))
# Resting conditional orders are filled by a background sweeper, never by reads.
ORDER_SWEEPER_ENABLED = os.getenv("ORDER_SWEEPER_ENABLED", "true").lower() == "true"
ORDER_SWEEP_INTERVAL_SECONDS = float(os.getenv("ORDER_SWEEP_INTERVAL_SECONDS", "5"))
# An order only executes against a quote whose last trade is at most this old
# (by the exchange's own trade timestamp, not when it was fetched). 20 minutes
# admits exchanges whose feed is delayed ~15 minutes, and excludes a closed
# market's last close.
MAX_QUOTE_AGE_SECONDS = float(os.getenv("MAX_QUOTE_AGE_SECONDS", "1200"))

# --- Research job queue -----------------------------------------------------------
# Research runs in a worker process, never in the API process. "embedded" (the
# default): every API process starts one worker subprocess and stops it with
# itself, so a deployment needs no extra service; "external": workers are
# deployed separately (`python -m backend.worker`) and the API starts none;
# "off": nothing runs queued jobs (tests).
RESEARCH_WORKER_MODE = _setting("RESEARCH_WORKER_MODE", "embedded").lower()
# Hard limit on one job's run time; the job process is killed when it passes.
RESEARCH_JOB_TIMEOUT_SECONDS = float(os.getenv("RESEARCH_JOB_TIMEOUT_SECONDS", "900"))
# A running job belongs to its worker only while the worker keeps renewing
# this lease; a worker that stops renewing is presumed dead and the job is
# handed to another worker (at most RESEARCH_JOB_MAX_ATTEMPTS runs in total).
RESEARCH_JOB_LEASE_SECONDS = float(os.getenv("RESEARCH_JOB_LEASE_SECONDS", "60"))
RESEARCH_JOB_MAX_ATTEMPTS = int(os.getenv("RESEARCH_JOB_MAX_ATTEMPTS", "2"))
# Queued plus running jobs one user may have at a time.
RESEARCH_MAX_ACTIVE_JOBS_PER_USER = int(os.getenv("RESEARCH_MAX_ACTIVE_JOBS_PER_USER", "3"))
# Finished jobs (and their results) are deleted after this many days.
RESEARCH_JOB_RETENTION_DAYS = float(os.getenv("RESEARCH_JOB_RETENTION_DAYS", "7"))
RESEARCH_WORKER_POLL_SECONDS = float(os.getenv("RESEARCH_WORKER_POLL_SECONDS", "1"))
if RESEARCH_WORKER_MODE not in ("embedded", "external", "off"):
    raise RuntimeError("RESEARCH_WORKER_MODE must be one of: embedded, external, off")

def _postgres_url(url: str) -> str:
    """PostgreSQL is the only supported database. Bare postgresql:// and
    postgres:// URLs (as managed providers issue them) are pinned to the
    psycopg 3 driver; SQLAlchemy would otherwise pick psycopg2, which is not
    installed, or reject postgres:// outright."""
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    if not url.startswith("postgresql+psycopg://"):
        raise RuntimeError("DATABASE_URL must be a PostgreSQL URL "
                           "(postgresql+psycopg://user:password@host:5432/dbname)")
    return url


DATABASE_URL = _postgres_url(_setting(
    "DATABASE_URL", "postgresql+psycopg://quantumsentinel:quantumsentinel@localhost:5432/quantumsentinel"))
# Connections each process may hold: DB_POOL_SIZE kept open plus up to
# DB_MAX_OVERFLOW more. Every API process and research worker has its own
# pool, so the sum across all of them must stay below the server's
# max_connections (PostgreSQL's default is 100). Overflow connections are
# closed as soon as they are returned, so steady load above DB_POOL_SIZE
# reconnects constantly (measured: 709 new sessions per 20k requests at
# 10+10 vs 1 at 20+0, +21% throughput, p99 108 -> 63 ms): size the pool for
# the load and keep overflow for bursts only.
DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "20"))
DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "0"))
# A request that cannot get a connection within this long fails (503) instead
# of queueing indefinitely behind a saturated pool.
DB_POOL_TIMEOUT_SECONDS = float(os.getenv("DB_POOL_TIMEOUT_SECONDS", "10"))
# Server-side limits on every session: no statement runs longer than this, no
# lock is waited on longer than this, and a session left idle inside an open
# transaction is terminated (its locks would otherwise block everyone).
DB_STATEMENT_TIMEOUT_MS = int(os.getenv("DB_STATEMENT_TIMEOUT_MS", "30000"))
DB_LOCK_TIMEOUT_MS = int(os.getenv("DB_LOCK_TIMEOUT_MS", "10000"))
DB_IDLE_IN_TRANSACTION_TIMEOUT_MS = int(os.getenv("DB_IDLE_IN_TRANSACTION_TIMEOUT_MS", "60000"))
# Apply pending schema migrations when an API process starts. Concurrent
# starts are serialised by a database lock, so this is safe with many
# workers; disable it to run `python -m backend.manage migrate` as a separate
# release step instead.
DB_MIGRATE_ON_STARTUP = os.getenv("DB_MIGRATE_ON_STARTUP", "true").lower() == "true"
if DB_POOL_SIZE < 1 or DB_MAX_OVERFLOW < 0:
    raise RuntimeError("DB_POOL_SIZE must be >= 1 and DB_MAX_OVERFLOW >= 0")
REDIS_URL = _setting("REDIS_URL")
PQC_PROVIDER = _setting("PQC_PROVIDER", "reference")
PQC_PROVIDER_URL = _setting("PQC_PROVIDER_URL")

# Wildcard CORS is convenient for a local demo, but is never acceptable once
# credentials are used in a deployed environment.
CORS_ORIGINS = [origin.strip() for origin in os.getenv(
    "CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"
).split(",") if origin.strip()]
ALLOWED_HOSTS = [host.strip() for host in os.getenv(
    "ALLOWED_HOSTS", "localhost,127.0.0.1,testserver"
).split(",") if host.strip()]

if ENVIRONMENT == "production" and PROCESS_ROLE != "research-job":
    # The API and research workers: both sign audit events and use the database.
    if not SERVER_DSA_PRIVATE_KEY or not SERVER_DSA_PUBLIC_KEY:
        raise RuntimeError("SERVER_DSA_PRIVATE_KEY and SERVER_DSA_PUBLIC_KEY are required in production")
    if not _setting("DATABASE_URL"):
        raise RuntimeError("DATABASE_URL is required in production")
    if PQC_PROVIDER == "reference" or not PQC_PROVIDER_URL:
        raise RuntimeError("Production requires a configured external liboqs/HSM PQC provider")
if ENVIRONMENT == "production" and PROCESS_ROLE == "api":
    if not _jwt_private_pem or not _jwt_public_pem:
        raise RuntimeError("JWT_PRIVATE_KEY and JWT_PUBLIC_KEY are required in production")
    if "*" in CORS_ORIGINS:
        raise RuntimeError("CORS_ORIGINS must be explicit in production")
    if not WEBHOOK_ENCRYPTION_KEY:
        raise RuntimeError("WEBHOOK_ENCRYPTION_KEY is required in production")
    if not PRIVATE_KEY_ENCRYPTION_KEY:
        raise RuntimeError("PRIVATE_KEY_ENCRYPTION_KEY is required in production")
    if not REDIS_URL:
        raise RuntimeError("REDIS_URL is required in production")
    if not _setting("REFRESH_TOKEN_SECRET"):
        raise RuntimeError("REFRESH_TOKEN_SECRET is required in production")
    if not _setting("CSRF_SECRET"):
        raise RuntimeError("CSRF_SECRET is required in production")
