"""The AI tier's settings, read from the environment.

This replaces backend.config for the AI tier, which must never import it:
backend.config generates or loads the API's keys. It also refuses to load
under the AI tier's process roles (QS_PROCESS_ROLE ai-worker or ai-job), so
an accidental import fails at once instead of quietly running with them.

Every setting may also be given as a file named by NAME_FILE, as in
backend.config.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

WORKER_ROLE = "ai-worker"
JOB_ROLE = "ai-job"

# Settings the AI worker must never be given: the trusted tier's database
# URL, keys and secrets, and every database password but its own. In
# production it refuses to start while any of them, or its *_FILE form, is set.
TRUSTED_SECRETS = (
    "DATABASE_URL", "POSTGRES_PASSWORD", "PGPASSWORD",
    "DB_APP_PASSWORD", "DB_RESEARCH_WORKER_PASSWORD", "DB_EVALUATOR_PASSWORD",
    "DB_MIGRATOR_PASSWORD", "DB_BACKUP_PASSWORD",
    "JWT_PRIVATE_KEY", "SERVER_DSA_PRIVATE_KEY", "PRIVATE_KEY_ENCRYPTION_KEY",
    "WEBHOOK_ENCRYPTION_KEY", "REFRESH_TOKEN_SECRET", "CSRF_SECRET", "REDIS_URL", "REDIS_PASSWORD",
)


class SettingsError(RuntimeError):
    pass


def setting(name: str, default: str | None = None, environ=None) -> str | None:
    environ = os.environ if environ is None else environ
    value = environ.get(name)
    if value:
        return value
    file_path = environ.get(f"{name}_FILE")
    if file_path:
        return Path(file_path).read_text(encoding="utf-8").strip()
    return default


def _number(name: str, default: str, environ, minimum: float) -> float:
    raw = setting(name, default, environ)
    try:
        value = float(raw)
    except ValueError:
        raise SettingsError(f"{name} must be a number, not {raw!r}") from None
    if value < minimum:
        raise SettingsError(f"{name} must be at least {minimum}")
    return value


def libpq_url(url: str) -> str:
    """A SQLAlchemy-style psycopg URL, as libpq takes it."""
    for prefix in ("postgresql+psycopg://", "postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql://" + url[len(prefix):]
    raise SettingsError("AI_DATABASE_URL must be a PostgreSQL URL (postgresql://qs_ai_worker:...@host:5432/db)")


@dataclass(frozen=True)
class Settings:
    environment: str
    database_url: str
    schema: str
    poll_seconds: float
    heartbeat_seconds: float
    lease_seconds: float
    job_memory_bytes: int
    result_max_bytes: int
    retention_days: float


def load(environ=None) -> Settings:
    """The AI worker's settings; raises SettingsError on anything wrong."""
    environ = os.environ if environ is None else environ
    role = setting("QS_PROCESS_ROLE", None, environ)
    if role != WORKER_ROLE:
        raise SettingsError(f"the AI worker runs as QS_PROCESS_ROLE={WORKER_ROLE}, not {role!r}")
    environment = (setting("ENVIRONMENT", "development", environ) or "development").lower()
    if environment == "production":
        present = [name for name in TRUSTED_SECRETS if environ.get(name) or environ.get(f"{name}_FILE")]
        if present:
            raise SettingsError("the AI worker must not be given the trusted tier's secrets: "
                                + ", ".join(present))
    url = setting("AI_DATABASE_URL", None, environ)
    if not url:
        raise SettingsError("AI_DATABASE_URL is required (connect as qs_ai_worker)")
    schema = setting("AI_DB_SCHEMA", "ai", environ)
    if not re.fullmatch(r"[a-z_][a-z0-9_]{0,62}", schema or ""):
        raise SettingsError(f"AI_DB_SCHEMA must be a plain schema name, not {schema!r}")
    heartbeat = _number("AI_WORKER_HEARTBEAT_SECONDS", "5", environ, 0.1)
    lease = _number("AI_JOB_LEASE_SECONDS", "60", environ, 1)
    if lease <= 2 * heartbeat:
        raise SettingsError("AI_JOB_LEASE_SECONDS must be more than twice AI_WORKER_HEARTBEAT_SECONDS")
    return Settings(
        environment=environment,
        database_url=libpq_url(url),
        schema=schema,
        poll_seconds=_number("AI_WORKER_POLL_SECONDS", "2", environ, 0.05),
        heartbeat_seconds=heartbeat,
        lease_seconds=lease,
        job_memory_bytes=int(_number("AI_JOB_MEMORY_BYTES", str(1 << 30), environ, 64 << 20)),
        result_max_bytes=int(_number("AI_JOB_RESULT_MAX_BYTES", str(1 << 20), environ, 1024)),
        retention_days=_number("AI_JOB_RETENTION_DAYS", "7", environ, 0),
    )
