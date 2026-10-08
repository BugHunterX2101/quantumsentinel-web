"""QuantumSentinel — database layer: PostgreSQL via SQLAlchemy 2 and psycopg 3.

Every process (API worker, research worker) owns one connection pool, sized
by DB_POOL_SIZE + DB_MAX_OVERFLOW. Each new connection is pinned to UTC and
given server-side statement, lock and idle-transaction limits, so a stuck
query or forgotten transaction cannot hold locks indefinitely.

The schema is versioned with Alembic (backend/migrations). ``init_db`` brings
the database to the latest revision; concurrent callers (several gunicorn
workers starting at once) are serialised by a transaction-scoped advisory
lock, and PostgreSQL's transactional DDL makes each upgrade all-or-nothing.
"""
import logging
import time
from pathlib import Path

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import declarative_base, sessionmaker
from starlette.requests import Request

from .config import (DATABASE_URL, DB_IDLE_IN_TRANSACTION_TIMEOUT_MS, DB_LOCK_TIMEOUT_MS,
                     DB_MAX_OVERFLOW, DB_MIGRATE_ON_STARTUP, DB_POOL_SIZE, DB_POOL_TIMEOUT_SECONDS,
                     DB_STATEMENT_TIMEOUT_MS)

log = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
# Arbitrary constant naming the "schema migration" advisory lock.
_MIGRATION_LOCK_KEY = 7_316_405_312_118
_MIGRATION_LOCK_WAIT_SECONDS = 600

engine = create_engine(
    DATABASE_URL,
    pool_size=DB_POOL_SIZE,
    max_overflow=DB_MAX_OVERFLOW,
    pool_timeout=DB_POOL_TIMEOUT_SECONDS,
    # Validate a pooled connection before handing it out, so a database
    # restart or failover costs one reconnect instead of a failed request.
    pool_pre_ping=True,
    # Recycle connections before typical server/proxy idle cut-offs.
    pool_recycle=1800,
    connect_args={"connect_timeout": 10, "application_name": "quantumsentinel"},
)


@event.listens_for(engine, "connect")
def _configure_session(dbapi_connection, _record) -> None:
    # Timestamps come back in UTC whatever the server's own TimeZone is, so
    # nothing derived from them (such as the audit chain's hashes) depends on
    # how the server happens to be configured.
    with dbapi_connection.cursor() as cursor:
        cursor.execute(
            "SELECT set_config('TimeZone', 'UTC', false),"
            " set_config('statement_timeout', %s, false),"
            " set_config('lock_timeout', %s, false),"
            " set_config('idle_in_transaction_session_timeout', %s, false)",
            (str(DB_STATEMENT_TIMEOUT_MS), str(DB_LOCK_TIMEOUT_MS),
             str(DB_IDLE_IN_TRANSACTION_TIMEOUT_MS)),
        )
    dbapi_connection.commit()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Sessions for GET and HEAD requests run in autocommit, which saves two round
# trips per request: BEGIN before the first query and ROLLBACK when the
# connection goes back to the pool. Reads return the same rows: under READ
# COMMITTED, PostgreSQL's default, every statement takes its own snapshot
# inside a transaction too. What autocommit loses is multi-statement
# atomicity and row locks held to commit, so a GET handler must not write
# more than one statement or lock rows (writing one row, as opening a paper
# account on first view does, stays atomic), nor stream rows through a
# server-side cursor (yield_per), which PostgreSQL opens only inside a
# transaction: such a handler takes get_transactional_db. Same pool: the
# isolation level is set on checkout and restored on return, without a
# round trip.
_ReadSession = sessionmaker(autocommit=False, autoflush=False,
                            bind=engine.execution_options(isolation_level="AUTOCOMMIT"))
_READ_METHODS = frozenset({"GET", "HEAD"})


def get_db(request: Request):
    db = (_ReadSession if request.method in _READ_METHODS else SessionLocal)()
    try:
        yield db
    finally:
        db.close()


def get_transactional_db():
    """A session that runs in a transaction whatever the request method."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _alembic_config(connection):
    from alembic.config import Config

    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    config.attributes["connection"] = connection
    return config


def head_revision() -> str:
    from alembic.script import ScriptDirectory

    return ScriptDirectory.from_config(_alembic_config(None)).get_current_head()


def current_revision(bind=None) -> str | None:
    from alembic.migration import MigrationContext

    if bind is not None:
        return MigrationContext.configure(bind).get_current_revision()
    with engine.connect() as connection:
        return MigrationContext.configure(connection).get_current_revision()


def schema_is_current() -> bool:
    return current_revision() == head_revision()


def migrate(target_engine=None) -> str:
    """Upgrade the schema to the latest revision; returns that revision."""
    from alembic import command

    with (target_engine or engine).begin() as connection:
        # A migration may build indexes on large tables: no statement limit.
        connection.execute(text("SET LOCAL statement_timeout = 0"))
        deadline = time.monotonic() + _MIGRATION_LOCK_WAIT_SECONDS
        while not connection.execute(text("SELECT pg_try_advisory_xact_lock(:key)"),
                                     {"key": _MIGRATION_LOCK_KEY}).scalar():
            if time.monotonic() > deadline:
                raise RuntimeError("timed out waiting for another process's schema migration")
            time.sleep(0.5)
        before = current_revision(connection)
        command.upgrade(_alembic_config(connection), "head")
        after = current_revision(connection)
    if before != after:
        log.info("database schema migrated from %s to %s", before or "empty", after)
    return after


def init_db() -> None:
    """Bring the schema to the latest revision, or refuse to run on a stale one."""
    if DB_MIGRATE_ON_STARTUP:
        migrate()
    elif not schema_is_current():
        raise RuntimeError(f"database schema is at revision {current_revision()}, expected "
                           f"{head_revision()}: run `python -m backend.manage migrate`")
