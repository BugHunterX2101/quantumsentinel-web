"""Test-suite settings applied before any backend module is imported.

Every test runs against PostgreSQL, the only database the application
supports. TEST_DATABASE_URL names a database the suite may use freely
(created if missing); the default matches the docker-compose.yml service:

    docker compose up -d postgres
    python -m pytest -q

Isolation is by schema: the application's own engine gets one schema for the
whole run, and each test that asks for ``make_engine`` gets a fresh schema of
its own, dropped when the test ends.
"""
import os
import uuid

import psycopg
import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.engine import make_url

# Tests drive research jobs explicitly (tests/test_research_jobs.py); an app
# started by TestClient must not spawn a real worker process.
os.environ.setdefault("RESEARCH_WORKER_MODE", "off")

TEST_DATABASE_URL = make_url(os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+psycopg://quantumsentinel:quantumsentinel@localhost:5432/quantumsentinel_test"))


def _libpq(url) -> str:
    return url.set(drivername="postgresql").render_as_string(hide_password=False)


def _ensure_database() -> None:
    try:
        psycopg.connect(_libpq(TEST_DATABASE_URL), connect_timeout=10).close()
        return
    except psycopg.OperationalError as exc:
        if "does not exist" not in str(exc):
            raise pytest.UsageError(
                f"PostgreSQL is not reachable at {TEST_DATABASE_URL.render_as_string()} "
                f"(set TEST_DATABASE_URL): {exc}") from exc
    with psycopg.connect(_libpq(TEST_DATABASE_URL.set(database="postgres")), autocommit=True) as conn:
        conn.execute(f'CREATE DATABASE "{TEST_DATABASE_URL.database}"')


def _admin(statement: str) -> None:
    with psycopg.connect(_libpq(TEST_DATABASE_URL), autocommit=True) as conn:
        conn.execute(statement)


def schema_url(schema: str):
    """TEST_DATABASE_URL with every connection confined to ``schema``."""
    return TEST_DATABASE_URL.update_query_dict({"options": f"-csearch_path={schema}"})


def new_schema(prefix: str) -> str:
    schema = f"{prefix}_{uuid.uuid4().hex[:12]}"
    _admin(f'CREATE SCHEMA "{schema}"')
    return schema


def drop_schema(schema: str) -> None:
    _admin(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')


_ensure_database()
APP_SCHEMA = new_schema("qs_app")
os.environ["DATABASE_URL"] = schema_url(APP_SCHEMA).render_as_string(hide_password=False)


def pytest_sessionfinish(session, exitstatus):
    from backend.database import engine

    engine.dispose()
    drop_schema(APP_SCHEMA)


@pytest.fixture
def make_engine():
    """Factory for engines on a fresh, fully created schema of their own.

    The engines use the application's own per-connection settings (UTC,
    statement/lock/idle-transaction limits). Everything is disposed and the
    schemas dropped when the test ends.
    """
    from backend import models  # noqa: F401  (registers every table)
    from backend.database import Base, _configure_session, migrate

    created = []

    def factory(migrated=False, **engine_kwargs):
        """``migrated=True`` builds the schema with the Alembic migrations
        (as a deployment does) instead of straight from the models."""
        schema = new_schema("qs_test")
        # Tagged with the schema name so teardown can end exactly this
        # engine's connections, including any a test left open.
        engine = create_engine(schema_url(schema), connect_args={"application_name": schema},
                               **engine_kwargs)
        event.listen(engine, "connect", _configure_session)
        created.append((engine, schema))
        if migrated:
            migrate(engine)
        else:
            Base.metadata.create_all(engine)
        return engine

    yield factory
    for engine, schema in created:
        engine.dispose()
        # A session a test never closed would otherwise hold locks until
        # idle_in_transaction_session_timeout ends it, stalling the drop.
        _admin("SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
               f"WHERE application_name = '{schema}' AND pid <> pg_backend_pid()")
        drop_schema(schema)
