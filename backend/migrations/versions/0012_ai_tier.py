"""The AI tier's own schema: feature flags, its job queue and its workers.

The AI tier (backend/ai) runs untrusted-input work, such as fetching paper
metadata from the internet, as qs_ai_worker. That role has no privilege on
the application schema, not even USAGE (db_roles.py), so its tables live in
a schema of their own beside it: ``ai`` beside ``public``
(db_roles.ai_schema_for).

* ``flags``: the feature flags. A stage of the AI tier runs only while both
  ``research_intelligence`` and the stage's own flag are on; a missing row
  is off. Operators set them with ``manage set-flag``; the application checks
  them when it enqueues a job and the AI worker inside its claim query, so a
  change takes effect without a restart.
* ``jobs``: the AI job queue, which only the application fills.
* ``workers``: the AI workers' heartbeats.

The owner role may not create schemas, so on a provisioned database the
superuser creates this one: ``provision-roles``, or ``database.migrate`` run
as the superuser, before it becomes the owner. A migrator that finds it
missing stops with an error that says so. Downgrading drops the tables and
keeps the (then empty) schema, so the owner role can upgrade again.

Revision ID: 0012
Revises: 0011
Create Date: 2026-11-20
"""
from typing import Sequence, Union

from alembic import op

revision: str = "0012"
down_revision: Union[str, Sequence[str], None] = "0011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _ai_schema() -> str:
    from backend.db_roles import ai_schema_for

    return ai_schema_for(op.get_bind().exec_driver_sql("SELECT current_schema()").scalar())


def upgrade() -> None:
    bind = op.get_bind()
    ai = _ai_schema()
    if not bind.exec_driver_sql("SELECT to_regnamespace(%(s)s) IS NOT NULL", {"s": ai}).scalar():
        if not bind.exec_driver_sql("SELECT has_database_privilege(current_database(), 'CREATE')").scalar():
            user = bind.exec_driver_sql("SELECT current_user").scalar()
            raise RuntimeError(
                f"schema {ai} does not exist and {user} may not create it: run "
                "`python -m backend.manage provision-roles` as the database superuser, then migrate again")
        op.execute(f"CREATE SCHEMA {_quote(ai)}")
    q = _quote(ai)
    op.execute(f"""
        CREATE TABLE {q}.flags (
            name varchar(64) PRIMARY KEY CHECK (name ~ '^[a-z][a-z0-9_]*$'),
            enabled boolean NOT NULL,
            set_at timestamptz NOT NULL DEFAULT now(),
            set_by varchar(320) NOT NULL CHECK (length(btrim(set_by)) > 0),
            reason text NOT NULL CHECK (length(btrim(reason)) > 0)
        )""")
    op.execute(f"""
        CREATE TABLE {q}.jobs (
            id varchar(36) PRIMARY KEY,
            kind varchar(64) NOT NULL,
            status varchar(16) NOT NULL DEFAULT 'queued'
                CHECK (status IN ('queued', 'running', 'succeeded', 'failed')),
            payload jsonb NOT NULL DEFAULT '{{}}'::jsonb,
            result jsonb,
            error text,
            requested_by varchar(320) NOT NULL,
            attempts integer NOT NULL DEFAULT 0 CHECK (attempts >= 0),
            max_attempts integer NOT NULL DEFAULT 1 CHECK (max_attempts >= 1),
            worker_id varchar(255),
            lease_expires_at timestamptz,
            created_at timestamptz NOT NULL DEFAULT now(),
            started_at timestamptz,
            finished_at timestamptz,
            CHECK ((status = 'running') = (lease_expires_at IS NOT NULL AND worker_id IS NOT NULL))
        )""")
    op.execute(f"CREATE INDEX jobs_queued ON {q}.jobs (created_at, id) WHERE status = 'queued'")
    op.execute(f"CREATE INDEX jobs_running ON {q}.jobs (lease_expires_at) WHERE status = 'running'")
    op.execute(f"CREATE INDEX jobs_finished ON {q}.jobs (finished_at) WHERE status IN ('succeeded', 'failed')")
    op.execute(f"""
        CREATE TABLE {q}.workers (
            id varchar(255) PRIMARY KEY,
            hostname varchar(255) NOT NULL,
            pid integer NOT NULL,
            started_at timestamptz NOT NULL,
            last_seen_at timestamptz NOT NULL
        )""")


def downgrade() -> None:
    q = _quote(_ai_schema())
    op.execute(f"DROP TABLE {q}.workers, {q}.jobs, {q}.flags")
