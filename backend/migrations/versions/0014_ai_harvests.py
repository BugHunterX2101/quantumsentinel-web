"""The AI tier's harvest bookkeeping: a shared rate slot and per-set cursors.

* ``sources``: one row per remote service the AI tier harvests from. Before
  each request the AI worker takes the next slot, ``next_allowed_at``, and
  moves it on by ``min_interval_seconds`` (backend/ai/harvest.py), so every
  worker shares one rate limit. arXiv asks for at most one request every
  three seconds; a ``Retry-After`` from it pushes the slot further.
* ``harvests``: one row per (source, set), written only by the AI worker.
  ``last_complete_until`` is the last day of the last date window harvested
  to its end; the next sync starts there (that day again, since OAI-PMH
  datestamps are days). ``window_from``/``window_until`` and the resumption
  token describe the window in progress, so an interrupted sync resumes
  where it stopped while its token is valid.
* ``jobs_one_arxiv_sync``: at most one arXiv sync is queued or running.

The application reads both tables; neither role may delete a row.

Revision ID: 0014
Revises: 0013
Create Date: 2026-11-20
"""
from typing import Sequence, Union

from alembic import op

revision: str = "0014"
down_revision: Union[str, Sequence[str], None] = "0013"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _ai() -> str:
    from backend.db_roles import ai_schema_for

    name = ai_schema_for(op.get_bind().exec_driver_sql("SELECT current_schema()").scalar())
    return '"' + name.replace('"', '""') + '"'


def upgrade() -> None:
    q = _ai()
    op.execute(f"""
        CREATE TABLE {q}.sources (
            name varchar(32) PRIMARY KEY CHECK (name IN ('arxiv')),
            min_interval_seconds numeric(8, 3) NOT NULL CHECK (min_interval_seconds >= 1),
            next_allowed_at timestamptz NOT NULL DEFAULT now()
        )""")
    op.execute(f"INSERT INTO {q}.sources (name, min_interval_seconds) VALUES ('arxiv', 3)")
    op.execute(f"""
        CREATE TABLE {q}.harvests (
            source varchar(32) NOT NULL REFERENCES {q}.sources (name),
            set_spec varchar(128) NOT NULL CHECK (set_spec ~ '^[A-Za-z][A-Za-z0-9.-]*(:[A-Za-z0-9.-]+)*$'),
            start_date date NOT NULL,
            last_complete_until date CHECK (last_complete_until >= start_date),
            window_from date,
            window_until date,
            resumption_token varchar(1024) CHECK (resumption_token ~ '^[\\x21-\\x7e]+$'),
            token_expires timestamptz,
            pages bigint NOT NULL DEFAULT 0 CHECK (pages >= 0),
            records bigint NOT NULL DEFAULT 0 CHECK (records >= 0),
            last_page_at timestamptz,
            last_error text CHECK (length(last_error) <= 2000),
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (source, set_spec),
            CHECK ((window_from IS NULL) = (window_until IS NULL)),
            CHECK (window_from <= window_until),
            CHECK (resumption_token IS NULL OR window_from IS NOT NULL)
        )""")
    op.execute(f"""CREATE UNIQUE INDEX jobs_one_arxiv_sync ON {q}.jobs (kind)
                   WHERE kind = 'arxiv_sync' AND status IN ('queued', 'running')""")


def downgrade() -> None:
    q = _ai()
    op.execute(f"DROP INDEX {q}.jobs_one_arxiv_sync")
    op.execute(f"DROP TABLE {q}.harvests, {q}.sources")
