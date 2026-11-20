"""What OpenAlex and Crossref say about the registry's documents.

* ``sources`` takes two more services, each with its own rate slot: OpenAlex
  and Crossref, at most one request a second each (backend/ai/enrich.py).
* ``document_metadata``: one row per (document, service), written only by
  the AI worker's metadata sync. ``status`` is ``found`` (the service knows
  the work), ``not_found`` (it answered 404 or 410) or ``refused`` (there
  was nothing to ask, or its answer did not pass the checks; ``reason``
  says which). ``lookup_doi`` is the DOI asked about. ``cited_by_count``
  and ``is_retracted`` are the service's own numbers, and ``fields`` keeps
  the few other fields read from its answer, bounded. ``checked_at`` is the
  last lookup; ``changed_at`` the last one whose answer differed.
* ``jobs_one_metadata_sync``: at most one metadata sync per service is
  queued or running.

The application reads the table; neither role may delete a row.

Revision ID: 0015
Revises: 0014
Create Date: 2026-11-20
"""
from typing import Sequence, Union

from alembic import op

revision: str = "0015"
down_revision: Union[str, Sequence[str], None] = "0014"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _ai() -> str:
    from backend.db_roles import ai_schema_for

    name = ai_schema_for(op.get_bind().exec_driver_sql("SELECT current_schema()").scalar())
    return '"' + name.replace('"', '""') + '"'


def upgrade() -> None:
    q = _ai()
    op.execute(f"ALTER TABLE {q}.sources DROP CONSTRAINT sources_name_check")
    op.execute(f"ALTER TABLE {q}.sources ADD CONSTRAINT sources_name_check"
               " CHECK (name IN ('arxiv', 'openalex', 'crossref'))")
    op.execute(f"INSERT INTO {q}.sources (name, min_interval_seconds) VALUES ('openalex', 1), ('crossref', 1)")
    op.execute(f"""
        CREATE TABLE {q}.document_metadata (
            document_id varchar(36) NOT NULL REFERENCES {q}.documents (id),
            source varchar(32) NOT NULL REFERENCES {q}.sources (name)
                CHECK (source IN ('openalex', 'crossref')),
            status varchar(16) NOT NULL CHECK (status IN ('found', 'not_found', 'refused')),
            lookup_doi varchar(255),
            reason varchar(500),
            external_id varchar(255),
            cited_by_count bigint CHECK (cited_by_count >= 0),
            is_retracted boolean,
            fields jsonb NOT NULL DEFAULT '{{}}'::jsonb
                CHECK (jsonb_typeof(fields) = 'object' AND octet_length(fields::text) <= 16384),
            checks bigint NOT NULL DEFAULT 1 CHECK (checks >= 1),
            checked_at timestamptz NOT NULL DEFAULT now(),
            changed_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (document_id, source),
            CHECK (status = 'found' OR (external_id IS NULL AND cited_by_count IS NULL
                                        AND is_retracted IS NULL AND fields = '{{}}'::jsonb)),
            CHECK ((status = 'refused') = (reason IS NOT NULL)),
            CHECK (status = 'refused' OR lookup_doi IS NOT NULL)
        )""")
    op.execute(f"CREATE INDEX document_metadata_checked ON {q}.document_metadata (source, status, checked_at)")
    op.execute(f"""CREATE UNIQUE INDEX jobs_one_metadata_sync ON {q}.jobs ((payload ->> 'source'))
                   WHERE kind = 'metadata_sync' AND status IN ('queued', 'running')""")


def downgrade() -> None:
    q = _ai()
    op.execute(f"DROP INDEX {q}.jobs_one_metadata_sync")
    op.execute(f"DROP TABLE {q}.document_metadata")
    op.execute(f"DELETE FROM {q}.sources WHERE name IN ('openalex', 'crossref')")
    op.execute(f"ALTER TABLE {q}.sources DROP CONSTRAINT sources_name_check")
    op.execute(f"ALTER TABLE {q}.sources ADD CONSTRAINT sources_name_check CHECK (name IN ('arxiv'))")
