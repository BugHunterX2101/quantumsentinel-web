"""The AI tier's document registry: one row per harvested paper.

``documents`` lives in the AI schema beside ``jobs`` (migration 0012). The AI
worker writes it (backend/ai/documents.py) after checking every field of a
harvested page; the application only reads it. A paper is keyed by
(source, source_id). arXiv's ``arXiv`` metadata format has no version, so a
row holds the latest metadata of its paper, and ``content_hash``
(``qs-doc-sha256-v1``) tells a changed record from a repeated one. A paper
arXiv reports deleted keeps its row, with ``deleted_at`` set. No role may
delete a row.

``reuse_class`` records what the paper's licence (``license_url``) lets the
product do with its full text: open, noncommercial or restricted, where
anything unrecognised is restricted.

The checks repeat the worker's own limits, so a row that bypassed it still
cannot hold an unbounded or malformed value.

Revision ID: 0013
Revises: 0012
Create Date: 2026-11-20
"""
from typing import Sequence, Union

from alembic import op

revision: str = "0013"
down_revision: Union[str, Sequence[str], None] = "0012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _ai() -> str:
    from backend.db_roles import ai_schema_for

    name = ai_schema_for(op.get_bind().exec_driver_sql("SELECT current_schema()").scalar())
    return '"' + name.replace('"', '""') + '"'


def upgrade() -> None:
    q = _ai()
    op.execute(f"""
        CREATE TABLE {q}.documents (
            id varchar(36) PRIMARY KEY DEFAULT gen_random_uuid()::text,
            source varchar(32) NOT NULL CHECK (source IN ('arxiv')),
            source_id varchar(64) NOT NULL
                CHECK (source_id ~ '^([0-9]{{4}}\\.[0-9]{{4,5}}|[a-z]+(-[a-z]+)*(\\.[A-Z]{{2}})?/[0-9]{{7}})$'),
            datestamp date NOT NULL,
            set_specs text[] NOT NULL DEFAULT '{{}}' CHECK (cardinality(set_specs) <= 20),
            title text NOT NULL CHECK (length(title) BETWEEN 1 AND 1000),
            authors jsonb NOT NULL DEFAULT '[]'::jsonb
                CHECK (jsonb_typeof(authors) = 'array' AND jsonb_array_length(authors) <= 100),
            author_count integer NOT NULL CHECK (author_count >= 0),
            abstract text NOT NULL CHECK (length(abstract) <= 8000),
            categories text[] NOT NULL DEFAULT '{{}}' CHECK (cardinality(categories) <= 20),
            primary_category varchar(64),
            license_url varchar(255),
            reuse_class varchar(16) NOT NULL CHECK (reuse_class IN ('open', 'noncommercial', 'restricted')),
            doi varchar(255),
            journal_ref varchar(500),
            created date NOT NULL,
            updated date,
            content_hash varchar(81) NOT NULL CHECK (content_hash ~ '^qs-doc-sha256-v1:[0-9a-f]{{64}}$'),
            first_seen_at timestamptz NOT NULL DEFAULT now(),
            changed_at timestamptz NOT NULL DEFAULT now(),
            deleted_at timestamptz,
            CONSTRAINT documents_source_paper UNIQUE (source, source_id)
        )""")
    op.execute(f"CREATE INDEX documents_datestamp ON {q}.documents (source, datestamp)")


def downgrade() -> None:
    op.execute(f"DROP TABLE {_ai()}.documents")
