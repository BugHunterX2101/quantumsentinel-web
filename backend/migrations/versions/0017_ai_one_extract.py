"""At most one hypothesis extraction job at a time.

``jobs_one_extract``: an ``extract`` job (backend/ai/extract.py) asks the
model about every document not yet extracted with the current prompt and
model, newest first. Two at once would ask about the same documents and pay
twice for answers of which only one can be recorded.

Revision ID: 0017
Revises: 0016
Create Date: 2026-11-21
"""
from typing import Sequence, Union

from alembic import op

revision: str = "0017"
down_revision: Union[str, Sequence[str], None] = "0016"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _ai() -> str:
    from backend.db_roles import ai_schema_for

    name = ai_schema_for(op.get_bind().exec_driver_sql("SELECT current_schema()").scalar())
    return '"' + name.replace('"', '""') + '"'


def upgrade() -> None:
    q = _ai()
    op.execute(f"""CREATE UNIQUE INDEX jobs_one_extract ON {q}.jobs (kind)
                   WHERE kind = 'extract' AND status IN ('queued', 'running')""")


def downgrade() -> None:
    op.execute(f"DROP INDEX {_ai()}.jobs_one_extract")
