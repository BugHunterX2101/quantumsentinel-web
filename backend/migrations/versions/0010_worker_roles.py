"""Research worker roles: which process role each live worker runs as.

A worker claims only the job kinds its QS_PROCESS_ROLE allows
(research_jobs.claimable_kinds), so a job's "workers online" must count only
the workers that can claim it: a research-evaluator worker can run no
ordinary job, and no other worker can run a holdout evaluation.

The column defaults to 'research-worker' on the server, so the rows of
workers that were running before this revision (none of which was an
evaluator) keep their meaning.

Revision ID: 0010
Revises: 0009
Create Date: 2026-11-20
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0010"
down_revision: Union[str, Sequence[str], None] = "0009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("research_workers", sa.Column("role", sa.String(32), nullable=False,
                                                server_default="research-worker"))


def downgrade() -> None:
    op.drop_column("research_workers", "role")
