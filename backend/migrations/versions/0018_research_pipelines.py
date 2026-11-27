"""Research pipelines: bounded runs of research jobs, and idempotent enqueue.

``research_pipelines``: a pipeline may queue at most ``max_jobs`` jobs.
``jobs_queued`` counts every job ever queued in it and is never lowered, so
a job that failed, was cancelled or was purged still spent its share. CHECKs
hold the counter within the budget and ``closed_at`` to the status (a closed
pipeline queues nothing more).

``research_pipeline_days``: the jobs one user queued in pipelines on one UTC
day, for the daily budget (RESEARCH_PIPELINE_DAILY_JOBS). Also never
lowered.

``research_jobs`` gains:

* ``pipeline_id``: the pipeline whose budget paid for the job;
* ``stage``: the step of that pipeline the job runs;
* ``idempotency_key``: unique per user while the job is kept, so a retried
  request gets the job the first one queued instead of a second job;
* ``request_hash``: the sha256 of the request queued under that key, which
  a retry must match (set exactly when the key is).

Revision ID: 0018
Revises: 0017
Create Date: 2026-11-27
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0018"
down_revision: Union[str, Sequence[str], None] = "0017"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "research_pipelines",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("max_jobs", sa.Integer(), nullable=False),
        sa.Column("jobs_queued", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.CheckConstraint("max_jobs >= 1", name="ck_research_pipelines_budget"),
        sa.CheckConstraint("jobs_queued >= 0 AND jobs_queued <= max_jobs",
                           name="ck_research_pipelines_within_budget"),
        sa.CheckConstraint("(status = 'open' AND closed_at IS NULL) OR (status = 'closed' AND closed_at IS NOT NULL)",
                           name="ck_research_pipelines_status"),
        sa.CheckConstraint("length(name) >= 1", name="ck_research_pipelines_name"),
    )
    op.create_index("ix_research_pipelines_user_created", "research_pipelines", ["user_id", "created_at"])

    op.create_table(
        "research_pipeline_days",
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column("jobs_queued", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("user_id", "day"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.CheckConstraint("jobs_queued >= 0", name="ck_research_pipeline_days_count"),
    )

    op.add_column("research_jobs", sa.Column("pipeline_id", sa.String(), nullable=True))
    op.add_column("research_jobs", sa.Column("stage", sa.String(64), nullable=True))
    op.add_column("research_jobs", sa.Column("idempotency_key", sa.String(128), nullable=True))
    op.add_column("research_jobs", sa.Column("request_hash", sa.String(64), nullable=True))
    op.create_foreign_key("research_jobs_pipeline_id_fkey", "research_jobs", "research_pipelines",
                          ["pipeline_id"], ["id"])
    op.create_index("ix_research_jobs_pipeline", "research_jobs", ["pipeline_id"])
    op.create_index("uq_research_jobs_idempotency", "research_jobs", ["user_id", "idempotency_key"],
                    unique=True, postgresql_where=sa.text("idempotency_key IS NOT NULL"))
    # A stage names a step of a pipeline: a job outside one has none.
    op.create_check_constraint("ck_research_jobs_stage_in_pipeline", "research_jobs",
                               "stage IS NULL OR pipeline_id IS NOT NULL")
    op.create_check_constraint("ck_research_jobs_idempotency_hash", "research_jobs",
                               "(idempotency_key IS NULL) = (request_hash IS NULL)")


def downgrade() -> None:
    op.drop_constraint("ck_research_jobs_idempotency_hash", "research_jobs", type_="check")
    op.drop_constraint("ck_research_jobs_stage_in_pipeline", "research_jobs", type_="check")
    op.drop_index("uq_research_jobs_idempotency", table_name="research_jobs")
    op.drop_index("ix_research_jobs_pipeline", table_name="research_jobs")
    op.drop_constraint("research_jobs_pipeline_id_fkey", "research_jobs", type_="foreignkey")
    op.drop_column("research_jobs", "request_hash")
    op.drop_column("research_jobs", "idempotency_key")
    op.drop_column("research_jobs", "stage")
    op.drop_column("research_jobs", "pipeline_id")
    op.drop_table("research_pipeline_days")
    op.drop_index("ix_research_pipelines_user_created", table_name="research_pipelines")
    op.drop_table("research_pipelines")
