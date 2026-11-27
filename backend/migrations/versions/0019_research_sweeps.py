"""Research sweeps: a strategy expanded over a parameter grid in a pipeline.

``research_sweeps``: one sweep, append-only (UPDATE, DELETE and TRUNCATE are
refused by trigger). It records the strategy version it started from, the
grid, the period, capital and frozen datasets every candidate runs on, how
many grid points it generated and how many of them were refused as specs
(``n_rejected``), and the MinBTL cap it was checked against. A refused
candidate counts toward its family's trials (trial_matrix.family_state), so
``n_rejected`` can never be lowered. ``idempotency_key`` is unique per user,
so a retried request gets the sweep the first one created instead of paying
for a second.

``research_sweep_candidates``: one row per grid point. A valid point names
the strategy version it registered and the trial config it will run as; a
refused point names why it was refused instead. ``job_id`` and ``queued_at``
are set once, when the candidate's backtest is queued; a trigger refuses
every other change, and every DELETE and TRUNCATE.

Revision ID: 0019
Revises: 0018
Create Date: 2026-11-27
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0019"
down_revision: Union[str, Sequence[str], None] = "0018"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_QUEUE_ONCE = """
CREATE OR REPLACE FUNCTION qs_sweep_candidate_queue_once() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP = 'UPDATE'
       AND OLD.job_id IS NULL AND NEW.job_id IS NOT NULL
       AND NEW.sweep_id = OLD.sweep_id AND NEW.ordinal = OLD.ordinal
       AND NEW.params::text = OLD.params::text
       AND NEW.strategy_version_id IS NOT DISTINCT FROM OLD.strategy_version_id
       AND NEW.config_hash IS NOT DISTINCT FROM OLD.config_hash
       AND NEW.rejection IS NOT DISTINCT FROM OLD.rejection THEN
        RETURN NEW;
    END IF;
    RAISE EXCEPTION 'a sweep candidate is only ever queued once; it is not otherwise changed or removed'
        USING ERRCODE = 'restrict_violation';
END;
$$
"""


def upgrade() -> None:
    op.create_table(
        "research_sweeps",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("pipeline_id", sa.String(), nullable=False),
        sa.Column("base_version_id", sa.String(80), nullable=False),
        sa.Column("family_hash", sa.String(64), nullable=False),
        sa.Column("grid", sa.JSON(), nullable=False),
        sa.Column("period", sa.String(8), nullable=False),
        sa.Column("initial_capital", sa.Float(), nullable=False),
        sa.Column("dataset_ids", sa.JSON(), nullable=False),
        sa.Column("n_candidates", sa.Integer(), nullable=False),
        sa.Column("n_rejected", sa.Integer(), nullable=False),
        sa.Column("trials_after", sa.Integer(), nullable=False),
        sa.Column("minbtl_cap", sa.Integer(), nullable=False),
        sa.Column("idempotency_key", sa.String(128), nullable=True),
        sa.Column("request_hash", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["pipeline_id"], ["research_pipelines.id"]),
        sa.ForeignKeyConstraint(["base_version_id"], ["strategy_versions.version_id"]),
        sa.CheckConstraint("n_candidates >= 1 AND n_rejected >= 0 AND n_rejected <= n_candidates",
                           name="ck_research_sweeps_counts"),
        sa.CheckConstraint("minbtl_cap >= 1 AND trials_after <= minbtl_cap", name="ck_research_sweeps_minbtl"),
        sa.CheckConstraint("json_typeof(dataset_ids) = 'array' AND json_array_length(dataset_ids) > 0",
                           name="ck_research_sweeps_dataset_ids"),
        sa.CheckConstraint("(idempotency_key IS NULL) = (request_hash IS NULL)",
                           name="ck_research_sweeps_idempotency_hash"),
    )
    op.create_index("ix_research_sweeps_user_family", "research_sweeps", ["user_id", "family_hash"])
    op.create_index("ix_research_sweeps_pipeline", "research_sweeps", ["pipeline_id"])
    op.create_index("uq_research_sweeps_idempotency", "research_sweeps", ["user_id", "idempotency_key"],
                    unique=True, postgresql_where=sa.text("idempotency_key IS NOT NULL"))

    op.create_table(
        "research_sweep_candidates",
        sa.Column("sweep_id", sa.String(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("params", sa.JSON(), nullable=False),
        sa.Column("strategy_version_id", sa.String(80), nullable=True),
        sa.Column("config_hash", sa.String(64), nullable=True),
        sa.Column("rejection", sa.String(500), nullable=True),
        sa.Column("job_id", sa.String(), nullable=True),
        sa.Column("queued_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("sweep_id", "ordinal"),
        sa.ForeignKeyConstraint(["sweep_id"], ["research_sweeps.id"]),
        sa.ForeignKeyConstraint(["strategy_version_id"], ["strategy_versions.version_id"]),
        sa.CheckConstraint("ordinal >= 0", name="ck_research_sweep_candidates_ordinal"),
        # A candidate is either a registered version with its trial config,
        # or a refused spec with the reason; only a version is ever queued.
        sa.CheckConstraint("(strategy_version_id IS NULL) = (rejection IS NOT NULL)",
                           name="ck_research_sweep_candidates_outcome"),
        sa.CheckConstraint("(strategy_version_id IS NULL) = (config_hash IS NULL)",
                           name="ck_research_sweep_candidates_config"),
        sa.CheckConstraint("job_id IS NULL OR strategy_version_id IS NOT NULL",
                           name="ck_research_sweep_candidates_queued_version"),
        sa.CheckConstraint("(job_id IS NULL) = (queued_at IS NULL)", name="ck_research_sweep_candidates_queued_at"),
    )

    op.execute("CREATE TRIGGER research_sweeps_immutable BEFORE UPDATE OR DELETE ON research_sweeps"
               " FOR EACH ROW EXECUTE FUNCTION qs_refuse_change()")
    op.execute("CREATE TRIGGER research_sweeps_no_truncate BEFORE TRUNCATE ON research_sweeps"
               " FOR EACH STATEMENT EXECUTE FUNCTION qs_refuse_change()")
    op.execute(_QUEUE_ONCE)
    op.execute("CREATE TRIGGER research_sweep_candidates_queue_once BEFORE UPDATE OR DELETE"
               " ON research_sweep_candidates FOR EACH ROW EXECUTE FUNCTION qs_sweep_candidate_queue_once()")
    op.execute("CREATE TRIGGER research_sweep_candidates_no_truncate BEFORE TRUNCATE"
               " ON research_sweep_candidates FOR EACH STATEMENT EXECUTE FUNCTION qs_refuse_change()")


def downgrade() -> None:
    op.drop_table("research_sweep_candidates")  # its triggers go with it
    op.execute("DROP FUNCTION qs_sweep_candidate_queue_once()")
    op.drop_index("uq_research_sweeps_idempotency", table_name="research_sweeps")
    op.drop_index("ix_research_sweeps_pipeline", table_name="research_sweeps")
    op.drop_index("ix_research_sweeps_user_family", table_name="research_sweeps")
    op.drop_table("research_sweeps")
