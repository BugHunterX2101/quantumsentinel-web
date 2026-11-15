"""Trial evaluations: the measured outcome of each accepted strategy-version run.

The trial ledger (``research_trials``) counts every distinct configuration a
user evaluated in a research family, so that the Deflated Sharpe test
discounts a result by at least that many trials. A strategy-version backtest
(the ``strategy_backtest`` job) also records what it measured, in
``research_trial_evaluations``: one row per job whose result the worker
accepted, naming the trial it counted as (a composite foreign key to the
ledger's unique key), the strategy version it ran, the datasets it read, and
the Sharpe ratio, number of returns and return-series hash of its result
(services/research_trials.py).

The row is written in the transaction that stores the job's result, after the
worker has accepted the market data the job read. The trial itself is
recorded before, and kept whatever happens to the result, so a refused or
lost result leaves the trial counted and no evaluation. ``job_id`` names the
job without a foreign key: finished jobs are purged, and evaluations stay.

Both tables are append-only, by the mechanisms of migration 0004: no role but
the owner holds UPDATE, DELETE or TRUNCATE (db_roles.py), a row trigger
refuses UPDATE and DELETE and a statement trigger refuses TRUNCATE, for the
owner too. ``research_trials`` gains the triggers here: since its grants
became SELECT and INSERT, it was the one such table without them.

Revision ID: 0007
Revises: 0006
Create Date: 2026-11-15
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: Union[str, Sequence[str], None] = "0006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLES = ("research_trials", "research_trial_evaluations")


def upgrade() -> None:
    op.create_table(
        "research_trial_evaluations",
        sa.Column("job_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("family_hash", sa.String(64), nullable=False),
        sa.Column("config_hash", sa.String(64), nullable=False),
        sa.Column("strategy_version_id", sa.String(80), nullable=False),
        sa.Column("sharpe", sa.Float(), nullable=False),
        sa.Column("n_obs", sa.Integer(), nullable=False),
        sa.Column("returns_sha256", sa.String(96), nullable=False),
        sa.Column("dataset_ids", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("job_id"),
        sa.ForeignKeyConstraint(["user_id", "family_hash", "config_hash"],
                                ["research_trials.user_id", "research_trials.family_hash",
                                 "research_trials.config_hash"],
                                name="fk_research_trial_evaluations_trial"),
        sa.ForeignKeyConstraint(["strategy_version_id"], ["strategy_versions.version_id"]),
        # NaN sorts above Infinity in PostgreSQL, so this also refuses NaN.
        sa.CheckConstraint("sharpe > '-Infinity'::float8 AND sharpe < 'Infinity'::float8",
                           name="ck_research_trial_evaluations_sharpe_finite"),
        sa.CheckConstraint("n_obs >= 2", name="ck_research_trial_evaluations_n_obs"),
        sa.CheckConstraint("returns_sha256 ~ '^qs-returns-v1:[0-9a-f]{64}$'",
                           name="ck_research_trial_evaluations_returns_sha256"),
        sa.CheckConstraint("json_typeof(dataset_ids) = 'array' AND json_array_length(dataset_ids) > 0",
                           name="ck_research_trial_evaluations_dataset_ids"),
    )
    for table in TABLES:
        op.execute(f"CREATE TRIGGER {table}_immutable BEFORE UPDATE OR DELETE ON {table}"
                   " FOR EACH ROW EXECUTE FUNCTION qs_refuse_change()")
        op.execute(f"CREATE TRIGGER {table}_no_truncate BEFORE TRUNCATE ON {table}"
                   " FOR EACH STATEMENT EXECUTE FUNCTION qs_refuse_change()")


def downgrade() -> None:
    op.drop_table("research_trial_evaluations")  # its triggers go with it
    op.execute("DROP TRIGGER research_trials_no_truncate ON research_trials")
    op.execute("DROP TRIGGER research_trials_immutable ON research_trials")
