"""Strategy validations: the validation policy's verdicts and the trial matrices they judged.

A strategy validation (the ``strategy_validation`` job) re-runs every
strategy version evaluated in a candidate's research family on the
candidate's datasets (the trial matrix, services/trial_matrix.py) and judges
the candidate against the validation policy (services/validation_policy.py).
The worker computes the verdict itself, from the matrix it accepted, and
records it here, in the transaction that stores the job's result:

* ``trial_matrices``: each matrix once, content-addressed. ``canonical`` is
  the canonical JSON its id hashes (scheme, dates, columns), and a CHECK
  holds the id to that hash, so a stored matrix cannot differ from the one a
  verdict names.
* ``strategy_validations``: one verdict per job. It names the evaluation it
  judged (a foreign key: evaluations are never deleted), the strategy
  version, the matrix, and the policy, with the policy's canonical JSON,
  which a CHECK holds to the policy id. The verdict (every gate's figures)
  is JSON; CHECKs hold its ``passed``, ``policy`` and ``candidate`` to the
  row's columns. ``job_id`` names the job without a foreign key: finished
  jobs are purged, and verdicts stay.

Both tables are append-only, by the mechanisms of migration 0004: no role
but the owner holds UPDATE, DELETE or TRUNCATE (db_roles.py), a row trigger
refuses UPDATE and DELETE and a statement trigger refuses TRUNCATE, for the
owner too. A later policy never rewrites an earlier verdict: it is recorded
under its own id.

Revision ID: 0008
Revises: 0007
Create Date: 2026-11-15
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0008"
down_revision: Union[str, Sequence[str], None] = "0007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLES = ("trial_matrices", "strategy_validations")


def upgrade() -> None:
    op.create_table(
        "trial_matrices",
        sa.Column("matrix_id", sa.String(96), nullable=False),
        sa.Column("canonical", sa.LargeBinary(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("matrix_id"),
        sa.CheckConstraint("matrix_id = 'qs-tm-sha256-v1:' || encode(sha256(canonical), 'hex')",
                           name="ck_trial_matrices_id_is_hash"),
    )
    op.create_table(
        "strategy_validations",
        sa.Column("job_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("evaluation_job_id", sa.String(), nullable=False),
        sa.Column("strategy_version_id", sa.String(80), nullable=False),
        sa.Column("family_hash", sa.String(64), nullable=False),
        sa.Column("matrix_id", sa.String(96), nullable=False),
        sa.Column("policy_id", sa.String(96), nullable=False),
        sa.Column("policy_canonical", sa.LargeBinary(), nullable=False),
        sa.Column("passed", sa.Boolean(), nullable=False),
        sa.Column("verdict", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("job_id"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["evaluation_job_id"], ["research_trial_evaluations.job_id"]),
        sa.ForeignKeyConstraint(["strategy_version_id"], ["strategy_versions.version_id"]),
        sa.ForeignKeyConstraint(["matrix_id"], ["trial_matrices.matrix_id"]),
        sa.CheckConstraint("policy_id = 'qs-vp-sha256-v1:' || encode(sha256(policy_canonical), 'hex')",
                           name="ck_strategy_validations_policy_id_is_hash"),
        sa.CheckConstraint("json_typeof(verdict) = 'object'"
                           " AND verdict->>'passed' = CASE WHEN passed THEN 'true' ELSE 'false' END"
                           " AND verdict->>'policy' = policy_id"
                           " AND verdict->>'candidate' = strategy_version_id",
                           name="ck_strategy_validations_verdict_matches"),
    )
    op.create_index("ix_strategy_validations_user_id", "strategy_validations", ["user_id"])
    for table in TABLES:
        op.execute(f"CREATE TRIGGER {table}_immutable BEFORE UPDATE OR DELETE ON {table}"
                   " FOR EACH ROW EXECUTE FUNCTION qs_refuse_change()")
        op.execute(f"CREATE TRIGGER {table}_no_truncate BEFORE TRUNCATE ON {table}"
                   " FOR EACH STATEMENT EXECUTE FUNCTION qs_refuse_change()")


def downgrade() -> None:
    op.drop_table("strategy_validations")  # its index and triggers go with it
    op.drop_table("trial_matrices")
