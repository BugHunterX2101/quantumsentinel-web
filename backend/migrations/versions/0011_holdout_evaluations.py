"""Holdout evaluations: the one-shot grade of a validated strategy on its sealed holdout.

``holdout_attempts`` is the ledger of attempts. The research evaluator writes
a row when it claims a ``final_evaluation`` job, and commits it before it
reads any data, so an attempt is spent even if the run then fails. The unique
key allows one attempt per user, strategy version, dataset family and holdout
start: a version is graded once on a holdout, and again only after
``set-holdout`` moves the start later. The rows of a research family also
count as trials in that family (trial_matrix.family_state), so a strategy
picked after a holdout grade is deflated for it.

``holdout_evaluations`` holds each grade with every figure it rests on, and
nothing else does. The job's result and audit event carry only ids and
pass/fail. The research worker can read neither the figures nor this table
(db_roles.py). Each row stores:

* the holdout policy, by id and canonical JSON (a CHECK holds the id to the
  hash);
* whether it passed (a CHECK holds it to the grade's own ``passed``);
* the grade;
* the grade's signed manifest. ``manifest_id`` is the hash of
  ``canonical``. A CHECK holds the manifest's ``holdout_job_id``,
  ``policy_id``, ``passed``, ``signing_key_id`` and ``manifest_version`` (3)
  to the row, and the signature to the forms validation manifests use
  (migration 0009).

Both tables are append-only, by the mechanisms of migration 0004: no role
but the owner holds UPDATE, DELETE or TRUNCATE (db_roles.py). A row trigger
refuses UPDATE and DELETE, and a statement trigger refuses TRUNCATE, for the
owner too.

Revision ID: 0011
Revises: 0010
Create Date: 2026-11-20
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0011"
down_revision: Union[str, Sequence[str], None] = "0010"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_MANIFEST = "convert_from(canonical, 'UTF8')::json"


def _immutable(table: str) -> None:
    op.execute(f"CREATE TRIGGER {table}_immutable BEFORE UPDATE OR DELETE ON {table}"
               " FOR EACH ROW EXECUTE FUNCTION qs_refuse_change()")
    op.execute(f"CREATE TRIGGER {table}_no_truncate BEFORE TRUNCATE ON {table}"
               " FOR EACH STATEMENT EXECUTE FUNCTION qs_refuse_change()")


def upgrade() -> None:
    op.create_table(
        "holdout_attempts",
        sa.Column("job_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("validation_job_id", sa.String(), nullable=False),
        sa.Column("strategy_version_id", sa.String(80), nullable=False),
        sa.Column("family_hash", sa.String(64), nullable=False),
        sa.Column("dataset_family", sa.String(64), nullable=False),
        sa.Column("holdout_start", sa.Date(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("job_id"),
        sa.UniqueConstraint("user_id", "strategy_version_id", "dataset_family", "holdout_start",
                            name="uq_holdout_attempts_once_per_holdout"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["validation_job_id"], ["strategy_validations.job_id"]),
        sa.ForeignKeyConstraint(["strategy_version_id"], ["strategy_versions.version_id"]),
    )
    op.create_index("ix_holdout_attempts_family", "holdout_attempts", ["user_id", "family_hash"])
    _immutable("holdout_attempts")

    op.create_table(
        "holdout_evaluations",
        sa.Column("job_id", sa.String(), nullable=False),
        sa.Column("policy_id", sa.String(96), nullable=False),
        sa.Column("policy_canonical", sa.LargeBinary(), nullable=False),
        sa.Column("passed", sa.Boolean(), nullable=False),
        sa.Column("grade", sa.JSON(), nullable=False),
        sa.Column("manifest_id", sa.String(96), nullable=False),
        sa.Column("canonical", sa.LargeBinary(), nullable=False),
        sa.Column("signature", sa.Text(), nullable=False),
        sa.Column("signing_key_id", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("job_id"),
        sa.UniqueConstraint("manifest_id"),
        sa.ForeignKeyConstraint(["job_id"], ["holdout_attempts.job_id"]),
        sa.ForeignKeyConstraint(["signing_key_id"], ["server_signing_keys.key_id"]),
        sa.CheckConstraint("policy_id = 'qs-hp-sha256-v1:' || encode(sha256(policy_canonical), 'hex')",
                           name="ck_holdout_evaluations_policy_id_is_hash"),
        sa.CheckConstraint("json_typeof(grade) = 'object' AND (grade->>'passed')::boolean = passed",
                           name="ck_holdout_evaluations_passed_is_grade"),
        sa.CheckConstraint("manifest_id = 'qs-hm-sha256-v1:' || encode(sha256(canonical), 'hex')",
                           name="ck_holdout_evaluations_manifest_id_is_hash"),
        sa.CheckConstraint(f"json_typeof({_MANIFEST}) = 'object'"
                           f" AND {_MANIFEST}->>'manifest_version' = '3'"
                           f" AND {_MANIFEST}->>'holdout_job_id' = job_id"
                           f" AND {_MANIFEST}->>'policy_id' = policy_id"
                           f" AND ({_MANIFEST}->>'passed')::boolean = passed"
                           f" AND ({_MANIFEST}->>'signing_key_id') IS NOT DISTINCT FROM signing_key_id",
                           name="ck_holdout_evaluations_manifest_names_its_row"),
        sa.CheckConstraint("(signature ~ '^ml-dsa:([0-9a-f]{2})+$' AND signing_key_id IS NOT NULL)"
                           " OR (signature ~ '^hmac-sha256:[0-9a-f]{64}$' AND signing_key_id IS NULL)",
                           name="ck_holdout_evaluations_signature_form"),
    )
    _immutable("holdout_evaluations")


def downgrade() -> None:
    op.drop_table("holdout_evaluations")  # its triggers go with it
    op.drop_table("holdout_attempts")
