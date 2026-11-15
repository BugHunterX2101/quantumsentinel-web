"""Strategy versions: immutable, content-addressed strategy specs and who holds them.

``strategy_versions`` stores one strategy spec (services/strategy_spec.py) as
its canonical JSON bytes, under its id ``qs-sv-sha256-v1:<hex>``. The id is
the spec's content, so two users who write the same spec share one row;
``user_strategy_versions`` records each user who registered it, as
``user_datasets`` does for datasets.

``user_strategy_tags`` gives a user's own names, such as ``1.2.0``, to the
versions they hold. A tag is an alias for reading, never an identity: it
belongs to one user, means nothing to any gate, and once given it always
names the same version (a changed strategy is a new version with a new tag).

All three tables are append-only, by the mechanisms of migration 0004:

* no role but the owner holds UPDATE, DELETE or TRUNCATE (db_roles.py);
* a row trigger refuses UPDATE and DELETE, for the owner too;
* a statement trigger refuses TRUNCATE, which row triggers never see.

CHECKs tie each version to its id: ``version_id`` must be the SHA-256 of the
stored canonical spec, and the columns kept for querying (``dsl``,
``template``, ``family``) must equal those inside it. A tag can only name a
version its user holds (a composite foreign key to the link table).

A lifecycle (validated, holdout evaluated, retired) is left to the revisions
that add what produces those states (R3's validation gate, R1-5b's
evaluator): a state that nothing can earn yet would only be a state the
application could write about itself.

Revision ID: 0006
Revises: 0005
Create Date: 2026-11-15
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: Union[str, Sequence[str], None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLES = ("strategy_versions", "user_strategy_versions", "user_strategy_tags")
ID_IS_SPEC_HASH = "version_id = 'qs-sv-sha256-v1:' || encode(sha256(spec_canonical), 'hex')"
_SPEC = "convert_from(spec_canonical, 'UTF8')::jsonb"
COLUMNS_ARE_SPEC = (f"dsl = {_SPEC} ->> 'dsl' AND template = {_SPEC} -> 'signal' ->> 'template'"
                    f" AND family = {_SPEC} -> 'universe' ->> 'family'")


def upgrade() -> None:
    op.create_table(
        "strategy_versions",
        sa.Column("version_id", sa.String(80), nullable=False),
        sa.Column("dsl", sa.String(32), nullable=False),
        sa.Column("template", sa.String(32), nullable=False),
        sa.Column("family", sa.String(64), nullable=False),
        sa.Column("spec_canonical", sa.LargeBinary(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("version_id"),
        sa.CheckConstraint(ID_IS_SPEC_HASH, name="ck_strategy_versions_id_is_spec_hash"),
        sa.CheckConstraint(COLUMNS_ARE_SPEC, name="ck_strategy_versions_columns_are_spec"),
    )
    op.create_table(
        "user_strategy_versions",
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("version_id", sa.String(80), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["version_id"], ["strategy_versions.version_id"]),
        sa.PrimaryKeyConstraint("user_id", "version_id"),
    )
    op.create_table(
        "user_strategy_tags",
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("label", sa.String(64), nullable=False),
        sa.Column("version_id", sa.String(80), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id", "version_id"],
                                ["user_strategy_versions.user_id", "user_strategy_versions.version_id"],
                                name="fk_user_strategy_tags_held_version"),
        sa.PrimaryKeyConstraint("user_id", "label"),
        sa.CheckConstraint("label ~ '^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$'", name="ck_user_strategy_tags_label"),
    )
    for table in TABLES:
        op.execute(f"CREATE TRIGGER {table}_immutable BEFORE UPDATE OR DELETE ON {table}"
                   " FOR EACH ROW EXECUTE FUNCTION qs_refuse_change()")
        op.execute(f"CREATE TRIGGER {table}_no_truncate BEFORE TRUNCATE ON {table}"
                   " FOR EACH STATEMENT EXECUTE FUNCTION qs_refuse_change()")


def downgrade() -> None:
    # Their triggers go with them; qs_refuse_change() belongs to 0004.
    for table in reversed(TABLES):
        op.drop_table(table)
