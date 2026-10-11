"""Dataset registry: frozen dataset versions and who holds them.

``dataset_versions`` describes one content-addressed dataset blob
(services/dataset_format.py, services/dataset_store.py): its identity, licence
flags, coverage and the hashes that name and verify it. The blob itself lives
on the dataset volume, not here: a bytea value is capped at 1 GB and is fully
WAL-logged.

Rows are immutable. Three mechanisms hold that, as planned for every
hash-identified table:

* no role but the owner holds UPDATE, DELETE or TRUNCATE (db_roles.py);
* a row trigger refuses UPDATE and DELETE, for the owner too;
* a statement trigger refuses TRUNCATE, which row triggers never see.

Two CHECKs tie each row to its id: ``dataset_id`` must be the SHA-256 of the
stored canonical manifest, and the identity columns must equal the identity
inside that manifest. A writer that computes the id wrongly, or describes a
dataset as something it is not, fails the insert.

``user_datasets`` records which user snapshotted which dataset. A dataset
under a personal-use licence is listed only to the users who fetched it.

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-11
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: Union[str, Sequence[str], None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

IDENTITY = ("family", "provider", "licence_id", "frequency", "session_boundary", "adjustment",
            "survivorship")
ID_MATCHES_MANIFEST = ("dataset_id = 'qs-ds-sha256-v1:' || encode(sha256(manifest_canonical), 'hex')")
IDENTITY_MATCHES_MANIFEST = (
    "(convert_from(manifest_canonical, 'UTF8')::jsonb -> 'identity') = jsonb_build_object("
    + ", ".join(f"'{name}', {name}" for name in IDENTITY) + ")")


def upgrade() -> None:
    op.execute("""
        CREATE OR REPLACE FUNCTION qs_refuse_change() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION '% on % is refused: its rows are immutable', TG_OP, TG_TABLE_NAME
                USING ERRCODE = 'restrict_violation';
        END
        $$
    """)
    op.create_table(
        "dataset_versions",
        sa.Column("dataset_id", sa.String(80), nullable=False),
        *(sa.Column(name, sa.String(64), nullable=False) for name in IDENTITY),
        sa.Column("personal_use_only", sa.Boolean(), nullable=False),
        sa.Column("redistribution_allowed", sa.Boolean(), nullable=False),
        sa.Column("point_in_time", sa.Boolean(), nullable=False),
        sa.Column("coverage_start", sa.Date(), nullable=False),
        sa.Column("coverage_end", sa.Date(), nullable=False),
        sa.Column("symbols", sa.JSON(), nullable=False),
        sa.Column("manifest_canonical", sa.LargeBinary(), nullable=False),
        sa.Column("tables_json", sa.JSON(), nullable=False),
        sa.Column("blob_format", sa.String(32), nullable=False),
        sa.Column("blob_sha256", sa.String(64), nullable=False),
        sa.Column("blob_size", sa.BigInteger(), nullable=False),
        sa.Column("provider_client", sa.String(64), nullable=False),
        sa.Column("request_json", sa.JSON(), nullable=False),
        sa.Column("retrieved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("dataset_id"),
        sa.CheckConstraint(ID_MATCHES_MANIFEST, name="ck_dataset_versions_id_is_manifest_hash"),
        sa.CheckConstraint(IDENTITY_MATCHES_MANIFEST, name="ck_dataset_versions_identity_is_manifest"),
        sa.CheckConstraint("blob_sha256 ~ '^[0-9a-f]{64}$'", name="ck_dataset_versions_blob_sha256"),
        sa.CheckConstraint("blob_size > 0", name="ck_dataset_versions_blob_size"),
        sa.CheckConstraint("coverage_start <= coverage_end", name="ck_dataset_versions_coverage"),
    )
    op.execute("CREATE TRIGGER dataset_versions_immutable BEFORE UPDATE OR DELETE ON dataset_versions"
               " FOR EACH ROW EXECUTE FUNCTION qs_refuse_change()")
    op.execute("CREATE TRIGGER dataset_versions_no_truncate BEFORE TRUNCATE ON dataset_versions"
               " FOR EACH STATEMENT EXECUTE FUNCTION qs_refuse_change()")
    op.create_table(
        "user_datasets",
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("dataset_id", sa.String(80), nullable=False),
        sa.Column("job_id", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["dataset_id"], ["dataset_versions.dataset_id"]),
        sa.PrimaryKeyConstraint("user_id", "dataset_id"),
    )


def downgrade() -> None:
    op.drop_table("user_datasets")
    op.drop_table("dataset_versions")  # its triggers go with it
    op.execute("DROP FUNCTION qs_refuse_change()")
