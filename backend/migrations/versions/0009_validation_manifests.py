"""Validation manifests: the signed manifest (v3) of each strategy validation's verdict.

The worker builds and signs a manifest of every verdict it records
(services/validation_manifest.py), in the same transaction, and stores it
here once:

* ``manifest_id`` is the hash of ``canonical``, the manifest's canonical
  JSON (a CHECK holds it), so a stored manifest cannot differ from its id.
* ``job_id`` names the verdict (a foreign key to strategy_validations, and
  unique: one manifest per verdict); a CHECK holds the manifest's own
  ``validation_job_id`` and ``signing_key_id`` to the row's, and its
  ``manifest_version`` to 3.
* ``signature`` signs ``canonical``: ``ml-dsa:<hex>`` with the key
  ``signing_key_id`` names (a foreign key to server_signing_keys, so the key
  stays resolvable), or, outside production only, ``hmac-sha256:<hex>`` with
  no key. A CHECK holds the signature to one of those forms.

Verdicts recorded before this revision have no manifest; none is made.

Append-only, by the mechanisms of migration 0004: no role but the owner holds
UPDATE, DELETE or TRUNCATE (db_roles.py), a row trigger refuses UPDATE and
DELETE and a statement trigger refuses TRUNCATE, for the owner too.

Revision ID: 0009
Revises: 0008
Create Date: 2026-11-19
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0009"
down_revision: Union[str, Sequence[str], None] = "0008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_MANIFEST = "convert_from(canonical, 'UTF8')::json"


def upgrade() -> None:
    op.create_table(
        "validation_manifests",
        sa.Column("manifest_id", sa.String(96), nullable=False),
        sa.Column("job_id", sa.String(), nullable=False),
        sa.Column("canonical", sa.LargeBinary(), nullable=False),
        sa.Column("signature", sa.Text(), nullable=False),
        sa.Column("signing_key_id", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("manifest_id"),
        sa.UniqueConstraint("job_id"),
        sa.ForeignKeyConstraint(["job_id"], ["strategy_validations.job_id"]),
        sa.ForeignKeyConstraint(["signing_key_id"], ["server_signing_keys.key_id"]),
        sa.CheckConstraint("manifest_id = 'qs-vm-sha256-v1:' || encode(sha256(canonical), 'hex')",
                           name="ck_validation_manifests_id_is_hash"),
        sa.CheckConstraint(f"json_typeof({_MANIFEST}) = 'object'"
                           f" AND {_MANIFEST}->>'manifest_version' = '3'"
                           f" AND {_MANIFEST}->>'validation_job_id' = job_id"
                           f" AND ({_MANIFEST}->>'signing_key_id') IS NOT DISTINCT FROM signing_key_id",
                           name="ck_validation_manifests_names_its_row"),
        sa.CheckConstraint("(signature ~ '^ml-dsa:([0-9a-f]{2})+$' AND signing_key_id IS NOT NULL)"
                           " OR (signature ~ '^hmac-sha256:[0-9a-f]{64}$' AND signing_key_id IS NULL)",
                           name="ck_validation_manifests_signature_form"),
    )
    op.execute("CREATE TRIGGER validation_manifests_immutable BEFORE UPDATE OR DELETE ON validation_manifests"
               " FOR EACH ROW EXECUTE FUNCTION qs_refuse_change()")
    op.execute("CREATE TRIGGER validation_manifests_no_truncate BEFORE TRUNCATE ON validation_manifests"
               " FOR EACH STATEMENT EXECUTE FUNCTION qs_refuse_change()")


def downgrade() -> None:
    op.drop_table("validation_manifests")  # its triggers go with it
