"""webhooks.secret_hash: VARCHAR(128) -> TEXT.

The column holds the Fernet ciphertext of the webhook signing secret, which
is 140 characters for the 32-byte secret the API generates. SQLite never
enforced the declared length; PostgreSQL does, so creating any webhook failed
with "value too long for type character varying(128)". VARCHAR -> TEXT is
binary-compatible in PostgreSQL: a catalog change, no table rewrite.

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-06
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: Union[str, Sequence[str], None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column("webhooks", "secret_hash", type_=sa.Text(),
                    existing_type=sa.String(length=128), existing_nullable=False)


def downgrade() -> None:
    # Any ciphertext longer than 128 characters makes this fail rather than
    # silently truncate a secret.
    op.alter_column("webhooks", "secret_hash", type_=sa.String(length=128),
                    existing_type=sa.Text(), existing_nullable=False)
