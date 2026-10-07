"""Indexes for the per-request queries; 64-bit audit chain sequence.

Measured on PostgreSQL 16 with 1M trades and 1M audit events (EXPLAIN
ANALYZE, warm cache), each per-user query below was a full table scan:

    open-order exposure        148 ms  ->  0.02 ms   ix_trades_user_status_filled
    positions rebuild          160 ms  ->  0.06 ms   ix_trades_user_status_filled
    sell-fill holding check    142 ms  ->  0.5 ms    ix_trades_user_status_filled
    order list                 282 ms  ->  0.6 ms    ix_trades_user_status_filled
    resting-order assets       107 ms  ->  3 ms      ix_trades_accepted_asset (partial, index-only)
    audit log                  266 ms  ->  2.7 ms    ix_audit_logs_user_created
    positions / keys / strategies 20-42 ms -> ~1 ms

ix_refresh_tokens_token_hash duplicated the unique index that
uq_refresh_token_hash already provides, so every token rotation paid for two.

Indexes are created IF NOT EXISTS, so on a very large table an operator can
build them beforehand with CREATE INDEX CONCURRENTLY under the same names and
this migration skips them. The sequence change rewrites audit_chain_links
(about 8 s per million links); int4 would overflow at 2^31 audit events.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-06
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: Union[str, Sequence[str], None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

INDEXES = (
    ("ix_trades_user_status_filled", "trades", ["user_id", "status", "filled_at"], {}),
    ("ix_trades_accepted_asset", "trades", ["asset", "submitted_at"],
     {"postgresql_where": sa.text("status = 'ACCEPTED'")}),
    ("ix_positions_user_asset", "positions", ["user_id", "asset"], {}),
    ("ix_audit_logs_user_created", "audit_logs", ["user_id", "created_at"], {}),
    ("ix_key_pairs_user_id", "key_pairs", ["user_id"], {}),
    ("ix_strategies_user_created", "strategies", ["user_id", "created_at"], {}),
)


def upgrade() -> None:
    for name, table, columns, options in INDEXES:
        op.create_index(name, table, columns, if_not_exists=True, **options)
    op.drop_index("ix_refresh_tokens_token_hash", table_name="refresh_tokens", if_exists=True)
    op.alter_column("audit_chain_links", "sequence", type_=sa.BigInteger(),
                    existing_type=sa.Integer(), existing_nullable=False)


def downgrade() -> None:
    op.alter_column("audit_chain_links", "sequence", type_=sa.Integer(),
                    existing_type=sa.BigInteger(), existing_nullable=False)
    op.create_index("ix_refresh_tokens_token_hash", "refresh_tokens", ["token_hash"], if_not_exists=True)
    for name, table, _columns, _options in reversed(INDEXES):
        op.drop_index(name, table_name=table, if_exists=True)
