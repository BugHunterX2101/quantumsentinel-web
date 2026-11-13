"""Dataset holdouts: where each dataset family's sealed final holdout starts.

Research tasks see a family's bars and corporate actions only before its
holdout start (services/dataset_holdouts.py, services/market_data.py). A
family with no row has no holdout.

Rows are append-only, like dataset_versions: a row trigger refuses UPDATE and
DELETE and a statement trigger refuses TRUNCATE, for the owner too. A family's
current holdout start is its latest row, and an insert trigger refuses a start
that is not later than every earlier one: moving the start earlier would seal
again data that research has already seen, which is no holdout at all. The
trigger serialises inserts for one family on a transaction-scoped advisory
lock, so two concurrent inserts cannot both pass the check.

Revision ID: 0005
Revises: 0004
Create Date: 2026-11-13
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: Union[str, Sequence[str], None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "dataset_holdouts",
        sa.Column("family", sa.String(64), nullable=False),
        sa.Column("holdout_start", sa.Date(), nullable=False),
        sa.Column("set_by", sa.String(320), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("family", "holdout_start"),
        sa.CheckConstraint("family ~ '^[A-Z][A-Z0-9_]{0,63}$'", name="ck_dataset_holdouts_family"),
        sa.CheckConstraint("length(btrim(set_by)) > 0", name="ck_dataset_holdouts_set_by"),
        sa.CheckConstraint("length(btrim(reason)) > 0", name="ck_dataset_holdouts_reason"),
    )
    op.execute("""
        CREATE OR REPLACE FUNCTION qs_holdout_moves_forward() RETURNS trigger LANGUAGE plpgsql AS $$
        DECLARE
            latest date;
        BEGIN
            PERFORM pg_advisory_xact_lock(hashtextextended('qs_dataset_holdouts:' || NEW.family, 0));
            SELECT max(holdout_start) INTO latest FROM dataset_holdouts WHERE family = NEW.family;
            IF latest IS NOT NULL AND NEW.holdout_start <= latest THEN
                RAISE EXCEPTION 'the holdout of % starts on %; it can only move later', NEW.family, latest
                    USING ERRCODE = 'restrict_violation';
            END IF;
            RETURN NEW;
        END
        $$
    """)
    op.execute("CREATE TRIGGER dataset_holdouts_move_forward BEFORE INSERT ON dataset_holdouts"
               " FOR EACH ROW EXECUTE FUNCTION qs_holdout_moves_forward()")
    op.execute("CREATE TRIGGER dataset_holdouts_immutable BEFORE UPDATE OR DELETE ON dataset_holdouts"
               " FOR EACH ROW EXECUTE FUNCTION qs_refuse_change()")
    op.execute("CREATE TRIGGER dataset_holdouts_no_truncate BEFORE TRUNCATE ON dataset_holdouts"
               " FOR EACH STATEMENT EXECUTE FUNCTION qs_refuse_change()")


def downgrade() -> None:
    op.drop_table("dataset_holdouts")  # its triggers go with it
    op.execute("DROP FUNCTION qs_holdout_moves_forward()")
