"""Hypotheses extracted from the registry's papers, and the AI tier's token budget.

* ``extractions``: one row per finished model answer for a document
  (backend/ai/hypotheses.py), whatever its status: ``ok``, ``refused``,
  ``truncated``, ``invalid`` (with ``error``) or ``unexpected_stop``. It
  keeps what the answer was for: the document's ``content_hash``, the hash
  of the exact message sent (``input_sha256``), the ``prompt_version``, the
  model asked for and the model that answered, the provider's response id
  and stop reason, the tokens used, and which hypotheses were dropped and
  why (``rejections``). (document, content_hash, prompt_version,
  requested_model) is unique, so a paper is extracted once per version,
  prompt and model. A request that got no answer leaves no row.
* ``hypotheses``: the checked hypotheses of an ``ok`` extraction, at most
  five, keyed by their place in the model's list. ``model_confidence`` is
  the model's own and for display only. What the platform observes of a
  hypothesis belongs to the trusted tier, never here.
* ``token_reservations``: the daily token budget (backend/ai/ledger.py).
  Before each request the AI worker reserves the counted input and the most
  output the request allows; the answer settles the reservation with the
  tokens used. A reservation is released only when its request never went
  out, and one left open by a worker that lost its job is marked
  ``orphaned`` and still counts in full. ``job_id`` is plain text, with no
  foreign key, as finished jobs are purged and the day's counts must stay.

``extractions`` and ``hypotheses`` are append-only: a trigger refuses any
UPDATE, DELETE or TRUNCATE, even by their owner. A reservation can be closed
once and is then fixed, and no reservation can be deleted. The AI worker
writes all three; the application only reads them. The checks repeat the
worker's own limits.

Revision ID: 0016
Revises: 0015
Create Date: 2026-11-20
"""
from typing import Sequence, Union

from alembic import op

revision: str = "0016"
down_revision: Union[str, Sequence[str], None] = "0015"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

ASSET_CLASSES = ("equities", "futures", "fx", "crypto", "rates", "credit", "commodities", "options",
                 "multi_asset", "unspecified")
SIGNAL_FAMILIES = ("ma_crossover", "momentum", "mean_reversion", "order_book_imbalance", "value", "carry",
                   "volatility", "liquidity", "seasonality", "event", "machine_learning", "other")
DIRECTIONS = ("long", "short", "long_short", "unspecified")
HORIZONS = ("intraday", "daily", "weekly", "monthly", "longer", "unspecified")
METRICS = ("sharpe_ratio", "annual_return", "t_statistic", "alpha", "information_ratio", "none")
CONFIDENCES = ("low", "medium", "high")
STATUSES = ("ok", "refused", "truncated", "invalid", "unexpected_stop")
ERRORS = ("not_utf8", "too_large", "not_json", "duplicate_key", "not_an_object", "wrong_keys", "wrong_type",
          "too_many_hypotheses", "unknown_enum")
MODEL = "'^[a-z0-9][a-z0-9.:@_-]{0,127}$'"


def _ai() -> str:
    from backend.db_roles import ai_schema_for

    name = ai_schema_for(op.get_bind().exec_driver_sql("SELECT current_schema()").scalar())
    return '"' + name.replace('"', '""') + '"'


def _in(values: tuple[str, ...]) -> str:
    return "(" + ", ".join(f"'{v}'" for v in values) + ")"


def upgrade() -> None:
    q = _ai()
    op.execute(f"""
        CREATE FUNCTION {q}.refuse_change() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION '% on % is refused: its rows are immutable', TG_OP, TG_TABLE_NAME
                USING ERRCODE = 'restrict_violation';
        END
        $$""")
    op.execute(f"""
        CREATE TABLE {q}.extractions (
            id varchar(36) PRIMARY KEY,
            document_id varchar(36) NOT NULL REFERENCES {q}.documents (id),
            content_hash varchar(81) NOT NULL CHECK (content_hash ~ '^qs-doc-sha256-v1:[0-9a-f]{{64}}$'),
            input_sha256 varchar(64) NOT NULL CHECK (input_sha256 ~ '^[0-9a-f]{{64}}$'),
            prompt_version varchar(84) NOT NULL CHECK (prompt_version ~ '^qs-prompt-sha256-v1:[0-9a-f]{{64}}$'),
            requested_model varchar(128) NOT NULL CHECK (requested_model ~ {MODEL}),
            response_model varchar(128) CHECK (response_model ~ {MODEL}),
            response_id varchar(128) CHECK (response_id ~ '^[!-~]{{1,128}}$'),
            stop_reason varchar(32) CHECK (stop_reason ~ '^[a-z_]{{1,32}}$'),
            status varchar(16) NOT NULL CHECK (status IN {_in(STATUSES)}),
            error varchar(64) CHECK (error IN {_in(ERRORS)}),
            hypothesis_count smallint NOT NULL CHECK (hypothesis_count BETWEEN 0 AND 5),
            rejections jsonb NOT NULL DEFAULT '[]'::jsonb CHECK (jsonb_typeof(rejections) = 'array'),
            input_tokens bigint NOT NULL CHECK (input_tokens >= 0),
            cache_creation_input_tokens bigint NOT NULL CHECK (cache_creation_input_tokens >= 0),
            cache_read_input_tokens bigint NOT NULL CHECK (cache_read_input_tokens >= 0),
            output_tokens bigint NOT NULL CHECK (output_tokens >= 0),
            job_id varchar(36) NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT extractions_once UNIQUE (document_id, content_hash, prompt_version, requested_model),
            CHECK ((status = 'invalid') = (error IS NOT NULL)),
            CHECK (status = 'ok' OR (hypothesis_count = 0 AND rejections = '[]'::jsonb)),
            CHECK (CASE WHEN jsonb_typeof(rejections) = 'array'
                        THEN hypothesis_count + jsonb_array_length(rejections) <= 5 END)
        )""")
    op.execute(f"CREATE INDEX extractions_created ON {q}.extractions (created_at)")
    op.execute(f"""
        CREATE TABLE {q}.hypotheses (
            extraction_id varchar(36) NOT NULL REFERENCES {q}.extractions (id),
            ordinal smallint NOT NULL CHECK (ordinal BETWEEN 1 AND 5),
            statement text NOT NULL CHECK (length(statement) BETWEEN 1 AND 500),
            evidence_quote text NOT NULL CHECK (length(evidence_quote) BETWEEN 20 AND 500),
            asset_class varchar(16) NOT NULL CHECK (asset_class IN {_in(ASSET_CLASSES)}),
            signal_family varchar(32) NOT NULL CHECK (signal_family IN {_in(SIGNAL_FAMILIES)}),
            direction varchar(16) NOT NULL CHECK (direction IN {_in(DIRECTIONS)}),
            horizon varchar(16) NOT NULL CHECK (horizon IN {_in(HORIZONS)}),
            reported_metric varchar(32) NOT NULL CHECK (reported_metric IN {_in(METRICS)}),
            reported_value varchar(32) NOT NULL,
            model_confidence varchar(8) NOT NULL CHECK (model_confidence IN {_in(CONFIDENCES)}),
            rationale text NOT NULL CHECK (length(rationale) <= 500),
            PRIMARY KEY (extraction_id, ordinal),
            CHECK ((reported_metric = 'none') = (reported_value = ''))
        )""")
    for table in ("extractions", "hypotheses"):
        op.execute(f"CREATE TRIGGER {table}_immutable BEFORE UPDATE OR DELETE ON {q}.{table}"
                   f" FOR EACH ROW EXECUTE FUNCTION {q}.refuse_change()")
        op.execute(f"CREATE TRIGGER {table}_no_truncate BEFORE TRUNCATE ON {q}.{table}"
                   f" FOR EACH STATEMENT EXECUTE FUNCTION {q}.refuse_change()")
    op.execute(f"""
        CREATE TABLE {q}.token_reservations (
            id varchar(36) PRIMARY KEY,
            job_id varchar(36) NOT NULL,
            attempt integer NOT NULL CHECK (attempt >= 1),
            worker_id varchar(255) NOT NULL,
            model varchar(128) NOT NULL CHECK (model ~ {MODEL}),
            day date NOT NULL,
            input_reserved bigint NOT NULL CHECK (input_reserved >= 0),
            output_reserved bigint NOT NULL CHECK (output_reserved >= 0),
            status varchar(16) NOT NULL DEFAULT 'reserved'
                CHECK (status IN ('reserved', 'settled', 'released', 'orphaned')),
            input_used bigint CHECK (input_used >= 0),
            cache_creation_used bigint CHECK (cache_creation_used >= 0),
            cache_read_used bigint CHECK (cache_read_used >= 0),
            output_used bigint CHECK (output_used >= 0),
            created_at timestamptz NOT NULL DEFAULT now(),
            closed_at timestamptz,
            CHECK ((status = 'reserved') = (closed_at IS NULL)),
            CHECK ((status = 'settled') = (input_used IS NOT NULL AND cache_creation_used IS NOT NULL
                                           AND cache_read_used IS NOT NULL AND output_used IS NOT NULL))
        )""")
    op.execute(f"CREATE INDEX token_reservations_day ON {q}.token_reservations (day)")
    op.execute(f"CREATE INDEX token_reservations_open ON {q}.token_reservations (job_id) WHERE status = 'reserved'")
    # A reservation is closed once, and what it reserved never changes.
    op.execute(f"""
        CREATE FUNCTION {q}.token_reservation_closes_once() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            IF OLD.status <> 'reserved' OR NEW.status = 'reserved'
               OR (NEW.id, NEW.job_id, NEW.attempt, NEW.worker_id, NEW.model, NEW.day, NEW.input_reserved,
                   NEW.output_reserved, NEW.created_at)
                  IS DISTINCT FROM (OLD.id, OLD.job_id, OLD.attempt, OLD.worker_id, OLD.model, OLD.day,
                                    OLD.input_reserved, OLD.output_reserved, OLD.created_at) THEN
                RAISE EXCEPTION 'a token reservation is closed once and keeps what it reserved'
                    USING ERRCODE = 'restrict_violation';
            END IF;
            RETURN NEW;
        END
        $$""")
    op.execute(f"CREATE TRIGGER token_reservations_close_once BEFORE UPDATE ON {q}.token_reservations"
               f" FOR EACH ROW EXECUTE FUNCTION {q}.token_reservation_closes_once()")
    op.execute(f"CREATE TRIGGER token_reservations_kept BEFORE DELETE ON {q}.token_reservations"
               f" FOR EACH ROW EXECUTE FUNCTION {q}.refuse_change()")
    op.execute(f"CREATE TRIGGER token_reservations_no_truncate BEFORE TRUNCATE ON {q}.token_reservations"
               f" FOR EACH STATEMENT EXECUTE FUNCTION {q}.refuse_change()")


def downgrade() -> None:
    q = _ai()
    op.execute(f"DROP TABLE {q}.token_reservations, {q}.hypotheses, {q}.extractions")
    op.execute(f"DROP FUNCTION {q}.token_reservation_closes_once()")
    op.execute(f"DROP FUNCTION {q}.refuse_change()")
