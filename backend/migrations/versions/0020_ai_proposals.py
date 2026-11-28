"""Strategy signals proposed for extracted hypotheses (gate G0).

* ``proposals``: one row per hypothesis per proposal prompt version and
  requested model (backend/ai/proposals.py), whatever its status:
  ``proposed`` (a template, its parameters and ``allow_short``),
  ``not_expressible`` (the model found no template that expresses the
  hypothesis), ``out_of_scope`` (the hypothesis's family has no template, so
  no request was made), ``refused``, ``truncated``, ``invalid`` (with
  ``error``) or ``unexpected_stop``. It keeps what the answer was for: the
  hash of the exact message sent (``input_sha256``, none without a request),
  the prompt version, the model asked for and the model that answered, the
  provider's response id and stop reason, and the tokens used.
* ``jobs_one_propose``: at most one ``propose`` job queued or running, so two
  never ask about the same hypotheses and pay twice.

A proposal is only a proposal: the AI worker that writes it has no privilege
on any table of the trusted tier, and a strategy version comes from it only
when a person accepts it there. The parameters are columns, one per name the
templates use, and the checks repeat the worker's own: each template has
exactly its own parameters, in range and consistent; any other status has
none. ``window`` is a reserved word, so it is always quoted.

Append-only: a trigger refuses any UPDATE, DELETE or TRUNCATE, even by the
owner. The AI worker writes; the application only reads.

Revision ID: 0020
Revises: 0019
Create Date: 2026-11-28
"""
from typing import Sequence, Union

from alembic import op

revision: str = "0020"
down_revision: Union[str, Sequence[str], None] = "0019"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

STATUSES = ("proposed", "not_expressible", "out_of_scope", "refused", "truncated", "invalid", "unexpected_stop")
ERRORS = ("not_utf8", "too_large", "not_json", "duplicate_key", "not_an_object", "wrong_keys", "wrong_type",
          "unknown_enum", "rationale_length", "wrong_params", "duplicate_param", "param_out_of_range",
          "params_inconsistent")
MODEL = "'^[a-z0-9][a-z0-9.:@_-]{0,127}$'"
PARAMS = ("fast", "slow", "lookback", "n_long", "n_short", '"window"', "n_std_centi")


def _ai() -> str:
    from backend.db_roles import ai_schema_for

    name = ai_schema_for(op.get_bind().exec_driver_sql("SELECT current_schema()").scalar())
    return '"' + name.replace('"', '""') + '"'


def _in(values: tuple[str, ...]) -> str:
    return "(" + ", ".join(f"'{v}'" for v in values) + ")"


def _only(*names: str) -> str:
    """Exactly ``names`` of the parameters are set."""
    return " AND ".join(f"{p} IS {'NOT ' if p in names else ''}NULL" for p in PARAMS)


def upgrade() -> None:
    q = _ai()
    op.execute(f"""
        CREATE TABLE {q}.proposals (
            id varchar(36) PRIMARY KEY,
            extraction_id varchar(36) NOT NULL,
            ordinal smallint NOT NULL,
            input_sha256 varchar(64) CHECK (input_sha256 ~ '^[0-9a-f]{{64}}$'),
            prompt_version varchar(84) NOT NULL CHECK (prompt_version ~ '^qs-prompt-sha256-v1:[0-9a-f]{{64}}$'),
            requested_model varchar(128) NOT NULL CHECK (requested_model ~ {MODEL}),
            response_model varchar(128) CHECK (response_model ~ {MODEL}),
            response_id varchar(128) CHECK (response_id ~ '^[!-~]{{1,128}}$'),
            stop_reason varchar(32) CHECK (stop_reason ~ '^[a-z_]{{1,32}}$'),
            status varchar(16) NOT NULL CHECK (status IN {_in(STATUSES)}),
            error varchar(32) CHECK (error IN {_in(ERRORS)}),
            template varchar(32) CHECK (template IN ('ma_crossover', 'momentum', 'mean_reversion')),
            fast smallint, slow smallint, lookback smallint, n_long smallint, n_short smallint,
            "window" smallint, n_std_centi smallint,
            allow_short boolean,
            rationale text CHECK (length(rationale) <= 500),
            input_tokens bigint NOT NULL CHECK (input_tokens >= 0),
            cache_creation_input_tokens bigint NOT NULL CHECK (cache_creation_input_tokens >= 0),
            cache_read_input_tokens bigint NOT NULL CHECK (cache_read_input_tokens >= 0),
            output_tokens bigint NOT NULL CHECK (output_tokens >= 0),
            job_id varchar(36) NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now(),
            FOREIGN KEY (extraction_id, ordinal) REFERENCES {q}.hypotheses (extraction_id, ordinal),
            CONSTRAINT proposals_once UNIQUE (extraction_id, ordinal, prompt_version, requested_model),
            CHECK ((status = 'invalid') = (error IS NOT NULL)),
            CHECK ((status = 'proposed') = (template IS NOT NULL)),
            CHECK ((status = 'proposed') = (allow_short IS NOT NULL)),
            CHECK ((status IN ('proposed', 'not_expressible')) = (rationale IS NOT NULL)),
            CHECK ((status = 'out_of_scope') = (input_sha256 IS NULL)),
            CHECK (status <> 'out_of_scope' OR (response_model IS NULL AND response_id IS NULL
                   AND stop_reason IS NULL AND input_tokens = 0 AND cache_creation_input_tokens = 0
                   AND cache_read_input_tokens = 0 AND output_tokens = 0)),
            CHECK (CASE template
                WHEN 'ma_crossover' THEN {_only("fast", "slow")}
                    AND fast BETWEEN 2 AND 200 AND slow BETWEEN 5 AND 500 AND slow > fast
                WHEN 'momentum' THEN {_only("lookback", "n_long", "n_short")}
                    AND lookback BETWEEN 2 AND 500 AND n_long BETWEEN 0 AND 3 AND n_short BETWEEN 0 AND 3
                    AND n_long + n_short > 0 AND (allow_short OR n_long >= 1)
                WHEN 'mean_reversion' THEN {_only('"window"', "n_std_centi")}
                    AND "window" BETWEEN 2 AND 500 AND n_std_centi BETWEEN 1 AND 1000
                ELSE {_only()}
            END)
        )""")
    op.execute(f"CREATE INDEX proposals_created ON {q}.proposals (created_at)")
    op.execute(f"CREATE TRIGGER proposals_immutable BEFORE UPDATE OR DELETE ON {q}.proposals"
               f" FOR EACH ROW EXECUTE FUNCTION {q}.refuse_change()")
    op.execute(f"CREATE TRIGGER proposals_no_truncate BEFORE TRUNCATE ON {q}.proposals"
               f" FOR EACH STATEMENT EXECUTE FUNCTION {q}.refuse_change()")
    op.execute(f"""CREATE UNIQUE INDEX jobs_one_propose ON {q}.jobs (kind)
                   WHERE kind = 'propose' AND status IN ('queued', 'running')""")


def downgrade() -> None:
    q = _ai()
    op.execute(f"DROP INDEX {q}.jobs_one_propose")
    op.execute(f"DROP TABLE {q}.proposals")
