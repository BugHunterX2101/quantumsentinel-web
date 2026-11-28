"""Strategy-signal proposals: the checks, the prompt and the table (backend/ai/proposals.py, migration 0020).

The model's answer is untrusted: the hypothesis it reads came from a paper
that may be written to steer it. These tests show every answer the checks
must refuse refused for its reason, and that the three copies of the
template bounds (the AI tier's PARAMS, the table's CHECK and the strategy
DSL the trusted tier parses an accepted proposal with) give the same answer
on every boundary. The driver is tested end to end in tests/test_propose.py.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from psycopg import errors
from pydantic import ValidationError

from backend.ai import hypotheses, proposals
from backend.ai.ledger import Usage
from backend.ai.proposals import check_output
from backend.services import strategy_spec
from test_hypotheses import Db, MODEL, PROMPT as PAPER_PROMPT, GOLDEN, check_output as check_paper

HYPOTHESIS = {"statement": "Time-series momentum in commodity futures earns a positive risk-adjusted return.",
              "evidence_quote": "earns an annualized Sharpe ratio of 0.85 over 1990–2010",
              "asset_class": "futures", "signal_family": "momentum", "direction": "long_short",
              "horizon": "monthly"}


def answer(template="ma_crossover", params=None, allow_short=False, rationale="A crossover expresses it.",
           **extra) -> str:
    if params is None:
        params = {"fast": 20, "slow": 50} if template == "ma_crossover" else {}
    items = [{"name": k, "value": v} for k, v in params.items()] if isinstance(params, dict) else params
    return json.dumps({"template": template, "params": items, "allow_short": allow_short,
                       "rationale": rationale, **extra})


# -- the checks ---------------------------------------------------------------------

def test_a_well_formed_answer_becomes_a_proposal_with_sorted_parameters():
    checked = check_output("end_turn", answer("momentum", {"n_short": 1, "lookback": 126, "n_long": 1}, True))
    assert checked.status == "proposed" and checked.error is None and checked.rationale is None
    assert checked.proposal == proposals.Proposal(
        "momentum", (("lookback", 126), ("n_long", 1), ("n_short", 1)), True, "A crossover expresses it.")


def test_template_and_parameter_names_are_read_case_insensitively_and_the_rationale_normalised():
    checked = check_output("end_turn", answer(" MA_Crossover ", [{"name": "FAST", "value": 20},
                                                                {"name": "slow ", "value": 50}],
                                              rationale="  A​crossover   expresses\nit. "))
    assert checked.proposal.template == "ma_crossover" and checked.proposal.params == (("fast", 20), ("slow", 50))
    assert checked.proposal.rationale == "Acrossover expresses it."


def test_not_expressible_keeps_its_rationale_and_takes_no_parameters():
    checked = check_output("end_turn", answer("not_expressible", [], rationale="No template fits."))
    assert (checked.status, checked.proposal, checked.rationale) == ("not_expressible", None, "No template fits.")
    assert check_output("end_turn", answer("not_expressible", {"fast": 2})).error == "wrong_params"


@pytest.mark.parametrize("stop_reason, status", [
    ("refusal", "refused"), ("max_tokens", "truncated"), ("pause_turn", "unexpected_stop"),
    ("tool_use", "unexpected_stop"), (None, "unexpected_stop"), ("", "unexpected_stop"),
])
def test_only_a_normal_stop_is_read(stop_reason, status):
    checked = check_output(stop_reason, answer())
    assert (checked.status, checked.proposal, checked.error) == (status, None, None)


INVALID = [
    (b"\xff\xfe", "not_utf8"), ("\ud800", "not_utf8"), (None, "not_json"), (5, "not_json"),
    ("x" * (proposals.MAX_OUTPUT_BYTES + 1), "too_large"), ("I propose a crossover.", "not_json"),
    ('{"template": "momentum", ' + answer()[1:], "duplicate_key"), ("[]", "not_an_object"),
    (answer().replace('"allow_short": false', '"allow_short": NaN'), "not_json"),
    (answer(verified=True), "wrong_keys"), (json.dumps({"template": "ma_crossover"}), "wrong_keys"),
    (answer(template=5), "wrong_type"), (answer(params={}).replace("[]", "{}"), "wrong_type"),
    (answer(allow_short="no"), "wrong_type"), (answer(rationale=None), "wrong_type"),
    (answer(params=["fast"]), "wrong_type"), (answer(params=[{"name": "fast"}]), "wrong_keys"),
    (answer(params=[{"name": "fast", "value": 20, "unit": "bars"}]), "wrong_keys"),
    (answer(params=[{"name": 1, "value": 20}]), "wrong_type"),
    (answer(params={"fast": 20.0, "slow": 50}), "wrong_type"), (answer(params={"fast": True, "slow": 50}), "wrong_type"),
    (answer(params={"fast": "20", "slow": 50}), "wrong_type"),
    (answer(template="pairs_trading"), "unknown_enum"), (answer(params={"fast": 20, "speed": 50}), "unknown_enum"),
    (answer(rationale="x" * (proposals.MAX_RATIONALE + 1)), "rationale_length"),
    (answer(params=[{"name": "fast", "value": 20}, {"name": "fast", "value": 30}, {"name": "slow", "value": 50}]),
     "duplicate_param"),
    (answer(params={"fast": 20, "slow": 50, "lookback": 10}), "wrong_params"),
    (answer(params={"fast": 20}), "wrong_params"),
    (answer(params={"fast": 1, "slow": 50}), "param_out_of_range"),
    (answer(params={"fast": 50, "slow": 20}), "params_inconsistent"),
]


@pytest.mark.parametrize("text, error", INVALID)
def test_an_answer_the_checks_must_refuse_is_invalid_for_its_reason(text, error):
    checked = check_output("end_turn", text)
    assert (checked.status, checked.error, checked.proposal) == ("invalid", error, None)
    assert error in proposals.ERRORS


def test_a_deep_answer_is_not_json_not_a_crash():
    text = "[" * 8_000 + "]" * 8_000
    assert len(text) <= proposals.MAX_OUTPUT_BYTES and check_output("end_turn", text).error == "not_json"


def test_every_error_is_shown_reachable():
    assert {error for _, error in INVALID} == set(proposals.ERRORS)


# -- three copies of the bounds -----------------------------------------------------

def _bound_cases():
    """(template, name, value, params): ``name`` on and just past each of its
    bounds, the other parameters valid."""
    base = {"ma_crossover": {"fast": 20, "slow": 300}, "momentum": {"lookback": 126, "n_long": 1, "n_short": 1},
            "mean_reversion": {"window": 20, "n_std_centi": 200}}
    for template, ranges in proposals.PARAMS.items():
        for name, (low, high) in ranges.items():
            for value in (low - 1, low, high, high + 1):
                params = {**base[template], name: value}
                if template == "ma_crossover" and name == "slow":
                    params["fast"] = 2  # slow > fast wherever slow's own bound is in range
                yield template, name, value, params


BOUND_CASES = list(_bound_cases())


def _boundary_cases():
    """(template, params, allow_short) on and around every bound and rule."""
    for template, _, _, params in BOUND_CASES:
        yield template, params, True
    yield "ma_crossover", {"fast": 30, "slow": 30}, False
    yield "ma_crossover", {"fast": 30, "slow": 31}, False
    yield "momentum", {"lookback": 20, "n_long": 0, "n_short": 0}, True
    yield "momentum", {"lookback": 20, "n_long": 0, "n_short": 1}, False
    yield "momentum", {"lookback": 20, "n_long": 0, "n_short": 1}, True
    yield "momentum", {"lookback": 20, "n_long": 1, "n_short": 0}, False


BOUNDARY_CASES = list(_boundary_cases())


def _accepts(template, params, allow_short) -> bool:
    try:
        proposals.check_params(template, params, allow_short)
    except proposals._Invalid:
        return False
    return True


def _dsl_accepts(template, params, allow_short) -> bool:
    """Whether the trusted tier's DSL parses the proposal, with the fields a
    person picks fixed (nine symbols: a third of the universe is 3, the
    proposal's own momentum cap)."""
    spec = {"dsl": "qs-strategy-v1", "signal": {"template": template, **params},
            "universe": {"family": "EQUITY_EOD", "symbols": [f"S{i}" for i in range(9)]},
            "sizing": {"method": "fixed_fractional", "fraction_bps": 200}, "cost_model": "retail",
            "allow_short": allow_short, "leverage_limit_bps": 10_000, "execution": "next_bar_close",
            "risk_limits": {"max_drawdown_bps": 2_000, "max_position_bps": 5_000}}
    try:
        strategy_spec.parse(spec)
    except (ValidationError, ValueError):
        return False
    return True


@pytest.mark.parametrize("template, params, allow_short", BOUNDARY_CASES)
def test_the_checks_and_the_strategy_dsl_agree_on_every_bound(template, params, allow_short):
    assert _accepts(template, params, allow_short) == _dsl_accepts(template, params, allow_short)


def test_the_boundary_cases_include_both_answers_for_every_parameter():
    for template, ranges in proposals.PARAMS.items():
        for name, (low, high) in ranges.items():
            cases = {value: params for t, n, value, params in BOUND_CASES if (t, n) == (template, name)}
            assert set(cases) == {low - 1, low, high, high + 1}, name
            assert _accepts(template, cases[low], True) and _accepts(template, cases[high], True), name
            assert not _accepts(template, cases[low - 1], True) and not _accepts(template, cases[high + 1], True), name


def test_the_momentum_cap_is_a_third_of_the_largest_universe_the_dsl_allows():
    assert proposals.MAX_MOMENTUM_SIDE == strategy_spec.MAX_SYMBOLS // 3


def test_the_templates_and_their_parameters_are_the_dsls():
    models = {"ma_crossover": strategy_spec.MACrossover, "momentum": strategy_spec.Momentum,
              "mean_reversion": strategy_spec.MeanReversion}
    assert set(proposals.TEMPLATES) == set(models) and set(proposals.TEMPLATES) <= set(hypotheses.SIGNAL_FAMILIES)
    for template, model in models.items():
        assert model.model_fields["template"].annotation.__args__ == (template,)
        assert set(model.model_fields) - {"template"} == set(proposals.PARAMS[template])


def test_the_prompt_states_the_bounds_the_checks_apply():
    for template, ranges in proposals.PARAMS.items():
        for name, (low, high) in ranges.items():
            if template != "momentum" or name == "lookback":
                assert f"{name} ({low} to {high})" in proposals.SYSTEM_PROMPT
    assert f"each 0 to {proposals.MAX_MOMENTUM_SIDE}" in proposals.SYSTEM_PROMPT
    assert proposals.PARAMS["momentum"]["n_long"] == proposals.PARAMS["momentum"]["n_short"] == (
        0, proposals.MAX_MOMENTUM_SIDE)
    assert f"at most {proposals.MAX_RATIONALE} characters" in proposals.SYSTEM_PROMPT


def test_the_schema_offers_exactly_what_the_checks_accept():
    schema = proposals.OUTPUT_SCHEMA
    assert schema["required"] == list(proposals.KEYS) == list(schema["properties"])
    assert schema["additionalProperties"] is False
    assert schema["properties"]["template"]["enum"] == list(proposals.CHOICES)
    item = schema["properties"]["params"]["items"]
    assert item["properties"]["name"]["enum"] == list(proposals.PARAM_NAMES) and item["additionalProperties"] is False
    assert not any(word in json.dumps(schema).lower() for word in ("verified", "confidence", "backtest"))


# -- the prompt ---------------------------------------------------------------------

def test_the_prompt_sends_only_the_hypothesis_fields_normalised():
    prompt = proposals.prompt_for({**HYPOTHESIS, "statement": " Momentum​  pays. ", "rationale": "secret",
                                   "model_confidence": "unmistakable"})
    sent = json.loads(prompt.user_message.split("\n", 1)[1])
    assert list(sent) == list(proposals.HYPOTHESIS_FIELDS) and sent["statement"] == "Momentum pays."
    assert "secret" not in prompt.user_message and "unmistakable" not in prompt.user_message
    assert prompt.input_sha256 == hashlib.sha256(prompt.user_message.encode("utf-8")).hexdigest()


def test_only_template_families_are_put_to_the_model():
    assert {f for f in hypotheses.SIGNAL_FAMILIES if proposals.in_scope(f)} == set(proposals.TEMPLATES)


def test_the_prompt_version_hashes_what_shapes_the_answer(monkeypatch):
    version = proposals._prompt_version()
    assert version == proposals.PROMPT_VERSION and version.startswith("qs-prompt-sha256-v1:")
    for name, value in (("SYSTEM_PROMPT", "other"), ("USER_TEMPLATE", "{hypothesis}"), ("MAX_OUTPUT_TOKENS", 2048),
                        ("OUTPUT_SCHEMA", {}), ("TASK", "other")):
        with monkeypatch.context() as patch:
            patch.setattr(proposals, name, value)
            assert proposals._prompt_version() != version, name
    assert version != hypotheses.PROMPT_VERSION


# -- the table ----------------------------------------------------------------------

def test_the_migration_checks_the_same_values_as_the_code():
    path = Path(__file__).parents[1] / "backend" / "migrations" / "versions" / "0020_ai_proposals.py"
    spec = importlib.util.spec_from_file_location("migration_0020", path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    assert migration.STATUSES == proposals.STATUSES and migration.ERRORS == proposals.ERRORS
    assert migration.MODEL == "'^" + proposals._MODEL.pattern + "$'"
    assert {p.strip('"') for p in migration.PARAMS} == set(proposals.PARAM_NAMES)


class Pdb(Db):
    def extraction(self, source_id="2611.00001") -> str:
        document = self.document(source_id)
        self.record(document, check_paper("end_turn", GOLDEN, PAPER_PROMPT))
        return self.conn.execute(f'SELECT id FROM "{self.schema}".extractions WHERE document_id = %s',
                                 (document[0],)).fetchone()[0]

    def propose(self, extraction_id, checked, ordinal=1, model=MODEL, **fields):
        values = {"response_model": "claude-opus-5-5-20261101", "response_id": "msg_01ABC",
                  "stop_reason": "end_turn", "usage": Usage(900, 0, 0, 120), **fields}
        with self.conn.transaction():
            return proposals.record_proposal(
                self.conn, self.schema, job_id="job-1", extraction_id=extraction_id, ordinal=ordinal,
                prompt=proposals.prompt_for(HYPOTHESIS), requested_model=model, checked=checked, **values)

    def insert(self, extraction_id, **overrides):
        values = {"id": "p1", "extraction_id": extraction_id, "ordinal": 1, "input_sha256": "0" * 64,
                  "prompt_version": proposals.PROMPT_VERSION, "requested_model": MODEL, "status": "proposed",
                  "template": "ma_crossover", "fast": 20, "slow": 50, "allow_short": False, "rationale": "r",
                  "input_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
                  "output_tokens": 0, "job_id": "j", **overrides}
        names = ", ".join(f'"{k}"' for k in values)
        marks = ", ".join(["%s"] * len(values))
        with self.conn.transaction():
            self.conn.execute(f'INSERT INTO "{self.schema}".proposals ({names}) VALUES ({marks})',
                              list(values.values()))


@pytest.fixture
def pdb(make_engine):
    database = Pdb(make_engine(migrated=True))
    yield database
    database.conn.close()


def test_a_proposal_is_recorded_once_per_prompt_version_and_model(pdb):
    extraction_id = pdb.extraction()
    checked = check_output("end_turn", answer("mean_reversion", {"window": 20, "n_std_centi": 200}, True))
    first = pdb.propose(extraction_id, checked)
    assert first is not None and pdb.propose(extraction_id, checked) is None
    assert pdb.propose(extraction_id, checked, model="claude-sonnet-5-5") is not None
    [row] = pdb.rows("proposals", "id = %s", (first,))
    assert (row["status"], row["template"], row["window"], row["n_std_centi"], row["allow_short"]) == (
        "proposed", "mean_reversion", 20, 200, True)
    assert (row["fast"], row["slow"], row["lookback"], row["n_long"], row["n_short"]) == (None,) * 5
    assert (row["prompt_version"], row["input_tokens"], row["output_tokens"], row["job_id"]) == (
        proposals.PROMPT_VERSION, 900, 120, "job-1")
    assert row["input_sha256"] == proposals.prompt_for(HYPOTHESIS).input_sha256


def test_every_status_an_answer_can_have_is_recorded(pdb):
    extraction_id = pdb.extraction()
    assert pdb.propose(extraction_id, check_output("end_turn", answer("not_expressible", [], rationale="None fits.")),
                       ordinal=1) is not None
    assert pdb.propose(extraction_id, check_output("refusal", ""), ordinal=2) is not None
    for model, checked in (("m-a", check_output("max_tokens", "{")), ("m-b", check_output("pause_turn", answer())),
                           ("m-c", check_output("end_turn", "prose"))):
        assert pdb.propose(extraction_id, checked, model=model) is not None
    rows = {r["status"]: r for r in pdb.rows("proposals")}
    assert set(rows) == {"not_expressible", "refused", "truncated", "unexpected_stop", "invalid"}
    assert rows["invalid"]["error"] == "not_json" and rows["not_expressible"]["rationale"] == "None fits."
    assert all(r["template"] is None and r["allow_short"] is None for r in rows.values())


def test_an_out_of_scope_hypothesis_is_recorded_without_a_request(pdb):
    extraction_id = pdb.extraction()
    with pdb.conn.transaction():
        found = proposals.record_out_of_scope(pdb.conn, pdb.schema, job_id="j", extraction_id=extraction_id,
                                              ordinal=1, requested_model=MODEL)
        again = proposals.record_out_of_scope(pdb.conn, pdb.schema, job_id="j", extraction_id=extraction_id,
                                              ordinal=1, requested_model=MODEL)
    assert found is not None and again is None
    [row] = pdb.rows("proposals")
    assert (row["status"], row["input_sha256"], row["stop_reason"], row["input_tokens"]) == (
        "out_of_scope", None, None, 0)


def test_what_the_provider_reports_is_kept_only_if_it_is_well_formed(pdb):
    extraction_id = pdb.extraction()
    pdb.propose(extraction_id, check_output("end_turn", answer()), response_model="Claude Opus",
                response_id="msg 01", stop_reason="END TURN")
    [row] = pdb.rows("proposals")
    assert (row["response_model"], row["response_id"], row["stop_reason"]) == (None, None, None)
    with pytest.raises(ValueError, match="status"):
        pdb.propose(extraction_id, proposals.Checked("verified"))
    with pytest.raises(ValueError, match="status"):
        pdb.propose(extraction_id, proposals.Checked("out_of_scope"))
    with pytest.raises(ValueError, match="model id"):
        pdb.propose(extraction_id, check_output("end_turn", answer()), model="Claude; DROP TABLE")


def test_a_proposal_needs_a_recorded_hypothesis(pdb):
    extraction_id = pdb.extraction()
    with pytest.raises(errors.ForeignKeyViolation):
        pdb.insert("no-such-extraction")
    with pytest.raises(errors.ForeignKeyViolation):
        pdb.insert(extraction_id, ordinal=5)


def test_recorded_proposals_cannot_be_changed_even_by_their_owner(pdb):
    pdb.propose(pdb.extraction(), check_output("end_turn", answer()))
    for statement in (f'UPDATE "{pdb.schema}".proposals SET fast = 21', f'DELETE FROM "{pdb.schema}".proposals',
                      f'TRUNCATE "{pdb.schema}".proposals'):
        with pytest.raises(errors.RestrictViolation, match="immutable"):
            with pdb.conn.transaction():
                pdb.conn.execute(statement)
    assert len(pdb.rows("proposals")) == 1


_NOT_PROPOSED = {"template": None, "fast": None, "slow": None, "allow_short": None, "rationale": None}


@pytest.mark.parametrize("overrides", [
    {"status": "verified"}, {"status": "invalid"}, {"error": "not_json"},
    {"status": "invalid", "error": "bad", **_NOT_PROPOSED}, {"template": None}, {"allow_short": None},
    {"rationale": None}, {"rationale": "r" * 501}, {"template": "pairs_trading"}, {"input_sha256": None},
    {"input_sha256": "X" * 64}, {"prompt_version": "v1"}, {"requested_model": "Claude"}, {"response_model": "a b"},
    {"response_id": "a b"}, {"stop_reason": "End"}, {"output_tokens": -1}, {"lookback": 20}, {"slow": None},
    {"window": 20}, {"status": "out_of_scope", **_NOT_PROPOSED},
    {"status": "out_of_scope", "input_sha256": None, "output_tokens": 5, **_NOT_PROPOSED},
    {"status": "out_of_scope", "input_sha256": None, "stop_reason": "end_turn", **_NOT_PROPOSED},
    {"status": "refused", "lookback": 5, **_NOT_PROPOSED}, {"status": "refused", **_NOT_PROPOSED, "rationale": "r"},
    {"status": "not_expressible", **_NOT_PROPOSED},
])
def test_the_database_refuses_a_proposal_the_code_would_never_write(pdb, overrides):
    extraction_id = pdb.extraction()
    with pytest.raises(errors.CheckViolation):
        pdb.insert(extraction_id, **overrides)


def test_the_database_accepts_every_status_the_code_may_write(pdb):
    extraction_id = pdb.extraction()
    rows = [("p1", {}), ("p2", {"status": "not_expressible", **_NOT_PROPOSED, "rationale": "r"}),
            ("p3", {"status": "out_of_scope", "input_sha256": None, **_NOT_PROPOSED}),
            ("p4", {"status": "invalid", "error": "too_large", **_NOT_PROPOSED})]
    for (ident, overrides), model in zip(rows, ("m-a", "m-b", "m-c", "m-d")):
        pdb.insert(extraction_id, id=ident, requested_model=model, **overrides)
    assert len(pdb.rows("proposals")) == 4


@pytest.mark.parametrize("template, params, allow_short", BOUNDARY_CASES)
def test_the_database_agrees_with_the_checks_on_every_bound(pdb, template, params, allow_short):
    extraction_id = pdb.extraction()
    overrides = {"template": template, "allow_short": allow_short, "fast": None, "slow": None, **params}
    if _accepts(template, params, allow_short):
        pdb.insert(extraction_id, **overrides)
    else:
        with pytest.raises(errors.CheckViolation):
            pdb.insert(extraction_id, **overrides)
