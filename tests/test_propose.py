"""Strategy-signal proposals end to end (backend/ai/propose.py, backend/ai/proposals.py, migration 0020).

The worker runs each propose job as it does in production: the driver in
the worker, each request in a job process, against the adversarial fake
model. A hypothesis (its statement or its quote) can tell the fake to
misbehave (``[[fake:<behaviour>]]``) as a model the paper has subverted, or
a failing provider, would. The tests show every such answer recorded for
what it is, every request ending in exactly one ledger action, and only the
hypotheses of current extractions asked about. The role privileges are
tested in tests/test_db_roles.py; the checks themselves in
tests/test_proposals.py.
"""
import collections
import dataclasses
import hashlib
from unittest import mock

import pytest

from backend import manage
from backend.ai import child, extract, hypotheses, ledger, models, propose, proposals, queue
from backend.ai import settings as ai_settings
from backend.ai.kinds import KINDS
from backend.ai.worker import AiWorker
from test_ai_tier import Ai, ai  # noqa: F401 - the ai fixture
from test_extract import SENTENCE, Ext, paper

STATEMENT = "The paper states a testable trading hypothesis."
OLD_PROMPT = hypotheses.PROMPT_SCHEME + ":" + "0" * 64
MA = {"fast": 20, "slow": 50}


def hyp(behaviour: str | None = None, family: str = "ma_crossover", direction: str = "long", **fields) -> dict:
    """A hypothesis; its statement carries the directive, if any."""
    return {"statement": STATEMENT + (f" [[fake:{behaviour}]]" if behaviour else ""), "evidence_quote": SENTENCE,
            "asset_class": "futures", "signal_family": family, "direction": direction, "horizon": "monthly",
            **fields}


class Prop(Ext):
    def extraction(self, number: int, *found: dict, model: str = ai_settings.DEFAULT_MODEL,
                   version: str = hypotheses.PROMPT_VERSION) -> str:
        """An extraction of paper ``number`` (planted if it is not there yet)
        holding ``found``, in a transaction of its own, so a later one is newer."""
        if not self.rows("SELECT 1 FROM {ai}.documents WHERE source_id = %s", (f"2611.{number:05d}",)):
            self.plant(paper(number))
        [doc] = self.rows("SELECT id, content_hash, title, abstract FROM {ai}.documents WHERE source_id = %s",
                          (f"2611.{number:05d}",))
        checked = hypotheses.Checked("ok", tuple(
            hypotheses.Hypothesis(ordinal=n, reported_metric="none", reported_value="", model_confidence="medium",
                                  rationale="The quote states it.", **h) for n, h in enumerate(found, start=1)))
        with self.ai.conn.transaction(), mock.patch.object(hypotheses, "PROMPT_VERSION", version):
            extraction_id = hypotheses.record_extraction(
                self.ai.conn, self.ai.schema, job_id="planted", document_id=doc["id"],
                content_hash=doc["content_hash"], prompt=hypotheses.prompt_for(doc["title"], doc["abstract"]),
                requested_model=model, response_model=model, response_id=None, stop_reason="end_turn",
                usage=ledger.Usage(), checked=checked)
        assert extraction_id is not None
        return extraction_id

    def run(self, payload=None, **environ) -> tuple[str, dict]:
        job_id = self.ai.enqueue(payload or {}, kind="propose")
        runner = self.worker(**environ)
        try:
            ended = runner.run_once()
        finally:
            runner.close()
        return ended, self.ai.job(job_id)

    def proposals(self) -> list[dict]:
        return self.rows(
            "SELECT h.statement, h.evidence_quote, h.signal_family, h.direction, p.* FROM {ai}.proposals p"
            " JOIN {ai}.hypotheses h USING (extraction_id, ordinal) ORDER BY p.created_at, p.id")


@pytest.fixture
def prop(ai):
    """The propose stage on, and no hypotheses yet."""
    ai.switch(True, True, "propose")
    return Prop(ai)


@pytest.fixture
def quick(monkeypatch):
    """Retries without waiting, and at most two in a row."""
    monkeypatch.setattr(extract, "RETRY_SECONDS", 0)
    monkeypatch.setattr(extract, "MAX_RETRIES", 2)


# -- the job process's request -----------------------------------------------------

def payload(found: dict | None = None, **overrides) -> dict:
    return {"provider": "fake", "model": "claude-opus-5-5", "timeout_seconds": 150, "task": proposals.TASK,
            "user_message": proposals.prompt_for(found or hyp()).user_message, **overrides}


def test_the_request_takes_the_proposal_prompt_schema_and_limit_from_the_code_never_the_payload():
    provider, request = models.request_from(payload())
    assert (provider, request.task) == ("fake", proposals.TASK)
    assert (request.system, request.output_schema, request.max_tokens) == (
        proposals.SYSTEM_PROMPT, proposals.OUTPUT_SCHEMA, proposals.MAX_OUTPUT_TOKENS)
    assert models.request_from({**payload(), "task": "hypotheses"})[1].system == hypotheses.SYSTEM_PROMPT
    without = payload()
    del without["task"]
    assert models.request_from(without)[1].task == "hypotheses"
    for task in ("strategy", "", 5, None, ["proposal"]):
        with pytest.raises(ValueError, match="task must be one of"):
            models.request_from(payload(task=task))


def test_the_reservation_covers_what_the_fake_counts_and_the_fake_proposes_honestly():
    prompt = proposals.prompt_for(hyp())
    reply = models.answer(payload())
    assert reply["usage"]["input_tokens"] < propose.input_token_bound(prompt)
    checked = proposals.check_output(reply["stop_reason"], reply["text"])
    assert checked.status == "proposed" and checked.proposal.template == "ma_crossover"
    assert dict(checked.proposal.params) == MA and checked.proposal.allow_short is False


def test_an_oversize_proposal_is_cut_to_just_past_its_own_limit_so_it_stays_too_large():
    """The proposal limit, not the larger hypothesis limit, bounds the text."""
    text = models.answer(payload(hyp("oversize")))["text"]
    assert proposals.MAX_OUTPUT_BYTES < len(text.encode("utf-8")) <= proposals.MAX_OUTPUT_BYTES + 4
    assert proposals.check_output("end_turn", text).error == "too_large"


def test_the_fake_reads_only_the_proposal_prompt_and_known_behaviours():
    with pytest.raises(ValueError, match="only the proposal prompt"):
        models.answer(payload(user_message="Propose a strategy."))
    with pytest.raises(ValueError, match="no proposal behaviour"):
        models.answer(payload(hyp("obey")))
    with pytest.raises(ValueError, match="no proposal behaviour"):
        models.answer(payload(hyp("invent_quote")))  # a behaviour of the hypothesis prompt only


@pytest.mark.parametrize("value", [None, [], {"max_hypotheses": 0}, {"max_hypotheses": True},
                                   {"max_hypotheses": 1.5}, {"max_hypotheses": "5"},
                                   {"max_hypotheses": propose.MAX_HYPOTHESES + 1},
                                   {"max_hypotheses": 5, "model": "x"}, {"max_docs": 5}])
def test_the_job_payload_is_checked(value):
    with pytest.raises(ValueError):
        propose.check_payload(value)


def test_the_job_payload_may_name_a_limit_or_none():
    assert propose.check_payload({}) is None
    assert propose.check_payload({"max_hypotheses": propose.MAX_HYPOTHESES}) == propose.MAX_HYPOTHESES


def test_the_kind_runs_one_request_per_job_process_within_its_deadline():
    spec = KINDS["propose"]
    assert (spec.target, spec.driver, spec.stage) == ("backend.ai.models:answer", "backend.ai.propose:run", "propose")
    assert extract.REQUEST_TIMEOUT_SECONDS < spec.child_timeout_seconds < spec.timeout_seconds
    assert spec.child_output_max_bytes > proposals.MAX_OUTPUT_BYTES * 6


# -- the driver, end to end --------------------------------------------------------

PROPOSED_MA = ("proposed", None, "ma_crossover", MA, False)
CASES = [  # (hypothesis, (status, error, template, params, allow_short))
    (hyp(), PROPOSED_MA),
    (hyp(family="momentum", direction="long_short"),
     ("proposed", None, "momentum", {"lookback": 126, "n_long": 1, "n_short": 1}, True)),
    (hyp(family="momentum", direction="short"),
     ("proposed", None, "momentum", {"lookback": 126, "n_long": 0, "n_short": 1}, True)),
    (hyp(family="momentum", direction="long"),
     ("proposed", None, "momentum", {"lookback": 126, "n_long": 1, "n_short": 0}, False)),
    (hyp(family="mean_reversion", direction="unspecified"),
     ("proposed", None, "mean_reversion", {"window": 20, "n_std_centi": 200}, False)),
    (hyp("overspend"), PROPOSED_MA),
    (hyp("decline"), ("not_expressible", None, None, {}, None)),
    (hyp("certify"), ("invalid", "wrong_keys", None, {}, None)),
    (hyp("duplicate_key"), ("invalid", "duplicate_key", None, {}, None)),
    (hyp("oversize"), ("invalid", "too_large", None, {}, None)),
    (hyp("not_json"), ("invalid", "not_json", None, {}, None)),
    (hyp("unknown_template"), ("invalid", "unknown_enum", None, {}, None)),
    (hyp("wrong_params"), ("invalid", "wrong_params", None, {}, None)),
    (hyp("duplicate_param"), ("invalid", "duplicate_param", None, {}, None)),
    (hyp("out_of_range"), ("invalid", "param_out_of_range", None, {}, None)),
    (hyp("inconsistent"), ("invalid", "params_inconsistent", None, {}, None)),
    (hyp("float_param"), ("invalid", "wrong_type", None, {}, None)),
    (hyp("bool_param"), ("invalid", "wrong_type", None, {}, None)),
    (hyp("long_rationale"), ("invalid", "rationale_length", None, {}, None)),
    (hyp("truncate"), ("truncated", None, None, {}, None)),
    (hyp("refuse"), ("refused", None, None, {}, None)),
    (hyp(evidence_quote=SENTENCE + " [[fake:odd_stop]]"), ("unexpected_stop", None, None, {}, None)),
    (hyp(family="value"), ("out_of_scope", None, None, {}, None)),
    (hyp("crash", family="carry"), ("out_of_scope", None, None, {}, None)),  # never sent, so never run
]


def _key(row: dict) -> tuple:
    return row["statement"], row["evidence_quote"], row["signal_family"], row["direction"]


def test_every_answer_a_subverted_model_gives_is_recorded_for_what_it_is_and_settled_as_reported(prop):
    for start in range(0, len(CASES), hypotheses.MAX_HYPOTHESES):
        prop.extraction(start + 1, *(h for h, _ in CASES[start:start + hypotheses.MAX_HYPOTHESES]))
    ended, job = prop.run()
    assert ended == "succeeded", job
    rows = {_key(r): r for r in prop.proposals()}
    assert len(rows) == len(CASES)
    asked = 0
    for found, expected in CASES:
        row = rows[_key(found)]
        params = {name: row[name] for name in proposals.PARAM_NAMES if row[name] is not None}
        assert (row["status"], row["error"], row["template"], params, row["allow_short"]) == expected, found
        assert (row["prompt_version"], row["requested_model"], row["job_id"]) == (
            proposals.PROMPT_VERSION, ai_settings.DEFAULT_MODEL, job["id"])
        if row["status"] == "out_of_scope":
            assert (row["input_sha256"], row["response_id"], row["input_tokens"], row["output_tokens"]) == (
                None, None, 0, 0)
            continue
        asked += 1
        prompt = proposals.prompt_for(found)
        assert (row["input_sha256"], row["response_model"]) == (prompt.input_sha256, ai_settings.DEFAULT_MODEL)
        assert row["response_id"] == "fake_" + hashlib.sha256(prompt.user_message.encode()).hexdigest()[:24]
        if row["status"] == "proposed":
            assert row["rationale"] == f"The hypothesis describes a {row['template']} signal."
    assert rows[_key(hyp("decline"))]["rationale"] == "Declined, as asked."
    result = job["result"]
    statuses = collections.Counter(expected[0] for _, expected in CASES)
    assert {k: result[k] for k in proposals.STATUSES} == {k: statuses[k] for k in proposals.STATUSES}
    assert (result["hypotheses"], result["duplicates"], result["caught_up"], result["budget_exhausted"],
            result["rate_limited"], result["retries"]) == (len(CASES), 0, True, False, False, 0)
    assert (result["provider"], result["prompt_version"]) == ("fake", proposals.PROMPT_VERSION)
    reservations = prop.reservations()
    assert [r["status"] for r in reservations] == ["settled"] * asked
    assert all((r["job_id"], r["attempt"], r["output_reserved"]) == (job["id"], 1, proposals.MAX_OUTPUT_TOKENS)
               for r in reservations)
    assert all(r["input_used"] < r["input_reserved"] for r in reservations)
    # What the provider reports is settled as it is, even past the reservation.
    assert rows[_key(hyp("overspend"))]["output_tokens"] == proposals.MAX_OUTPUT_TOKENS + 500
    assert max(r["output_used"] for r in reservations) == proposals.MAX_OUTPUT_TOKENS + 500
    assert result["input_tokens"] == sum(r["input_used"] for r in reservations) == \
        sum(r["input_tokens"] for r in rows.values())
    assert result["output_tokens"] == sum(r["output_used"] for r in reservations)


def test_a_paper_is_extracted_then_its_hypothesis_proposed_by_the_two_stages(prop):
    prop.ai.flag("extract", True)
    sentence = "A 50-day moving average crossover earns a Sharpe ratio of 0.85 on commodity futures."
    prop.plant(paper(1, abstract=sentence))
    job_id = prop.ai.enqueue({}, kind="extract")
    runner = prop.worker()
    try:
        assert runner.run_once() == "succeeded", prop.ai.job(job_id)
    finally:
        runner.close()
    ended, job = prop.run()
    assert ended == "succeeded", job
    [row] = prop.proposals()
    assert (row["evidence_quote"], row["signal_family"], row["status"], row["template"]) == (
        sentence, "ma_crossover", "proposed", "ma_crossover")
    assert (row["fast"], row["slow"]) == (20, 50)


def test_each_hypothesis_is_asked_about_once_per_prompt_and_model_newest_extraction_first(prop):
    prop.extraction(1, hyp(family="momentum"))
    newest = prop.extraction(2, hyp(), hyp(family="value"))
    ended, job = prop.run({"max_hypotheses": 1})
    assert ended == "succeeded" and (job["result"]["hypotheses"], job["result"]["caught_up"]) == (1, False)
    assert [(r["extraction_id"], r["ordinal"]) for r in prop.proposals()] == [(newest, 1)]
    ended, job = prop.run()
    assert ended == "succeeded" and (job["result"]["hypotheses"], job["result"]["caught_up"]) == (2, True)
    ended, job = prop.run()
    assert ended == "succeeded" and (job["result"]["hypotheses"], job["result"]["caught_up"]) == (0, True)
    assert len(prop.proposals()) == 3 and len(prop.reservations()) == 2
    # Another model asks only about the hypotheses its own extractions found.
    ended, job = prop.run(AI_MODEL="claude-sonnet-5-5")
    assert ended == "succeeded" and (job["result"]["hypotheses"], job["result"]["caught_up"]) == (0, True)
    prop.extraction(1, hyp(family="mean_reversion"), model="claude-sonnet-5-5")
    ended, job = prop.run(AI_MODEL="claude-sonnet-5-5")
    assert ended == "succeeded" and job["result"]["proposed"] == 1
    assert [r["requested_model"] for r in prop.proposals()][-1] == "claude-sonnet-5-5"
    assert len(prop.reservations()) == 3


def test_only_the_hypotheses_of_current_extractions_are_asked_about(prop):
    """Not those of an earlier version of a paper, of a deleted paper, or of
    an earlier hypothesis prompt: the extract driver no longer counts them."""
    prop.extraction(1, hyp())
    prop.plant(paper(1, abstract=SENTENCE + " We also test equities."))  # paper 1 changes
    prop.extraction(2, hyp())
    prop.plant({"identifier": "oai:arXiv.org:2611.00002", "datestamp": "2026-11-20", "deleted": True})
    prop.extraction(3, hyp(), version=OLD_PROMPT)
    assert len(prop.rows("SELECT 1 FROM {ai}.hypotheses")) == 3
    ended, job = prop.run()
    assert ended == "succeeded" and (job["result"]["hypotheses"], job["result"]["caught_up"]) == (0, True)
    assert prop.proposals() == [] and prop.reservations() == []
    current = prop.extraction(1, hyp(family="momentum"))  # the changed paper, extracted again
    ended, job = prop.run()
    assert ended == "succeeded" and job["result"]["hypotheses"] == 1
    assert [r["extraction_id"] for r in prop.proposals()] == [current]


def test_a_hypothesis_no_template_expresses_costs_no_request_and_no_tokens(prop):
    prop.extraction(1, hyp(family="value"), hyp("crash", family="order_book_imbalance"))
    ended, job = prop.run(AI_DAILY_INPUT_TOKENS=None, AI_DAILY_OUTPUT_TOKENS=None)
    assert ended == "succeeded", job
    assert (job["result"]["hypotheses"], job["result"]["out_of_scope"], job["result"]["caught_up"],
            job["result"]["budget_exhausted"]) == (2, 2, True, False)
    assert prop.reservations() == [] and [r["status"] for r in prop.proposals()] == ["out_of_scope"] * 2


def test_no_budget_no_request(prop):
    prop.extraction(1, hyp(family="value"), hyp())
    ended, job = prop.run(AI_DAILY_INPUT_TOKENS=None, AI_DAILY_OUTPUT_TOKENS=None)
    assert ended == "succeeded", job
    assert (job["result"]["budget_exhausted"], job["result"]["hypotheses"], job["result"]["caught_up"]) == (
        True, 1, False)
    assert prop.reservations() == [] and [r["status"] for r in prop.proposals()] == ["out_of_scope"]


def test_an_overspent_answer_counts_as_reported_and_can_end_the_days_budget(prop):
    prop.extraction(1, hyp())
    prop.extraction(2, hyp("overspend"))
    ended, job = prop.run(AI_DAILY_OUTPUT_TOKENS=str(proposals.MAX_OUTPUT_TOKENS + 400))
    assert ended == "succeeded", job
    assert (job["result"]["hypotheses"], job["result"]["budget_exhausted"]) == (1, True)
    assert [r["statement"] for r in prop.proposals()] == [STATEMENT + " [[fake:overspend]]"]
    assert len(prop.reservations()) == 1


def test_without_a_provider_the_job_fails(prop):
    prop.extraction(1, hyp())
    ended, job = prop.run(AI_MODEL_PROVIDER=None)
    assert ended == "failed" and job["error"] == \
        "The strategy proposal failed: no model provider is configured (AI_MODEL_PROVIDER)"
    assert prop.reservations() == [] and prop.proposals() == []


def test_a_bad_payload_fails_the_job(prop):
    ended, job = prop.run({"max_hypotheses": 0})
    assert ended == "failed" and "max_hypotheses must be" in job["error"]


def test_a_busy_provider_is_retried_and_settled_with_nothing_each_time_then_fails_naming_the_hypothesis(
        prop, quick):
    extraction_id = prop.extraction(1, hyp("busy"))
    ended, job = prop.run()
    assert ended == "failed" and job["error"] == (
        "The strategy proposal failed: the model provider answered HTTP 529"
        f" (2 retries in a row, at hypothesis 1 of extraction {extraction_id})")
    assert [(r["status"], r["input_used"], r["output_used"]) for r in prop.reservations()] == [("settled", 0, 0)] * 3
    assert prop.proposals() == []


def test_a_long_rate_limit_ends_the_job_without_failing_it(prop):
    prop.extraction(1, hyp())
    prop.extraction(2, hyp("rate_limited"))
    ended, job = prop.run()
    assert ended == "succeeded", job
    assert (job["result"]["rate_limited"], job["result"]["hypotheses"], job["result"]["caught_up"]) == (
        True, 0, False)
    assert [(r["status"], r["output_used"]) for r in prop.reservations()] == [("settled", 0)]
    assert prop.proposals() == []


def test_a_refused_request_fails_the_job(prop):
    prop.extraction(1, hyp("bad_request"))
    ended, job = prop.run()
    assert ended == "failed" and job["error"] == "The strategy proposal failed: the model provider answered HTTP 400"
    assert [r["status"] for r in prop.reservations()] == ["settled"] and prop.proposals() == []


def test_a_reply_with_impossible_usage_is_refused_abandoned_and_fails_the_job(prop):
    extraction_id = prop.extraction(1, hyp("bad_usage"))
    ended, job = prop.run()
    assert ended == "failed", job
    assert job["error"].startswith("The strategy proposal failed: the reply about hypothesis 1 of extraction"
                                   f" {extraction_id} did not pass the checks: its usage")
    assert [r["status"] for r in prop.reservations()] == ["orphaned"] and prop.proposals() == []


def test_a_crashing_request_counts_in_full_each_time_and_fails_naming_the_hypothesis(prop, quick):
    extraction_id = prop.extraction(1, hyp(family="value"), hyp("crash"))
    ended, job = prop.run()
    assert ended == "failed", job
    assert job["error"].startswith("The strategy proposal failed: the request failed:")
    assert job["error"].endswith(f"(2 retries in a row, at hypothesis 2 of extraction {extraction_id})")
    assert "crashed, as asked" in job["error"]
    assert [r["status"] for r in prop.reservations()] == ["orphaned"] * 3
    assert [r["status"] for r in prop.proposals()] == ["out_of_scope"]
    with prop.ai.conn.transaction():
        assert ledger.spent_today(prop.ai.conn, prop.ai.schema)[1] == 3 * proposals.MAX_OUTPUT_TOKENS


def test_a_hanging_request_is_stopped_at_its_deadline_and_counts_in_full(prop, monkeypatch):
    monkeypatch.setattr(extract, "MAX_RETRIES", 0)
    monkeypatch.setitem(KINDS, "propose", dataclasses.replace(KINDS["propose"], child_timeout_seconds=3))
    prop.extraction(1, hyp("hang"))
    ended, job = prop.run()
    assert ended == "failed", job
    assert "ran longer than 3 s" in job["error"]
    assert [r["status"] for r in prop.reservations()] == ["orphaned"] and prop.proposals() == []


def test_a_request_that_failed_once_is_retried_and_its_first_reservation_still_counts(prop, monkeypatch):
    monkeypatch.setattr(extract, "RETRY_SECONDS", 0)
    real, calls = AiWorker.run_child, []

    def flaky(self, *args):
        calls.append(args)
        if len(calls) == 1:
            raise child.JobFailed("the network went away")
        return real(self, *args)

    monkeypatch.setattr(AiWorker, "run_child", flaky)
    prop.extraction(1, hyp())
    ended, job = prop.run()
    assert ended == "succeeded", job
    assert (job["result"]["retries"], job["result"]["hypotheses"], job["result"]["proposed"]) == (1, 1, 1)
    assert [r["status"] for r in prop.reservations()] == ["orphaned", "settled"]
    assert len(calls) == 2 and calls[0][2] == calls[1][2]
    assert set(calls[0][2]) == {"provider", "model", "user_message", "timeout_seconds", "task"}
    assert calls[0][2]["task"] == proposals.TASK


def test_switching_the_stage_off_mid_request_abandons_the_reservation_and_hands_the_job_back(prop, monkeypatch):
    real = AiWorker.run_child

    def switch_off(self, *args):
        prop.ai.flag("propose", False)
        return real(self, *args)

    monkeypatch.setattr(AiWorker, "run_child", switch_off)
    prop.extraction(1, hyp("hang"))
    ended, job = prop.run()
    assert ended == "released (stage switched off)"
    assert (job["status"], job["attempts"]) == ("queued", 0)
    assert [r["status"] for r in prop.reservations()] == ["orphaned"] and prop.proposals() == []


def test_a_worker_that_lost_its_job_records_nothing_and_the_reaper_counts_its_request(prop, monkeypatch):
    real = AiWorker.run_child

    def steal(self, *args):
        out = real(self, *args)
        prop.sql("UPDATE {ai}.jobs SET worker_id = 'thief' WHERE kind = 'propose'")
        return out

    monkeypatch.setattr(AiWorker, "run_child", steal)
    prop.extraction(1, hyp())
    ended, job = prop.run()
    assert ended == "lost (lease lapsed)" and job["status"] == "running"
    assert prop.proposals() == [] and [r["status"] for r in prop.reservations()] == ["reserved"]
    with prop.ai.conn.transaction():
        assert ledger.reclaim_orphans(prop.ai.conn, prop.ai.schema) == 1
    assert [r["status"] for r in prop.reservations()] == ["orphaned"]


def test_a_worker_that_lost_its_job_records_no_out_of_scope_hypothesis_either(prop, monkeypatch):
    real = propose._Propose.due

    def steal(self):
        rows = real(self)
        prop.sql("UPDATE {ai}.jobs SET worker_id = 'thief' WHERE kind = 'propose'")
        return rows

    monkeypatch.setattr(propose._Propose, "due", steal)
    prop.extraction(1, hyp(family="value"))
    ended, job = prop.run()
    assert ended == "lost (lease lapsed)" and prop.proposals() == []


def test_a_hypothesis_answered_twice_is_recorded_once_and_counted_as_a_duplicate(prop, monkeypatch):
    """As when two workers' runs overlap: the second answer still counts its tokens."""
    real, served = propose._Propose.due, []

    def twice(self):
        rows = real(self)
        served.append(rows)
        return rows * 2 if len(served) == 1 else rows

    monkeypatch.setattr(propose._Propose, "due", twice)
    prop.extraction(1, hyp(family="value"), hyp())
    ended, job = prop.run()
    assert ended == "succeeded", job
    assert (job["result"]["hypotheses"], job["result"]["out_of_scope"], job["result"]["proposed"],
            job["result"]["duplicates"]) == (4, 1, 1, 2)
    assert len(prop.proposals()) == 2 and [r["status"] for r in prop.reservations()] == ["settled"] * 2


def test_a_proposal_job_stops_at_the_kinds_deadline(prop, monkeypatch):
    monkeypatch.setitem(KINDS, "propose", dataclasses.replace(KINDS["propose"], timeout_seconds=0))
    prop.extraction(1, hyp(family="value"), hyp())
    ended, job = prop.run()
    assert ended == "succeeded" and (job["result"]["hypotheses"], job["result"]["caught_up"]) == (0, False)
    assert prop.reservations() == [] and prop.proposals() == []


def test_a_job_process_in_production_refuses_the_fake_whatever_its_worker_was_told(prop, monkeypatch):
    monkeypatch.setattr(extract, "MAX_RETRIES", 0)
    monkeypatch.setenv("ENVIRONMENT", "production")
    prop.extraction(1, hyp())
    ended, job = prop.run()
    assert ended == "failed" and "never runs in production" in job["error"], job
    assert [r["status"] for r in prop.reservations()] == ["orphaned"] and prop.proposals() == []


def test_one_proposal_job_at_a_time_beside_an_extraction(prop):
    prop.ai.enqueue({}, kind="propose")
    with pytest.raises(queue.EnqueueError, match="already queued or running"):
        prop.ai.enqueue({"max_hypotheses": 5}, kind="propose")
    prop.ai.flag("extract", True)
    prop.ai.enqueue({}, kind="extract")


def test_proposals_are_append_only(prop):
    prop.extraction(1, hyp(family="value"))
    prop.run()
    for statement in ("UPDATE {ai}.proposals SET status = 'proposed'", "DELETE FROM {ai}.proposals",
                      "TRUNCATE {ai}.proposals"):
        with pytest.raises(Exception, match="is refused: its rows are immutable"), prop.ai.conn.transaction():
            prop.sql(statement)
    assert len(prop.proposals()) == 1


def test_the_cli_checks_the_hypothesis_limit_before_it_queues(capsys):
    assert manage._ai_propose(None, 0, "ops@example.com") == 1
    assert "max_hypotheses must be" in capsys.readouterr().err


def test_every_proposal_fake_behaviour_is_shown_working():
    directed = {d for found, _ in CASES if found["signal_family"] in proposals.TEMPLATES
                for d in models._DIRECTIVE.findall(found["statement"] + " " + found["evidence_quote"])}
    failures = {"busy", "rate_limited", "bad_request", "bad_usage", "crash", "hang"}
    assert directed | failures == set(models.PROPOSAL_FAKE_BEHAVIOURS)
