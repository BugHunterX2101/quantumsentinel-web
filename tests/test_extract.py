"""Hypothesis extraction end to end (backend/ai/extract.py, backend/ai/models.py, migration 0017).

The worker runs each extract job as it does in production: the driver in
the worker, each request in a job process, against the adversarial fake
model. A paper's abstract can tell the fake to misbehave
(``[[fake:<behaviour>]]``) as a model the paper has subverted, or a failing
provider, would. The tests show every such answer recorded for what it is,
and every request ending in exactly one ledger action: settled with what
the provider reports, or abandoned (counting in full) when its fate is
unknown. The role privileges are tested in tests/test_db_roles.py.
"""
import dataclasses
import hashlib

import pytest

from backend import manage
from backend.ai import child, documents, extract, hypotheses, ledger, models, queue
from backend.ai import settings as ai_settings
from backend.ai.documents import clean_page
from backend.ai.kinds import KINDS
from backend.ai.worker import AiWorker
from test_ai_tier import Ai, ai  # noqa: F401 - the ai fixture
from test_arxiv_registry import page, raw

TITLE = "Momentum in Commodity Futures"
SENTENCE = "We find that a 12-month momentum strategy earns a Sharpe ratio of 0.85 over 1990-2010."
BUDGET = {"AI_DAILY_INPUT_TOKENS": str(10 ** 9), "AI_DAILY_OUTPUT_TOKENS": str(10 ** 9)}
FAKE = {"AI_MODEL_PROVIDER": "fake", **BUDGET}


def paper(number: int, behaviour: str | None = None, *, title: str = TITLE, abstract: str | None = None) -> dict:
    """A paper, newer the higher its number; its abstract carries the directive, if any."""
    if abstract is None:
        abstract = SENTENCE + (f" [[fake:{behaviour}]]" if behaviour else "")
    return raw(id=f"2611.{number:05d}", title=title, abstract=abstract, datestamp=f"2026-10-{number:02d}")


class Ext:
    def __init__(self, ai: Ai):
        self.ai = ai

    def sql(self, statement: str, params=()):
        return self.ai.conn.execute(statement.replace("{ai}", f'"{self.ai.schema}"'), params)

    def plant(self, *records) -> None:
        with self.ai.conn.transaction():
            documents.upsert_page(self.ai.conn, self.ai.schema, clean_page(page(*records)))

    def worker(self, **environ) -> AiWorker:
        merged = {**FAKE, **environ}
        return self.ai.worker(**{k: v for k, v in merged.items() if v is not None})

    def run(self, payload=None, **environ) -> tuple[str, dict]:
        job_id = self.ai.enqueue(payload or {}, kind="extract")
        runner = self.worker(**environ)
        try:
            ended = runner.run_once()
        finally:
            runner.close()
        return ended, self.ai.job(job_id)

    def rows(self, query: str, params=()) -> list[dict]:
        cursor = self.sql(query, params)
        names = [c.name for c in cursor.description]
        return [dict(zip(names, row)) for row in cursor.fetchall()]

    def extractions(self) -> dict[str, dict]:
        return {r["source_id"]: r for r in self.rows(
            "SELECT d.source_id, d.title, d.abstract, e.* FROM {ai}.extractions e"
            " JOIN {ai}.documents d ON d.id = e.document_id")}

    def reservations(self) -> list[dict]:
        return self.rows("SELECT * FROM {ai}.token_reservations ORDER BY created_at, id")

    def document_id(self, number: int) -> str:
        return self.sql("SELECT id FROM {ai}.documents WHERE source_id = %s", (f"2611.{number:05d}",)).fetchone()[0]


@pytest.fixture
def ext(ai):
    """The extract stage on, and no papers yet."""
    ai.switch(True, True, "extract")
    return Ext(ai)


@pytest.fixture
def quick(monkeypatch):
    """Retries without waiting, and at most two in a row."""
    monkeypatch.setattr(extract, "RETRY_SECONDS", 0)
    monkeypatch.setattr(extract, "MAX_RETRIES", 2)


# -- the job process's request -----------------------------------------------------

def payload(**overrides) -> dict:
    return {"provider": "fake", "model": "claude-opus-5-5", "timeout_seconds": 150,
            "user_message": hypotheses.prompt_for(TITLE, SENTENCE).user_message, **overrides}


def test_the_request_takes_the_prompt_schema_and_limit_from_the_code_never_the_payload():
    provider, request = models.request_from(payload(api_key="test-key-0123"))  # gitleaks:allow - fake
    assert provider == "fake" and request.api_key == "test-key-0123"  # gitleaks:allow - fake
    assert (request.system, request.output_schema, request.max_tokens) == (
        hypotheses.SYSTEM_PROMPT, hypotheses.OUTPUT_SCHEMA, hypotheses.MAX_OUTPUT_TOKENS)
    assert "test-key" not in repr(request)
    for key in ("system", "output_schema", "max_tokens", "tools"):
        with pytest.raises(ValueError, match="unknown payload keys"):
            models.request_from(payload(**{key: "x"}))


@pytest.mark.parametrize("overrides, message", [
    ({"provider": "anthropic"}, "provider must be one of"), ({"provider": None}, "provider must be one of"),
    ({"model": "Claude Opus"}, "model id"), ({"model": ""}, "model id"), ({"model": 5}, "model id"),
    ({"user_message": ""}, "user_message"), ({"user_message": "x" * (models.MAX_USER_MESSAGE + 1)}, "user_message"),
    ({"timeout_seconds": 0}, "timeout_seconds"), ({"timeout_seconds": 601}, "timeout_seconds"),
    ({"timeout_seconds": True}, "timeout_seconds"), ({"timeout_seconds": "150"}, "timeout_seconds"),
    ({"api_key": 5}, "api_key"),
])
def test_the_payload_is_checked(overrides, message):
    with pytest.raises(ValueError, match=message):
        models.request_from(payload(**overrides))
    with pytest.raises(ValueError):
        models.request_from(["not", "an", "object"])


def test_the_fake_never_runs_in_production(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "Production")
    with pytest.raises(PermissionError, match="never runs in production"):
        models.answer(payload())
    with pytest.raises(ai_settings.SettingsError, match="never production"):
        ai_settings.load({"QS_PROCESS_ROLE": "ai-worker", "ENVIRONMENT": "production",
                          "AI_DATABASE_URL": "postgresql+psycopg://u:p@db/x", "AI_MODEL_PROVIDER": "fake"})


@pytest.mark.parametrize("environ, message", [
    ({"AI_MODEL_PROVIDER": "anthropic"}, "AI_MODEL_PROVIDER must be one of"),
    ({"AI_MODEL": "Opus!"}, "AI_MODEL must be a model id"),
    ({"AI_DAILY_INPUT_TOKENS": "-1"}, "AI_DAILY_INPUT_TOKENS"),
    ({"AI_DAILY_OUTPUT_TOKENS": str(10 ** 9 + 1)}, "AI_DAILY_OUTPUT_TOKENS"),
    ({"AI_DAILY_OUTPUT_TOKENS": "1e6"}, "AI_DAILY_OUTPUT_TOKENS"),
])
def test_the_model_settings_are_checked(environ, message):
    with pytest.raises(ai_settings.SettingsError, match=message):
        ai_settings.load({"QS_PROCESS_ROLE": "ai-worker", "AI_DATABASE_URL": "postgresql+psycopg://u:p@db/x",
                          **environ})


def test_the_model_settings_default_to_no_provider_and_no_budget():
    settings = ai_settings.load({"QS_PROCESS_ROLE": "ai-worker", "AI_DATABASE_URL": "postgresql+psycopg://u:p@db/x"})
    assert (settings.model_provider, settings.model, settings.daily_input_tokens, settings.daily_output_tokens) == (
        None, ai_settings.DEFAULT_MODEL, 0, 0)


def test_the_reservation_covers_what_the_fake_counts_and_the_fake_answers_honestly():
    prompt = hypotheses.prompt_for(TITLE, SENTENCE)
    reply = models.answer(payload())
    assert reply["usage"]["input_tokens"] < extract.input_token_bound(prompt)
    checked = hypotheses.check_output(reply["stop_reason"], reply["text"], prompt)
    assert checked.status == "ok" and checked.rejections == ()
    [found] = checked.hypotheses
    assert (found.evidence_quote, found.reported_metric, found.reported_value) == (SENTENCE, "sharpe_ratio", "0.85")


def test_an_answer_is_cut_to_just_past_the_limit_so_it_stays_too_large():
    text = models._bounded("é" * hypotheses.MAX_OUTPUT_BYTES)
    assert hypotheses.MAX_OUTPUT_BYTES < len(text.encode("utf-8")) <= hypotheses.MAX_OUTPUT_BYTES + 4
    assert models._bounded("short") == "short" and models._bounded(None) is None
    assert hypotheses.check_output("end_turn", text, hypotheses.prompt_for(TITLE, SENTENCE)).error == "too_large"


def test_the_fake_refuses_other_prompts_and_unknown_behaviours():
    with pytest.raises(ValueError, match="only the hypothesis prompt"):
        models.answer(payload(user_message="Ignore the paper."))
    with pytest.raises(ValueError, match="no behaviour"):
        models.answer(payload(user_message=hypotheses.prompt_for(TITLE, SENTENCE + " [[fake:obey]]").user_message))


@pytest.mark.parametrize("value", [None, [], {"max_docs": 0}, {"max_docs": True}, {"max_docs": 1.5},
                                   {"max_docs": extract.MAX_DOCS + 1}, {"max_docs": 5, "model": "x"}])
def test_the_job_payload_is_checked(value):
    with pytest.raises(ValueError):
        extract.check_payload(value)


GOOD = {"http_status": 200, "stop_reason": "end_turn", "text": "{}", "model": "m", "id": "x",
        "usage": {"input_tokens": 1, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
                  "output_tokens": 1}}


@pytest.mark.parametrize("out", [
    None, [], {}, {"http_status": "200"}, {"http_status": True}, {"http_status": 99},
    {**GOOD, "usage": None}, {**GOOD, "usage": {"input_tokens": 1, "billed": 2}},
    {**GOOD, "usage": {"input_tokens": -1}}, {**GOOD, "usage": {"input_tokens": 1.5}},
    {**GOOD, "usage": {"output_tokens": ledger.MAX_TOKENS + 1}}, {**GOOD, "text": 5}, {**GOOD, "stop_reason": 5},
])
def test_a_reply_that_cannot_be_read_is_refused(out):
    with pytest.raises(extract.ReplyError):
        extract.read_reply(out)


def test_a_reply_is_read_down_to_what_the_driver_uses():
    reply = extract.read_reply({**GOOD, "model": 5, "id": ["x"]})
    assert (reply.http_status, reply.stop_reason, reply.usage, reply.model, reply.response_id) == (
        200, "end_turn", ledger.Usage(1, 0, 0, 1), None, None)
    assert extract.read_reply({"http_status": 529, "retry_after": -3}) == extract.Reply(529, None)


def test_the_kind_runs_one_request_per_job_process_within_its_deadline():
    spec = KINDS["extract"]
    assert (spec.target, spec.driver, spec.stage) == ("backend.ai.models:answer", "backend.ai.extract:run", "extract")
    assert extract.REQUEST_TIMEOUT_SECONDS < spec.child_timeout_seconds < spec.timeout_seconds
    assert spec.child_output_max_bytes > hypotheses.MAX_OUTPUT_BYTES * 6


# -- the driver, end to end --------------------------------------------------------

OUTCOMES = {  # behaviour: (status, error, hypotheses, rejections)
    None: ("ok", None, 1, []),
    "invent_quote": ("ok", None, 0, [{"ordinal": 1, "reason": "quote_not_found"}]),
    "invent_number": ("ok", None, 0, [{"ordinal": 1, "reason": "value_not_in_quote"}]),
    "certify": ("invalid", "wrong_keys", 0, []),
    "duplicate_key": ("invalid", "duplicate_key", 0, []),
    "flood": ("invalid", "too_many_hypotheses", 0, []),
    "oversize": ("invalid", "too_large", 0, []),
    "not_json": ("invalid", "not_json", 0, []),
    "truncate": ("truncated", None, 0, []),
    "refuse": ("refused", None, 0, []),
    "odd_stop": ("unexpected_stop", None, 0, []),
    "overspend": ("ok", None, 1, []),
}


def test_every_answer_a_subverted_model_gives_is_recorded_for_what_it_is_and_settled_as_reported(ext):
    behaviours = list(OUTCOMES)
    ext.plant(*(paper(n, b) for n, b in enumerate(behaviours, start=1)))
    ended, job = ext.run()
    assert ended == "succeeded", job
    rows = ext.extractions()
    assert len(rows) == len(OUTCOMES)
    for n, behaviour in enumerate(behaviours, start=1):
        row = rows[f"2611.{n:05d}"]
        assert (row["status"], row["error"], row["hypothesis_count"], row["rejections"]) == OUTCOMES[behaviour], \
            behaviour
        prompt = hypotheses.prompt_for(row["title"], row["abstract"])
        assert (row["input_sha256"], row["prompt_version"]) == (prompt.input_sha256, hypotheses.PROMPT_VERSION)
        assert (row["requested_model"], row["response_model"]) == (ai_settings.DEFAULT_MODEL,) * 2
        assert row["response_id"] == "fake_" + hashlib.sha256(prompt.user_message.encode()).hexdigest()[:24]
        assert row["job_id"] == job["id"]
    stored = ext.rows("SELECT evidence_quote, reported_metric, reported_value FROM {ai}.hypotheses")
    assert stored == [{"evidence_quote": SENTENCE, "reported_metric": "sharpe_ratio", "reported_value": "0.85"}] * 2
    result = job["result"]
    assert (result["documents"], result["ok"], result["invalid"], result["truncated"], result["refused"],
            result["unexpected_stop"], result["hypotheses"], result["rejected"], result["duplicates"]) == (
        12, 4, 5, 1, 1, 1, 2, 2, 0)
    assert (result["caught_up"], result["budget_exhausted"], result["rate_limited"], result["retries"]) == (
        True, False, False, 0)
    assert (result["provider"], result["prompt_version"]) == ("fake", hypotheses.PROMPT_VERSION)
    reservations = ext.reservations()
    assert [r["status"] for r in reservations] == ["settled"] * 12
    assert all((r["job_id"], r["attempt"], r["output_reserved"]) == (job["id"], 1, hypotheses.MAX_OUTPUT_TOKENS)
               for r in reservations)
    assert all(r["input_used"] < r["input_reserved"] for r in reservations)
    # What the provider reports is settled as it is, even past the reservation.
    overspent = rows[f"2611.{behaviours.index('overspend') + 1:05d}"]
    assert overspent["output_tokens"] == hypotheses.MAX_OUTPUT_TOKENS + 500
    assert max(r["output_used"] for r in reservations) == hypotheses.MAX_OUTPUT_TOKENS + 500
    assert result["input_tokens"] == sum(r["input_used"] for r in reservations) == \
        sum(r["input_tokens"] for r in rows.values())
    assert result["output_tokens"] == sum(r["output_used"] for r in reservations)
    with ext.ai.conn.transaction():
        assert ledger.spent_today(ext.ai.conn, ext.ai.schema) == (result["input_tokens"], result["output_tokens"])


def test_each_paper_is_asked_about_once_per_version_prompt_and_model(ext):
    ext.plant(paper(1), paper(2))
    ended, job = ext.run({"max_docs": 1})
    assert ended == "succeeded" and (job["result"]["documents"], job["result"]["caught_up"]) == (1, False)
    assert list(ext.extractions()) == ["2611.00002"]  # newest first
    ended, job = ext.run()
    assert ended == "succeeded" and (job["result"]["documents"], job["result"]["caught_up"]) == (1, True)
    ended, job = ext.run()
    assert ended == "succeeded" and (job["result"]["documents"], job["result"]["caught_up"]) == (0, True)
    assert len(ext.reservations()) == 2
    # A changed paper, or another model, is asked about again.
    ext.plant(paper(1, abstract=SENTENCE + " We also test equities."))
    ended, job = ext.run(AI_MODEL="claude-sonnet-5-5")
    assert ended == "succeeded" and job["result"]["documents"] == 2
    ended, job = ext.run()
    assert ended == "succeeded" and job["result"]["documents"] == 1
    assert len(ext.rows("SELECT id FROM {ai}.extractions")) == 5 and len(ext.reservations()) == 5


def test_a_paper_that_cannot_be_sent_is_skipped_without_a_request(ext):
    ext.plant(paper(1), paper(2, title="͸️"))
    assert ext.sql("SELECT title FROM {ai}.documents WHERE source_id = '2611.00002'").fetchone()[0] == "͸️"
    ended, job = ext.run()
    assert ended == "succeeded", job
    assert (job["result"]["documents"], job["result"]["unsendable"], job["result"]["caught_up"]) == (1, 1, True)
    assert list(ext.extractions()) == ["2611.00001"] and len(ext.reservations()) == 1


def test_no_budget_no_request(ext):
    ext.plant(paper(1))
    ended, job = ext.run(AI_DAILY_INPUT_TOKENS=None, AI_DAILY_OUTPUT_TOKENS=None)
    assert ended == "succeeded", job
    assert (job["result"]["budget_exhausted"], job["result"]["documents"], job["result"]["caught_up"]) == (
        True, 0, False)
    assert ext.reservations() == [] and ext.extractions() == {}


def test_an_overspent_answer_counts_as_reported_and_can_end_the_days_budget(ext):
    ext.plant(paper(2, "overspend"), paper(1))
    ended, job = ext.run(AI_DAILY_OUTPUT_TOKENS=str(hypotheses.MAX_OUTPUT_TOKENS + 400))
    assert ended == "succeeded", job
    assert (job["result"]["documents"], job["result"]["budget_exhausted"]) == (1, True)
    assert list(ext.extractions()) == ["2611.00002"] and len(ext.reservations()) == 1


def test_without_a_provider_the_job_fails(ext):
    ext.plant(paper(1))
    ended, job = ext.run(AI_MODEL_PROVIDER=None)
    assert ended == "failed" and job["error"] == \
        "The hypothesis extraction failed: no model provider is configured (AI_MODEL_PROVIDER)"
    assert ext.reservations() == []


def test_a_bad_payload_fails_the_job(ext):
    ended, job = ext.run({"max_docs": 0})
    assert ended == "failed" and "max_docs must be" in job["error"]


def test_a_busy_provider_is_retried_and_settled_with_nothing_each_time_then_fails_naming_the_paper(ext, quick):
    ext.plant(paper(1, "busy"))
    ended, job = ext.run()
    assert ended == "failed" and job["error"] == (
        "The hypothesis extraction failed: the model provider answered HTTP 529"
        f" (2 retries in a row, at document {ext.document_id(1)})")
    reservations = ext.reservations()
    assert [(r["status"], r["input_used"], r["output_used"]) for r in reservations] == [("settled", 0, 0)] * 3
    assert ext.extractions() == {}


def test_a_long_rate_limit_ends_the_job_without_failing_it(ext):
    ext.plant(paper(2, "rate_limited"), paper(1))
    ended, job = ext.run()
    assert ended == "succeeded", job
    assert (job["result"]["rate_limited"], job["result"]["documents"], job["result"]["caught_up"]) == (True, 0, False)
    assert [(r["status"], r["output_used"]) for r in ext.reservations()] == [("settled", 0)]
    assert ext.extractions() == {}


def test_a_refused_request_fails_the_job(ext):
    ext.plant(paper(1, "bad_request"))
    ended, job = ext.run()
    assert ended == "failed" and job["error"] == "The hypothesis extraction failed: the model provider answered HTTP 400"
    assert [r["status"] for r in ext.reservations()] == ["settled"] and ext.extractions() == {}


def test_a_reply_with_impossible_usage_is_refused_abandoned_and_fails_the_job(ext):
    ext.plant(paper(1, "bad_usage"))
    ended, job = ext.run()
    assert ended == "failed", job
    assert job["error"].startswith("The hypothesis extraction failed: the reply about document"
                                   f" {ext.document_id(1)} did not pass the checks: its usage")
    assert [r["status"] for r in ext.reservations()] == ["orphaned"] and ext.extractions() == {}


def test_a_crashing_request_counts_in_full_each_time_and_fails_naming_the_paper(ext, quick):
    ext.plant(paper(1, "crash"))
    ended, job = ext.run()
    assert ended == "failed", job
    assert job["error"].startswith("The hypothesis extraction failed: the request failed:")
    assert job["error"].endswith(f"(2 retries in a row, at document {ext.document_id(1)})")
    assert "crashed, as asked" in job["error"]
    assert [r["status"] for r in ext.reservations()] == ["orphaned"] * 3
    assert ext.extractions() == {}
    with ext.ai.conn.transaction():
        assert ledger.spent_today(ext.ai.conn, ext.ai.schema)[1] == 3 * hypotheses.MAX_OUTPUT_TOKENS


def test_a_hanging_request_is_stopped_at_its_deadline_and_counts_in_full(ext, monkeypatch):
    monkeypatch.setattr(extract, "MAX_RETRIES", 0)
    monkeypatch.setitem(KINDS, "extract", dataclasses.replace(KINDS["extract"], child_timeout_seconds=3))
    ext.plant(paper(1, "hang"))
    ended, job = ext.run()
    assert ended == "failed", job
    assert "ran longer than 3 s" in job["error"]
    assert [r["status"] for r in ext.reservations()] == ["orphaned"] and ext.extractions() == {}


def test_a_request_that_failed_once_is_retried_and_its_first_reservation_still_counts(ext, monkeypatch):
    monkeypatch.setattr(extract, "RETRY_SECONDS", 0)
    real, calls = AiWorker.run_child, []

    def flaky(self, *args):
        calls.append(args)
        if len(calls) == 1:
            raise child.JobFailed("the network went away")
        return real(self, *args)

    monkeypatch.setattr(AiWorker, "run_child", flaky)
    ext.plant(paper(1))
    ended, job = ext.run()
    assert ended == "succeeded", job
    assert (job["result"]["retries"], job["result"]["documents"], job["result"]["ok"]) == (1, 1, 1)
    assert [r["status"] for r in ext.reservations()] == ["orphaned", "settled"]
    assert len(calls) == 2 and calls[0][2] == calls[1][2]
    assert set(calls[0][2]) == {"provider", "model", "user_message", "timeout_seconds"}


def test_the_retries_in_a_row_start_again_after_an_answer(ext, monkeypatch):
    monkeypatch.setattr(extract, "RETRY_SECONDS", 0)
    monkeypatch.setattr(extract, "MAX_RETRIES", 1)
    real, calls = AiWorker.run_child, []

    def flaky(self, *args):
        calls.append(args)
        if len(calls) in (1, 3):
            raise child.JobFailed("the network went away")
        return real(self, *args)

    monkeypatch.setattr(AiWorker, "run_child", flaky)
    ext.plant(paper(1), paper(2))
    ended, job = ext.run()
    assert ended == "succeeded", job
    assert (job["result"]["retries"], job["result"]["documents"]) == (2, 2)


def test_switching_the_stage_off_mid_request_abandons_the_reservation_and_hands_the_job_back(ext, monkeypatch):
    real = AiWorker.run_child

    def switch_off(self, *args):
        ext.ai.flag("extract", False)
        return real(self, *args)

    monkeypatch.setattr(AiWorker, "run_child", switch_off)
    ext.plant(paper(1, "hang"))
    ended, job = ext.run()
    assert ended == "released (stage switched off)"
    assert (job["status"], job["attempts"]) == ("queued", 0)
    assert [r["status"] for r in ext.reservations()] == ["orphaned"] and ext.extractions() == {}
    monkeypatch.setattr(AiWorker, "run_child", real)
    ext.plant(paper(1))  # the paper changes; the job resumes with it
    ext.ai.flag("extract", True)
    runner = ext.worker()
    try:
        assert runner.run_once() == "succeeded"
    finally:
        runner.close()
    assert ext.ai.job(job["id"])["result"]["documents"] == 1
    assert [(r["status"], r["attempt"]) for r in ext.reservations()] == [("orphaned", 1), ("settled", 1)]


def test_a_worker_that_lost_its_job_records_nothing_and_the_reaper_counts_its_request(ext, monkeypatch):
    real = AiWorker.run_child

    def steal(self, *args):
        out = real(self, *args)
        ext.sql("UPDATE {ai}.jobs SET worker_id = 'thief' WHERE kind = 'extract'")
        return out

    monkeypatch.setattr(AiWorker, "run_child", steal)
    ext.plant(paper(1))
    ended, job = ext.run()
    assert ended == "lost (lease lapsed)" and job["status"] == "running"
    assert ext.extractions() == {} and [r["status"] for r in ext.reservations()] == ["reserved"]
    with ext.ai.conn.transaction():
        assert ledger.reclaim_orphans(ext.ai.conn, ext.ai.schema) == 1
    assert [r["status"] for r in ext.reservations()] == ["orphaned"]


def test_an_extraction_stops_at_the_kinds_deadline(ext, monkeypatch):
    monkeypatch.setitem(KINDS, "extract", dataclasses.replace(KINDS["extract"], timeout_seconds=0))
    ext.plant(paper(1))
    ended, job = ext.run()
    assert ended == "succeeded" and (job["result"]["documents"], job["result"]["caught_up"]) == (0, False)
    assert ext.reservations() == []


def test_a_job_process_in_production_refuses_the_fake_whatever_its_worker_was_told(ext, monkeypatch):
    """The worker's settings came from elsewhere; its job processes still
    inherit ENVIRONMENT and refuse to run the fake."""
    monkeypatch.setattr(extract, "MAX_RETRIES", 0)
    monkeypatch.setenv("ENVIRONMENT", "production")
    ext.plant(paper(1))
    ended, job = ext.run()
    assert ended == "failed" and "never runs in production" in job["error"], job
    assert [r["status"] for r in ext.reservations()] == ["orphaned"] and ext.extractions() == {}


def test_one_extraction_at_a_time(ext):
    ext.ai.enqueue({}, kind="extract")
    with pytest.raises(queue.EnqueueError, match="already queued or running"):
        ext.ai.enqueue({"max_docs": 5}, kind="extract")


def test_the_cli_checks_the_document_limit_before_it_queues(capsys):
    assert manage._ai_extract(None, 0, "ops@example.com") == 1
    assert "max_docs must be" in capsys.readouterr().err


def test_every_fake_behaviour_is_shown_working():
    tested = set(OUTCOMES) - {None} | {"busy", "rate_limited", "bad_request", "bad_usage", "crash", "hang"}
    assert tested == set(models.FAKE_BEHAVIOURS)


def test_the_ledger_abandons_only_an_open_reservation(ai):
    with ai.conn.transaction():
        rid = ledger.reserve(ai.conn, ai.schema, job_id="j", attempt=1, worker_id="w", model="m",
                             input_tokens=10, max_output_tokens=10, limits=ledger.Limits(100, 100))
        assert ledger.abandon(ai.conn, ai.schema, rid) is True
        assert ledger.abandon(ai.conn, ai.schema, rid) is False
        assert ledger.settle(ai.conn, ai.schema, rid, ledger.Usage()) is False
        assert ledger.spent_today(ai.conn, ai.schema) == (10, 10)
