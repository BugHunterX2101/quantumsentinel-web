"""Read hypotheses out of the registry's papers: the extract driver.

The AI worker runs this for an ``extract`` job (backend/ai/kinds.py) with
the payload ``{"max_docs": 500}``, ``max_docs`` optional. It asks the model
(AI_MODEL_PROVIDER, AI_MODEL) about every document not yet extracted with
this document version, prompt version and model, newest first, in batches
of BATCH, one request at a time. For each:

1. It builds the prompt (hypotheses.prompt_for). A paper whose title is
   empty once normalised cannot be sent; it is skipped for the rest of the
   job and counted as ``unsendable``.
2. It reserves tokens in a transaction that first makes sure the job is
   still its own: ``input_token_bound`` of the prompt, and
   hypotheses.MAX_OUTPUT_TOKENS. A spent daily budget ends the job, which
   succeeds with ``budget_exhausted``.
3. A job process (backend/ai/models.py) makes the request.
4. In one transaction that again makes sure the job is its own, it settles
   the reservation with the tokens the provider reports, checks the answer
   (hypotheses.check_output) and records it (hypotheses.record_extraction),
   whatever its status.

Every request ends in exactly one ledger action:

* an answer: ``settle`` with its usage (if the reaper had already marked the
  reservation orphaned, the answer is still recorded and the reservation
  counts in full);
* a provider's refusal (an HTTP status other than 200): ``settle`` with no
  tokens, as no answer was produced;
* a job process that failed or ran out of time, or a reply that cannot be
  read: ``abandon``, counting in full, as the request may have been billed;
* a job handed back while its request was out (the stage switched off, the
  worker stopping): ``abandon`` too. A job lost to another worker writes
  nothing; the reaper marks its reservation orphaned.

A failed request (a job process that failed or ran out of time, HTTP 5xx or
529, or 429 with a wait of at most MAX_RETRY_AFTER_SECONDS) is retried after
its wait or RETRY_SECONDS, at most MAX_RETRIES times in a row; past that the
job fails, naming the document, so a paper that always breaks its request
stops the extraction instead of being paid for again and again. A 429 with a
longer wait ends the job, which succeeds with ``rate_limited``. Any other
status, or a reply that cannot be read, fails the job. A job also ends when
nothing is due, after ``max_docs`` documents, or at the kind's deadline.
Every wait keeps the job's lease and stops when its stage is switched off.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass

import psycopg
from psycopg import sql

from . import child, documents, hypotheses, ledger, pacing, queue
from .worker import Interrupted

log = logging.getLogger("backend.ai.extract")

BATCH = 50
MAX_DOCS = 1_000_000
# The provider's own deadline for one request; the job process's is the kind's.
REQUEST_TIMEOUT_SECONDS = 150
MAX_RETRIES = 5
RETRY_SECONDS = 30
MAX_RETRY_AFTER_SECONDS = 900
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504, 529})
# What a provider adds to a request with an output schema (the schema's own
# instructions), in tokens; generous.
STRUCTURED_OUTPUT_ALLOWANCE = 2048
COUNTS = ("documents", "ok", "refused", "truncated", "invalid", "unexpected_stop", "duplicates", "hypotheses",
          "rejected", "unsendable", "input_tokens", "output_tokens")


class ReplyError(ValueError):
    """A job process's reply is not one this driver can read."""


class _RateLimited(Exception):
    pass


def check_payload(payload) -> int | None:
    """The document limit of a payload; raises ValueError."""
    if not isinstance(payload, dict):
        raise ValueError("the payload is a JSON object")
    unknown = set(payload) - {"max_docs"}
    if unknown:
        raise ValueError(f"unknown payload keys: {', '.join(sorted(unknown))}")
    max_docs = payload.get("max_docs")
    if max_docs is not None and (isinstance(max_docs, bool) or not isinstance(max_docs, int)
                                 or not 1 <= max_docs <= MAX_DOCS):
        raise ValueError(f"max_docs must be a whole number from 1 to {MAX_DOCS}")
    return max_docs


def input_token_bound(prompt: hypotheses.Prompt) -> int:
    """The input tokens to reserve for ``prompt``: one per UTF-8 byte of all
    that is sent (several times what a request counts) and
    STRUCTURED_OUTPUT_ALLOWANCE. The provider's own count settles it."""
    sent = hypotheses.SYSTEM_PROMPT + prompt.user_message + json.dumps(hypotheses.OUTPUT_SCHEMA)
    return len(sent.encode("utf-8")) + STRUCTURED_OUTPUT_ALLOWANCE


@dataclass(frozen=True)
class Reply:
    http_status: int
    retry_after: int | None
    stop_reason: str | None = None
    text: str | None = None
    usage: ledger.Usage = ledger.Usage()
    model: str | None = None
    response_id: str | None = None


def read_reply(out) -> Reply:
    """A job process's reply, checked; raises ReplyError."""
    try:
        status, retry_after = pacing.response_status(out)
    except pacing.SourceError as exc:
        raise ReplyError(str(exc)) from None
    if status != 200:
        return Reply(status, retry_after)
    stop_reason, text, usage = out.get("stop_reason"), out.get("text"), out.get("usage")
    if stop_reason is not None and not isinstance(stop_reason, str):
        raise ReplyError("its stop reason is not text")
    if text is not None and not isinstance(text, str):
        raise ReplyError("its answer is not text")
    if not isinstance(usage, dict) or not set(usage) <= {"input_tokens", "cache_creation_input_tokens",
                                                         "cache_read_input_tokens", "output_tokens"}:
        raise ReplyError("its usage is not the provider's token counts")
    try:
        counted = ledger.Usage(**{name: usage.get(name) or 0 for name in usage})
    except ValueError as exc:
        raise ReplyError(f"its usage is not the provider's token counts: {exc}") from None
    model, response_id = out.get("model"), out.get("id")
    return Reply(status, retry_after, stop_reason, text, counted,
                 model if isinstance(model, str) else None, response_id if isinstance(response_id, str) else None)


def _t(schema: str, name: str) -> sql.Composable:
    return sql.Identifier(schema, name)


class Asker:
    """What a driver that asks the model one question at a time shares
    (here and backend/ai/propose.py): the job's ownership, its end, the
    request path with its ledger actions and retries, and the guard around
    the driver's loop. A subclass names its task (WHAT, PROMPT_VERSION,
    COUNTS, which include input_tokens and output_tokens) and gives
    ``check_payload`` and ``loop``."""
    WHAT: str
    PROMPT_VERSION: str
    COUNTS: tuple[str, ...]

    def __init__(self, worker, job: dict, spec):
        self.worker, self.job, self.spec = worker, job, spec
        self.conn, self.schema, self.settings = worker.conn, worker.schema, worker.settings
        self.totals = dict.fromkeys(self.COUNTS, 0)
        self.retries = self.retries_in_a_row = 0

    @staticmethod
    def check_payload(payload) -> int | None:
        raise NotImplementedError

    def loop(self, limit: int | None, deadline: float) -> str:
        """Ask until done; returns how the job ended (complete or fail)."""
        raise NotImplementedError

    def own(self) -> None:
        pacing.own(self.conn, self.schema, self.job["id"], self.worker.id)

    def limits(self) -> ledger.Limits:
        return ledger.Limits(self.settings.daily_input_tokens, self.settings.daily_output_tokens)

    def result(self, caught_up: bool, budget_exhausted: bool = False, rate_limited: bool = False) -> dict:
        return {"provider": self.settings.model_provider, "model": self.settings.model,
                "prompt_version": self.PROMPT_VERSION, "caught_up": caught_up,
                "budget_exhausted": budget_exhausted, "rate_limited": rate_limited, "retries": self.retries,
                **self.totals}

    def fail(self, reason: str) -> str:
        reason = documents.clean_text(reason, 1000) or "failed"
        with self.conn.transaction():
            self.own()
            queue.fail(self.conn, self.schema, self.job["id"], self.worker.id,
                       f"The {self.WHAT} failed: {reason}")
        log.warning("%s %s failed: %s", self.WHAT, self.job["id"], reason)
        return "failed"

    def complete(self, caught_up: bool, budget_exhausted: bool = False, rate_limited: bool = False) -> str:
        with self.conn.transaction():
            done = queue.complete(self.conn, self.schema, self.job["id"], self.worker.id,
                                  self.result(caught_up, budget_exhausted, rate_limited))
        return "succeeded" if done else "lost (lease lapsed)"

    def run(self) -> str:
        try:
            limit = self.check_payload(self.job["payload"])
        except ValueError as exc:
            return self.fail(str(exc))
        if self.settings.model_provider is None:
            return self.fail("no model provider is configured (AI_MODEL_PROVIDER)")
        deadline = time.monotonic() + self.spec.timeout_seconds
        try:
            return self.loop(limit, deadline)
        except ledger.BudgetExhausted as exc:
            log.info("%s %s: %s", self.WHAT, self.job["id"], exc)
            return self.complete(caught_up=False, budget_exhausted=True)
        except _RateLimited:
            log.info("%s %s: the model provider's allowance is spent", self.WHAT, self.job["id"])
            return self.complete(caught_up=False, rate_limited=True)
        except pacing.SourceError as exc:
            return self.fail(str(exc))
        except psycopg.DatabaseError as exc:
            if isinstance(exc, psycopg.OperationalError):
                raise
            return self.fail(f"the database refused a write: {type(exc).__name__}")

    def request(self, subject: str, user_message: str, input_tokens: int, max_output_tokens: int,
                task: str | None = None) -> tuple[str, Reply] | None:
        """Reserve tokens, make one request in a job process and read its
        reply. Returns the reservation and a reply with HTTP 200, which the
        caller settles and records in one transaction that makes sure the job
        is still its own; None after a failed request that will be retried.
        Raises SourceError when it will not be (naming ``subject``),
        BudgetExhausted or _RateLimited."""
        with self.conn.transaction():
            self.own()
            reservation = ledger.reserve(
                self.conn, self.schema, job_id=self.job["id"], attempt=self.job["attempts"],
                worker_id=self.worker.id, model=self.settings.model, input_tokens=input_tokens,
                max_output_tokens=max_output_tokens, limits=self.limits())
        payload = {"provider": self.settings.model_provider, "model": self.settings.model,
                   "user_message": user_message, "timeout_seconds": REQUEST_TIMEOUT_SECONDS}
        if task is not None:
            payload["task"] = task
        try:
            out = self.worker.run_child(self.job, self.spec.target, payload, self.spec.child_timeout_seconds,
                                        self.spec.child_output_max_bytes)
        except child.JobFailed as exc:
            self.abandon(reservation)
            self.retry(subject, f"the request failed: {exc}", None)
            return None
        except Interrupted as stop:
            if stop.release:  # the job is still this worker's until execute hands it back
                with self.conn.transaction():
                    ledger.abandon(self.conn, self.schema, reservation)
            raise
        try:
            reply = read_reply(out)
        except ReplyError as exc:
            self.abandon(reservation)
            raise pacing.SourceError(f"the reply about {subject} did not pass the checks: {exc}") from None
        if reply.http_status != 200:
            with self.conn.transaction():
                self.own()
                ledger.settle(self.conn, self.schema, reservation, ledger.Usage())
            if reply.http_status in RETRY_STATUSES:
                if reply.http_status == 429 and reply.retry_after is not None \
                        and reply.retry_after > MAX_RETRY_AFTER_SECONDS:
                    raise _RateLimited()
                self.retry(subject, f"the model provider answered HTTP {reply.http_status}", reply.retry_after)
                return None
            raise pacing.SourceError(f"the model provider answered HTTP {reply.http_status}")
        return reservation, reply

    def settle(self, reservation: str, reply: Reply) -> None:
        """Settle an answered reservation, in the caller's owned transaction."""
        if not ledger.settle(self.conn, self.schema, reservation, reply.usage):
            log.warning("%s %s: reservation %s was closed before its answer came; it counts in full",
                        self.WHAT, self.job["id"], reservation)

    def answered(self, reply: Reply) -> None:
        """Count an answered request."""
        self.retries_in_a_row = 0
        self.totals["input_tokens"] += reply.usage.input_total
        self.totals["output_tokens"] += reply.usage.output_tokens

    def abandon(self, reservation: str) -> None:
        with self.conn.transaction():
            self.own()
            ledger.abandon(self.conn, self.schema, reservation)

    def retry(self, subject: str, reason: str, retry_after: int | None) -> None:
        self.retries += 1
        self.retries_in_a_row += 1
        if self.retries_in_a_row > MAX_RETRIES:
            raise pacing.SourceError(f"{reason} ({MAX_RETRIES} retries in a row, at {subject})")
        delay = RETRY_SECONDS if retry_after is None else min(max(retry_after, 1), MAX_RETRY_AFTER_SECONDS)
        log.info("%s %s: %s; retrying in %d s", self.WHAT, self.job["id"], reason, delay)
        self.worker.sleep(self.job, delay)


class _Extract(Asker):
    WHAT = "hypothesis extraction"
    PROMPT_VERSION = hypotheses.PROMPT_VERSION
    COUNTS = COUNTS

    def __init__(self, worker, job: dict, spec):
        super().__init__(worker, job, spec)
        self.unsendable: list[str] = []

    check_payload = staticmethod(check_payload)

    def due(self) -> list[tuple[str, str, str, str]]:
        """The next batch of documents to ask about: (id, content_hash, title, abstract)."""
        with self.conn.transaction():
            return self.conn.execute(sql.SQL(
                "SELECT d.id, d.content_hash, d.title, d.abstract FROM {documents} d"
                " WHERE d.deleted_at IS NULL AND NOT (d.id = ANY(%(skip)s::text[]))"
                " AND NOT EXISTS (SELECT 1 FROM {extractions} e WHERE e.document_id = d.id"
                "                 AND e.content_hash = d.content_hash AND e.prompt_version = %(version)s"
                "                 AND e.requested_model = %(model)s)"
                " ORDER BY d.datestamp DESC, d.id LIMIT %(batch)s"
            ).format(documents=_t(self.schema, "documents"), extractions=_t(self.schema, "extractions")),
                {"skip": self.unsendable, "version": hypotheses.PROMPT_VERSION, "model": self.settings.model,
                 "batch": BATCH}).fetchall()

    def loop(self, max_docs: int | None, deadline: float) -> str:
        while True:
            batch = self.due()
            if not batch:
                return self.complete(caught_up=True)
            for document_id, content_hash, title, abstract in batch:
                if max_docs is not None and self.totals["documents"] >= max_docs:
                    return self.complete(caught_up=False)
                try:
                    prompt = hypotheses.prompt_for(title, abstract)
                except ValueError:
                    self.unsendable.append(document_id)
                    self.totals["unsendable"] += 1
                    continue
                while True:
                    if time.monotonic() >= deadline:
                        return self.complete(caught_up=False)
                    if self.ask(document_id, content_hash, prompt):
                        break

    def ask(self, document_id: str, content_hash: str, prompt: hypotheses.Prompt) -> bool:
        """One request, its ledger action and its record; False after a
        failed request that will be retried. Raises SourceError when it will
        not be, BudgetExhausted or _RateLimited."""
        asked = self.request(f"document {document_id}", prompt.user_message, input_token_bound(prompt),
                             hypotheses.MAX_OUTPUT_TOKENS)
        if asked is None:
            return False
        reservation, reply = asked
        checked = hypotheses.check_output(reply.stop_reason, reply.text, prompt)
        with self.conn.transaction():
            self.own()
            self.settle(reservation, reply)
            extraction_id = hypotheses.record_extraction(
                self.conn, self.schema, job_id=self.job["id"], document_id=document_id, content_hash=content_hash,
                prompt=prompt, requested_model=self.settings.model, response_model=reply.model,
                response_id=reply.response_id, stop_reason=reply.stop_reason, usage=reply.usage, checked=checked)
        self.answered(reply)
        self.totals["documents"] += 1
        if extraction_id is None:
            self.totals["duplicates"] += 1
        else:
            self.totals[checked.status] += 1
            self.totals["hypotheses"] += len(checked.hypotheses)
            self.totals["rejected"] += len(checked.rejections)
        return True


def run(worker, job: dict, spec) -> str:
    """The extract driver (backend/ai/kinds.py); returns how the job ended."""
    return _Extract(worker, job, spec).run()
