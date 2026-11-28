"""Propose a strategy signal for each extracted hypothesis: the propose driver.

The AI worker runs this for a ``propose`` job (backend/ai/kinds.py) with the
payload ``{"max_hypotheses": 500}``, ``max_hypotheses`` optional. It walks
the hypotheses that are due, newest extraction first, in batches of BATCH,
one at a time. A hypothesis is due when it has no proposal yet for this
proposal prompt version (proposals.PROMPT_VERSION) and model (AI_MODEL), and
it comes from a current extraction: one whose paper is not deleted, whose
content hash is the paper's current one, and which used the current
hypothesis prompt version and the same model. That is the set the extract
driver treats as done, so an older extraction of a paper (an earlier
version of it, an earlier prompt, another model) is never paid for again,
and neither is a deleted paper.

For each hypothesis:

* If its signal family is not one of the DSL's templates
  (proposals.in_scope), it is recorded as ``out_of_scope`` without a request
  or a reservation, in a transaction that makes sure the job is still its own.
* Otherwise the request goes through the extract driver's path
  (extract.Asker.request): tokens reserved (``input_token_bound`` and
  proposals.MAX_OUTPUT_TOKENS), one job process, the reply read, every
  request ending in exactly one ledger action, failures retried as there.
  Then, in one transaction that again makes sure the job is its own, the
  reservation is settled, the answer checked (proposals.check_output) and
  recorded (proposals.record_proposal), whatever its status.

A proposal is only ever a proposal: a person accepts it, or not, in the
trusted tier. A job ends when nothing is due, after ``max_hypotheses``
hypotheses (out of scope ones included), when the daily token budget is
spent (``budget_exhausted``), when the provider's allowance is
(``rate_limited``), or at the kind's deadline.
"""
from __future__ import annotations

import json
import time

from psycopg import sql

from . import extract, hypotheses, proposals

BATCH = 50
MAX_HYPOTHESES = 1_000_000
COUNTS = ("hypotheses", *proposals.STATUSES, "duplicates", "input_tokens", "output_tokens")


def check_payload(payload) -> int | None:
    """The hypothesis limit of a payload; raises ValueError."""
    if not isinstance(payload, dict):
        raise ValueError("the payload is a JSON object")
    unknown = set(payload) - {"max_hypotheses"}
    if unknown:
        raise ValueError(f"unknown payload keys: {', '.join(sorted(unknown))}")
    limit = payload.get("max_hypotheses")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int)
                              or not 1 <= limit <= MAX_HYPOTHESES):
        raise ValueError(f"max_hypotheses must be a whole number from 1 to {MAX_HYPOTHESES}")
    return limit


def input_token_bound(prompt: proposals.Prompt) -> int:
    """The input tokens to reserve for ``prompt``, counted as the extract
    driver counts its own (extract.input_token_bound)."""
    sent = proposals.SYSTEM_PROMPT + prompt.user_message + json.dumps(proposals.OUTPUT_SCHEMA)
    return len(sent.encode("utf-8")) + extract.STRUCTURED_OUTPUT_ALLOWANCE


def _t(schema: str, name: str) -> sql.Composable:
    return sql.Identifier(schema, name)


class _Propose(extract.Asker):
    WHAT = "strategy proposal"
    PROMPT_VERSION = proposals.PROMPT_VERSION
    COUNTS = COUNTS

    check_payload = staticmethod(check_payload)

    def due(self) -> list[tuple]:
        """The next batch of hypotheses: (extraction_id, ordinal, *HYPOTHESIS_FIELDS)."""
        fields = sql.SQL(", ").join(sql.SQL("h.") + sql.Identifier(f) for f in proposals.HYPOTHESIS_FIELDS)
        with self.conn.transaction():
            return self.conn.execute(sql.SQL(
                "SELECT h.extraction_id, h.ordinal, {fields} FROM {hypotheses} h"
                " JOIN {extractions} e ON e.id = h.extraction_id"
                " JOIN {documents} d ON d.id = e.document_id"
                " WHERE d.deleted_at IS NULL AND e.content_hash = d.content_hash"
                " AND e.prompt_version = %(extraction_version)s AND e.requested_model = %(model)s"
                " AND NOT EXISTS (SELECT 1 FROM {proposals} p WHERE p.extraction_id = h.extraction_id"
                "                 AND p.ordinal = h.ordinal AND p.prompt_version = %(version)s"
                "                 AND p.requested_model = %(model)s)"
                " ORDER BY e.created_at DESC, h.extraction_id, h.ordinal LIMIT %(batch)s"
            ).format(fields=fields, hypotheses=_t(self.schema, "hypotheses"),
                     extractions=_t(self.schema, "extractions"), documents=_t(self.schema, "documents"),
                     proposals=_t(self.schema, "proposals")),
                {"extraction_version": hypotheses.PROMPT_VERSION, "version": proposals.PROMPT_VERSION,
                 "model": self.settings.model, "batch": BATCH}).fetchall()

    def loop(self, limit: int | None, deadline: float) -> str:
        while True:
            batch = self.due()
            if not batch:
                return self.complete(caught_up=True)
            for extraction_id, ordinal, *values in batch:
                if limit is not None and self.totals["hypotheses"] >= limit:
                    return self.complete(caught_up=False)
                if time.monotonic() >= deadline:
                    return self.complete(caught_up=False)
                hypothesis = dict(zip(proposals.HYPOTHESIS_FIELDS, values))
                if not proposals.in_scope(hypothesis["signal_family"]):
                    self.out_of_scope(extraction_id, ordinal)
                    continue
                prompt = proposals.prompt_for(hypothesis)
                while not self.ask(extraction_id, ordinal, prompt):
                    if time.monotonic() >= deadline:
                        return self.complete(caught_up=False)

    def counted(self, proposal_id: str | None, status: str) -> None:
        self.totals["hypotheses"] += 1
        self.totals["duplicates" if proposal_id is None else status] += 1

    def out_of_scope(self, extraction_id: str, ordinal: int) -> None:
        with self.conn.transaction():
            self.own()
            proposal_id = proposals.record_out_of_scope(
                self.conn, self.schema, job_id=self.job["id"], extraction_id=extraction_id, ordinal=ordinal,
                requested_model=self.settings.model)
        self.counted(proposal_id, "out_of_scope")

    def ask(self, extraction_id: str, ordinal: int, prompt: proposals.Prompt) -> bool:
        """One request, its ledger action and its record; False after a
        failed request that will be retried."""
        asked = self.request(f"hypothesis {ordinal} of extraction {extraction_id}", prompt.user_message,
                             input_token_bound(prompt), proposals.MAX_OUTPUT_TOKENS, proposals.TASK)
        if asked is None:
            return False
        reservation, reply = asked
        checked = proposals.check_output(reply.stop_reason, reply.text)
        with self.conn.transaction():
            self.own()
            self.settle(reservation, reply)
            proposal_id = proposals.record_proposal(
                self.conn, self.schema, job_id=self.job["id"], extraction_id=extraction_id, ordinal=ordinal,
                prompt=prompt, requested_model=self.settings.model, response_model=reply.model,
                response_id=reply.response_id, stop_reason=reply.stop_reason, usage=reply.usage, checked=checked)
        self.answered(reply)
        self.counted(proposal_id, checked.status)
        return True


def run(worker, job: dict, spec) -> str:
    """The propose driver (backend/ai/kinds.py); returns how the job ended."""
    return _Propose(worker, job, spec).run()
