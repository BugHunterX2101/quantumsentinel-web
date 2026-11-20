"""Ask OpenAlex or Crossref about the registry's documents: the metadata_sync driver.

The AI worker runs this for a ``metadata_sync`` job (backend/ai/kinds.py)
with the job's payload ``{"source": "openalex", "max_docs": 500}``, where
``max_docs`` is optional. It looks up, one request at a time, every
document that is due (table ``document_metadata``, migration 0015):

* one never looked up at this service, or changed since its last lookup;
* one found more than FOUND_RECHECK_DAYS ago, or not found (or refused)
  more than RECHECK_DAYS ago;

newest first, in batches of BATCH. The DOI asked about is built here:

* OpenAlex: arXiv's own DOI for the paper (``10.48550/arXiv.<id>``, the
  subject class of an old-style identifier left out), which OpenAlex
  resolves to the merged work, journal version included.
* Crossref: the journal DOI arXiv lists for the paper, if any; arXiv's own
  DOIs are DataCite's, which Crossref does not know, so a document without
  another DOI is not asked about there.

A DOI that is not one metadata_fetch.DOI_PATTERN accepts is recorded as
``refused`` without a request. For the rest:

1. It takes the source's next rate slot (table ``sources``) and waits for
   it, so every worker together keeps its rate.
2. A job process (backend/ai/metadata_fetch.py) makes the one GET, with the
   URL built from AI_OPENALEX_URL or AI_CROSSREF_URL and the DOI, and, for
   OpenAlex, AI_OPENALEX_API_KEY in a header when it is set. The key is
   handed to that process alone and is in no payload the database keeps.
3. It checks what the job process returned (``clean_work``) and records the
   answer in one transaction that first makes sure the job is still its own.

An answer of 200 is recorded as ``found`` (or ``refused`` when its body is
not a work that passes the checks), 404 or 410 as ``not_found`` and 400 as
``refused``. A failed request (no answer, HTTP 5xx, or 429 with a wait of
at most MAX_RETRY_AFTER_SECONDS) is retried after its wait or
RETRY_SECONDS, at most MAX_RETRIES times in a row. A 429 with a longer wait
is the source's daily allowance spent: the job pushes the source's rate
slot past it (at most MAX_RATE_LIMIT_SECONDS) and ends with
``rate_limited``. Any other status fails the job. A sync also ends when
nothing is due, after ``max_docs`` documents, or at the kind's deadline.
Every wait keeps the job's lease and stops when its stage is switched off.
"""
from __future__ import annotations

import json
import logging
import re
import time

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from . import child, documents, metadata_fetch, pacing, queue

log = logging.getLogger("backend.ai.enrich")

SOURCES = metadata_fetch.SOURCES
BATCH = 100
MAX_DOCS = 1_000_000
FOUND_RECHECK_DAYS = 30
RECHECK_DAYS = 3
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 5
RETRY_SECONDS = 30
MAX_RETRY_AFTER_SECONDS = 900
MAX_RATE_LIMIT_SECONDS = 24 * 3600
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
NOT_FOUND_STATUSES = frozenset({404, 410})
ARXIV_DOI_PREFIX = "10.48550/"
# Crossref's update types that mean the work was taken back.
RETRACTION_TYPES = frozenset({"retraction", "withdrawal", "removal"})
COUNTS = ("looked_up", "found", "not_found", "refused", "changed")
MAX_COUNT = 10 ** 9

_OPENALEX_ID = re.compile(r"https://openalex\.org/(W[0-9]{1,15})")
_DAY = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_URL = re.compile(r"https?://[\x21-\x7e]{1,290}")
_UPDATE_TYPE = re.compile(r"[a-z][a-z_-]{0,63}")
_SUBJECT_CLASS = re.compile(r"\.[A-Z]{2}/")


class WorkError(ValueError):
    """A job process's answer is not a work that passes the checks."""


class _RateLimited(Exception):
    def __init__(self, seconds: int):
        super().__init__(seconds)
        self.seconds = seconds


def check_payload(payload) -> tuple[str, int | None]:
    """The source and document limit of a payload; raises ValueError."""
    if not isinstance(payload, dict):
        raise ValueError("the payload is a JSON object")
    unknown = set(payload) - {"source", "max_docs"}
    if unknown:
        raise ValueError(f"unknown payload keys: {', '.join(sorted(unknown))}")
    source, max_docs = payload.get("source"), payload.get("max_docs")
    if source not in SOURCES:
        raise ValueError(f"source must be one of {', '.join(SOURCES)}")
    if max_docs is not None and (isinstance(max_docs, bool) or not isinstance(max_docs, int)
                                 or not 1 <= max_docs <= MAX_DOCS):
        raise ValueError(f"max_docs must be a whole number from 1 to {MAX_DOCS}")
    return source, max_docs


def lookup_doi(source: str, source_id: str, doi: str | None) -> str | None:
    """The DOI to ask ``source`` about a document, or None for none."""
    if source == "openalex":
        return "10.48550/arXiv." + _SUBJECT_CLASS.sub("/", source_id)
    if doi is None or doi.lower().startswith(ARXIV_DOI_PREFIX):
        return None
    return doi


# -- checking an answer ----------------------------------------------------------------

def _text(value, limit: int, what: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise WorkError(f"its {what} is not a string")
    return documents.clean_text(value, limit) or None


def _count(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_COUNT:
        raise WorkError("its citation count is not a whole number from 0 to 10^9")
    return value


def _openalex(work: dict) -> tuple[str, int | None, bool | None, dict]:
    work_id = work.get("id")
    match = _OPENALEX_ID.fullmatch(work_id) if isinstance(work_id, str) else None
    if match is None:
        raise WorkError("it has no OpenAlex work id")
    retracted = work.get("is_retracted")
    if retracted is not None and not isinstance(retracted, bool):
        raise WorkError("its retraction flag is not true or false")
    day = _text(work.get("publication_date"), 10, "publication date")
    if day is not None and not _DAY.fullmatch(day):
        raise WorkError("its publication date is not YYYY-MM-DD")
    fields = {"doi": _text(work.get("doi"), 300, "DOI"), "type": _text(work.get("type"), 64, "type"),
              "publication_date": day, "venue": _text(work.get("venue"), 500, "venue"),
              "topic": _text(work.get("topic"), 500, "topic"), "field": _text(work.get("field"), 200, "field")}
    return match.group(1), _count(work.get("cited_by_count")), retracted, fields


def _crossref(work: dict, asked: str) -> tuple[str, int | None, bool | None, dict]:
    doi = work.get("doi")
    if not metadata_fetch.valid_doi(doi):
        raise WorkError("it has no valid DOI")
    if doi.lower() != asked.lower():
        raise WorkError("it is a work with another DOI")
    published = work.get("published")
    if published is not None and not (
            isinstance(published, list) and 1 <= len(published) <= 3
            and all(isinstance(p, int) and not isinstance(p, bool) for p in published)
            and 1000 <= published[0] <= 3000 and all(1 <= p <= 31 for p in published[1:])):
        raise WorkError("its publication date is not [year, month, day]")
    licences, updates = work.get("licence_urls"), work.get("update_types")
    if not isinstance(licences, list) or len(licences) > 10 \
            or not all(isinstance(u, str) and _URL.fullmatch(u) for u in licences):
        raise WorkError("its licence URLs are not a short list of URLs")
    if not isinstance(updates, list) or len(updates) > 50 \
            or not all(isinstance(t, str) and _UPDATE_TYPE.fullmatch(t) for t in updates):
        raise WorkError("its update types are not a short list of names")
    fields = {"type": _text(work.get("type"), 64, "type"), "venue": _text(work.get("venue"), 500, "venue"),
              "publisher": _text(work.get("publisher"), 500, "publisher"), "published": published,
              "licence_urls": sorted(set(licences)), "update_types": sorted(set(updates))}
    return doi.lower(), _count(work.get("cited_by_count")), bool(RETRACTION_TYPES & set(updates)), fields


def clean_work(source: str, work, asked: str) -> tuple[str, int | None, bool | None, dict]:
    """``(external_id, cited_by_count, is_retracted, fields)`` of a work a job
    process returned; raises WorkError."""
    if not isinstance(work, dict):
        raise WorkError("it is not an object")
    external_id, count, retracted, fields = _openalex(work) if source == "openalex" else _crossref(work, asked)
    fields = {key: value for key, value in fields.items() if value not in (None, [])}
    if len(json.dumps(fields)) > 8192:
        raise WorkError("its fields are too large")
    return external_id, count, retracted, fields


# -- the sync ----------------------------------------------------------------------------

def _t(schema: str, name: str) -> sql.Composable:
    return sql.Identifier(schema, name)


class _Enrich:
    def __init__(self, worker, job: dict, spec):
        self.worker, self.job, self.spec = worker, job, spec
        self.conn, self.schema = worker.conn, worker.schema
        self.totals = dict.fromkeys(COUNTS, 0)
        self.retries = self.retries_in_a_row = self.documents = 0
        self.source = ""

    def own(self) -> None:
        pacing.own(self.conn, self.schema, self.job["id"], self.worker.id)

    def due(self) -> list[tuple[str, str, str | None]]:
        """The next batch of documents to look up: (id, source_id, doi)."""
        only_journal_dois = sql.SQL(
            " AND d.doi IS NOT NULL AND lower(d.doi) NOT LIKE %(arxiv)s" if self.source == "crossref" else "")
        with self.conn.transaction():
            return self.conn.execute(sql.SQL(
                "SELECT d.id, d.source_id, d.doi FROM {documents} d"
                " LEFT JOIN {metadata} m ON m.document_id = d.id AND m.source = %(source)s"
                " WHERE d.deleted_at IS NULL{only}"
                " AND (m.document_id IS NULL OR m.checked_at < d.changed_at"
                "      OR (m.status = 'found' AND m.checked_at < now() - make_interval(days => %(found)s))"
                "      OR (m.status <> 'found' AND m.checked_at < now() - make_interval(days => %(other)s)))"
                " ORDER BY m.checked_at NULLS FIRST, d.datestamp DESC, d.id LIMIT %(batch)s"
            ).format(documents=_t(self.schema, "documents"), metadata=_t(self.schema, "document_metadata"),
                     only=only_journal_dois),
                {"source": self.source, "arxiv": ARXIV_DOI_PREFIX + "%", "found": FOUND_RECHECK_DAYS,
                 "other": RECHECK_DAYS, "batch": BATCH}).fetchall()

    def record(self, document_id: str, status: str, doi: str | None, reason: str | None = None,
               external_id: str | None = None, count: int | None = None, retracted: bool | None = None,
               fields: dict | None = None) -> None:
        """Record one lookup, in a transaction that first checks the job is
        still this worker's."""
        values = {"document_id": document_id, "source": self.source, "status": status,
                  "lookup_doi": doi[:255] if doi else None,
                  "reason": documents.clean_text(reason, 500) if reason else None,
                  "external_id": external_id, "count": count, "retracted": retracted,
                  "fields": Jsonb(fields or {})}
        with self.conn.transaction():
            self.own()
            changed = self.conn.execute(sql.SQL(
                "INSERT INTO {m} AS m (document_id, source, status, lookup_doi, reason, external_id,"
                " cited_by_count, is_retracted, fields)"
                " VALUES (%(document_id)s, %(source)s, %(status)s, %(lookup_doi)s, %(reason)s, %(external_id)s,"
                " %(count)s, %(retracted)s, %(fields)s)"
                " ON CONFLICT (document_id, source) DO UPDATE SET status = EXCLUDED.status,"
                " lookup_doi = EXCLUDED.lookup_doi, reason = EXCLUDED.reason, external_id = EXCLUDED.external_id,"
                " cited_by_count = EXCLUDED.cited_by_count, is_retracted = EXCLUDED.is_retracted,"
                " fields = EXCLUDED.fields, checks = m.checks + 1, checked_at = now(),"
                " changed_at = CASE WHEN (m.status, m.lookup_doi, m.reason, m.external_id, m.cited_by_count,"
                "   m.is_retracted, m.fields) IS DISTINCT FROM (EXCLUDED.status, EXCLUDED.lookup_doi,"
                "   EXCLUDED.reason, EXCLUDED.external_id, EXCLUDED.cited_by_count, EXCLUDED.is_retracted,"
                "   EXCLUDED.fields) THEN now() ELSE m.changed_at END"
                " RETURNING m.changed_at = m.checked_at"
            ).format(m=_t(self.schema, "document_metadata")), values).fetchone()[0]
        self.documents += 1
        self.totals[status] += 1
        self.totals["changed"] += bool(changed)

    def result(self, caught_up: bool, rate_limited: bool = False) -> dict:
        return {"source": self.source, "documents": self.documents, "caught_up": caught_up,
                "rate_limited": rate_limited, "retries": self.retries, **self.totals}

    def fail(self, reason: str) -> str:
        reason = documents.clean_text(reason, 1000) or "failed"
        with self.conn.transaction():
            self.own()
            queue.fail(self.conn, self.schema, self.job["id"], self.worker.id,
                       f"The {self.source or 'metadata'} sync failed: {reason}")
        log.warning("metadata sync %s failed: %s", self.job["id"], reason)
        return "failed"

    def complete(self, caught_up: bool, rate_limited: bool = False) -> str:
        with self.conn.transaction():
            done = queue.complete(self.conn, self.schema, self.job["id"], self.worker.id,
                                  self.result(caught_up, rate_limited))
        return "succeeded" if done else "lost (lease lapsed)"

    def run(self) -> str:
        try:
            self.source, max_docs = check_payload(self.job["payload"])
        except ValueError as exc:
            return self.fail(str(exc))
        deadline = time.monotonic() + self.spec.timeout_seconds
        try:
            while True:
                batch = self.due()
                if not batch:
                    return self.complete(caught_up=True)
                for document_id, source_id, doi in batch:
                    if max_docs is not None and self.documents >= max_docs:
                        return self.complete(caught_up=False)
                    asked = lookup_doi(self.source, source_id, doi)
                    if not metadata_fetch.valid_doi(asked):
                        self.record(document_id, "refused", None,
                                    reason=f"its DOI is not one {self.source} can be asked about")
                        continue
                    while True:
                        wait = pacing.take_slot(self.conn, self.schema, self.source)
                        if time.monotonic() + wait >= deadline:
                            return self.complete(caught_up=False)
                        self.worker.sleep(self.job, wait)
                        if self.look_up(document_id, asked):
                            break
        except _RateLimited as limited:
            pacing.push_slot(self.conn, self.schema, self.source, limited.seconds)
            log.info("metadata sync %s: %s's allowance is spent; next request in %d s", self.job["id"],
                     self.source, limited.seconds)
            return self.complete(caught_up=False, rate_limited=True)
        except pacing.SourceError as exc:
            return self.fail(str(exc))
        except psycopg.DatabaseError as exc:
            if isinstance(exc, psycopg.OperationalError):
                raise
            return self.fail(f"the database refused a lookup: {type(exc).__name__}")

    def look_up(self, document_id: str, asked: str) -> bool:
        """One request and its record; False after a failed request that
        will be retried. Raises SourceError when it will not be."""
        payload = {"source": self.source, "doi": asked, "timeout_seconds": REQUEST_TIMEOUT_SECONDS,
                   "base_url": getattr(self.worker.settings, f"{self.source}_url")}
        key = self.worker.settings.openalex_api_key
        if self.source == "openalex" and key:
            payload["api_key"] = key
        try:
            out = self.worker.run_child(self.job, self.spec.target, payload, self.spec.child_timeout_seconds,
                                        self.spec.child_output_max_bytes)
        except child.JobFailed as exc:
            self.retry(f"the request failed: {exc}", None)
            return False
        status, retry_after = pacing.response_status(out)
        if status in RETRY_STATUSES:
            if status == 429 and retry_after is not None and retry_after > MAX_RETRY_AFTER_SECONDS:
                raise _RateLimited(min(retry_after, MAX_RATE_LIMIT_SECONDS))
            self.retry(f"{self.source} answered HTTP {status}", retry_after)
            return False
        self.totals["looked_up"] += 1
        if status == 200:
            pacing.push_slot(self.conn, self.schema, self.source)
            self.retries_in_a_row = 0
            if "work" not in out:
                self.record(document_id, "refused", asked,
                            reason=f"its answer is not a work: {out.get('unreadable') or 'no reason given'}")
                return True
            try:
                external_id, count, retracted, fields = clean_work(self.source, out["work"], asked)
            except (WorkError, documents.RecordError) as exc:
                self.record(document_id, "refused", asked, reason=f"its answer did not pass the checks: {exc}")
                return True
            self.record(document_id, "found", asked, external_id=external_id, count=count,
                        retracted=retracted, fields=fields)
            return True
        if status in NOT_FOUND_STATUSES or status == 400:
            pacing.push_slot(self.conn, self.schema, self.source)
            self.retries_in_a_row = 0
            if status == 400:
                self.record(document_id, "refused", asked, reason=f"{self.source} refused the DOI (HTTP 400)")
            else:
                self.record(document_id, "not_found", asked)
            return True
        raise pacing.SourceError(f"{self.source} answered HTTP {status}")

    def retry(self, reason: str, retry_after: int | None) -> None:
        self.retries += 1
        self.retries_in_a_row += 1
        if self.retries_in_a_row > MAX_RETRIES:
            raise pacing.SourceError(f"{reason} ({MAX_RETRIES} retries in a row)")
        delay = RETRY_SECONDS if retry_after is None else min(max(retry_after, 1), MAX_RETRY_AFTER_SECONDS)
        log.info("metadata sync %s: %s; retrying in %d s", self.job["id"], reason, delay)
        pacing.push_slot(self.conn, self.schema, self.source, delay)


def run(worker, job: dict, spec) -> str:
    """The metadata_sync driver (backend/ai/kinds.py); returns how the job ended."""
    return _Enrich(worker, job, spec).run()
