"""Harvest one arXiv set into the document registry: the arxiv_sync driver.

The AI worker runs this for an ``arxiv_sync`` job (backend/ai/kinds.py)
with the job's payload ``{"set": "q-fin", "from": "2026-01-01",
"max_pages": 10}``, where ``from`` is needed only for a set never harvested
before (and is ignored after that) and ``max_pages`` is optional. It walks
the set's datestamps in date windows of WINDOW_DAYS, from its cursor (table
``harvests``, migration 0014) to today, one OAI-PMH page at a time:

1. It takes the next rate slot for arXiv (table ``sources``) and waits for
   it, so every worker together sends at most one request per
   ``min_interval_seconds``.
2. It fetches and parses the page in a job process (backend/ai/oai_fetch.py)
   with the request built here, from AI_ARXIV_OAI_URL and the cursor; the
   payload names no URL.
3. It checks what the job process returned (documents.clean_page) and, in
   one transaction that first makes sure the job is still its own, records
   the page (documents.upsert_page) and moves the cursor.

Each window starts on the last day of the one before it, since OAI-PMH
datestamps are days; recording a page twice changes nothing. A window ends
when a page has no resumption token, or arXiv says noRecordsMatch. A sync
ends when the window that reaches today is complete, when ``max_pages``
pages have been recorded, or at the kind's deadline; the cursor keeps its
place, so the next sync goes on from there.

A failed request (no answer, or HTTP 429/5xx) is retried after its
Retry-After (at most MAX_RETRY_AFTER_SECONDS) or RETRY_SECONDS, at most
MAX_RETRIES times in a row. An expired or refused resumption token restarts
its window. Anything else fails the job, with the reason kept on the cursor.
Every wait keeps the job's lease and stops when its stage is switched off.
"""
from __future__ import annotations

import datetime as dt
import logging
import re
import time

import psycopg
from psycopg import sql

from . import child, documents, queue
from .worker import Interrupted

log = logging.getLogger("backend.ai.harvest")

SOURCE = documents.SOURCE
METADATA_PREFIX = "arXiv"
WINDOW_DAYS = 31
EARLIEST = dt.date(1991, 1, 1)
MAX_PAGES = 100_000
REQUEST_TIMEOUT_SECONDS = 60
MAX_RETRIES = 5
RETRY_SECONDS = 30
MAX_RETRY_AFTER_SECONDS = 900
MAX_TOKEN_RESTARTS = 3
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
COUNTS = ("records", "inserted", "updated", "unchanged", "deletions", "deleted", "rejected")
_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")


class SyncError(Exception):
    """The sync cannot go on; the job fails with this reason."""


def check_payload(payload, today: dt.date | None = None) -> tuple[str, dt.date | None, int | None]:
    """(set, from, max_pages) of an arxiv_sync payload; raises ValueError."""
    today = today or dt.datetime.now(dt.timezone.utc).date()
    if not isinstance(payload, dict) or not set(payload) <= {"set", "from", "max_pages"}:
        raise ValueError("an arxiv_sync payload names only set, from and max_pages")
    set_spec = payload.get("set")
    if not isinstance(set_spec, str) or len(set_spec) > documents.MAX_SET_SPEC \
            or not documents._SET_SPEC.fullmatch(set_spec):
        raise ValueError("set must be an arXiv set spec, such as q-fin or physics:hep-th")
    start = payload.get("from")
    if start is not None:
        if not isinstance(start, str) or not _DATE.fullmatch(start):
            raise ValueError("from must be a date, YYYY-MM-DD")
        try:
            start = dt.date.fromisoformat(start)
        except ValueError:
            raise ValueError("from must be a date, YYYY-MM-DD") from None
        if not EARLIEST <= start <= today:
            raise ValueError(f"from must be between {EARLIEST} and today")
    max_pages = payload.get("max_pages")
    if max_pages is not None and (isinstance(max_pages, bool) or not isinstance(max_pages, int)
                                  or not 1 <= max_pages <= MAX_PAGES):
        raise ValueError(f"max_pages must be a whole number from 1 to {MAX_PAGES}")
    return set_spec, start, max_pages


def response_status(out) -> tuple[int, int | None]:
    """The HTTP status and Retry-After a job process reported; raises SyncError."""
    if not isinstance(out, dict):
        raise SyncError("the job process returned no result object")
    status, retry_after = out.get("http_status"), out.get("retry_after")
    if isinstance(status, bool) or not isinstance(status, int) or not 100 <= status <= 599:
        raise SyncError("the job process returned no valid HTTP status")
    if retry_after is not None and (isinstance(retry_after, bool) or not isinstance(retry_after, int)
                                    or retry_after < 0):
        retry_after = None
    return status, retry_after


def _t(schema: str, name: str) -> sql.Composable:
    return sql.Identifier(schema, name)


class _Sync:
    def __init__(self, worker, job: dict, spec):
        self.worker, self.job, self.spec = worker, job, spec
        self.conn, self.schema = worker.conn, worker.schema
        self.totals = dict.fromkeys(COUNTS, 0)
        self.pages = self.windows = self.retries = self.retries_in_a_row = 0

    # -- the database ---------------------------------------------------------------

    def own(self) -> None:
        """Inside a transaction: lock the job row, or raise Interrupted if
        the job is no longer this worker's."""
        row = self.conn.execute(sql.SQL(
            "SELECT 1 FROM {} WHERE id = %s AND worker_id = %s AND status = 'running' FOR UPDATE"
        ).format(_t(self.schema, "jobs")), (self.job["id"], self.worker.id)).fetchone()
        if row is None:
            raise Interrupted("lost (lease lapsed)", release=False)

    def open_cursor(self, set_spec: str, start: dt.date | None) -> dict:
        with self.conn.transaction():
            self.own()
            if start is not None:
                self.conn.execute(sql.SQL(
                    "INSERT INTO {} (source, set_spec, start_date) VALUES (%s, %s, %s)"
                    " ON CONFLICT (source, set_spec) DO NOTHING").format(_t(self.schema, "harvests")),
                    (SOURCE, set_spec, start))
            row = self.conn.execute(sql.SQL(
                "SELECT start_date, last_complete_until, window_from, window_until, resumption_token,"
                " token_expires FROM {} WHERE source = %s AND set_spec = %s FOR UPDATE"
            ).format(_t(self.schema, "harvests")), (SOURCE, set_spec)).fetchone()
        if row is None:
            raise SyncError(f"set {set_spec} has never been harvested: give the date to start from")
        keys = ("start_date", "last_complete_until", "window_from", "window_until", "token", "token_expires")
        return {"set": set_spec, **dict(zip(keys, row))}

    def take_slot(self) -> float:
        """Take the next request slot; returns the seconds until it."""
        with self.conn.transaction():
            row = self.conn.execute(sql.SQL(
                "UPDATE {} SET next_allowed_at = greatest(next_allowed_at, now())"
                " + make_interval(secs => min_interval_seconds::float8) WHERE name = %s"
                " RETURNING extract(epoch FROM next_allowed_at"
                " - make_interval(secs => min_interval_seconds::float8) - now())::float8"
            ).format(_t(self.schema, "sources")), (SOURCE,)).fetchone()
        if row is None:
            raise SyncError(f"source {SOURCE} is missing from the sources table")
        return max(0.0, row[0])

    def push_slot(self, seconds: float = 0) -> None:
        """After an answer: no request before min_interval_seconds (or
        ``seconds``, if longer) from now."""
        with self.conn.transaction():
            self.conn.execute(sql.SQL(
                "UPDATE {} SET next_allowed_at = greatest(next_allowed_at,"
                " now() + make_interval(secs => greatest(min_interval_seconds::float8, %s))) WHERE name = %s"
            ).format(_t(self.schema, "sources")), (float(seconds), SOURCE))

    def save(self, cursor: dict, page_counts: dict | None = None, error: str | None = None) -> None:
        """Inside a transaction: write the cursor (and a page's counts)."""
        counts = page_counts or {}
        self.conn.execute(sql.SQL(
            "UPDATE {} SET last_complete_until = %s, window_from = %s, window_until = %s,"
            " resumption_token = %s, token_expires = %s, pages = pages + %s, records = records + %s,"
            " last_page_at = CASE WHEN %s THEN now() ELSE last_page_at END, last_error = %s, updated_at = now()"
            " WHERE source = %s AND set_spec = %s").format(_t(self.schema, "harvests")),
            (cursor["last_complete_until"], cursor["window_from"], cursor["window_until"], cursor["token"],
             cursor["token_expires"], 1 if page_counts else 0,
             counts.get("records", 0) + counts.get("deletions", 0), bool(page_counts),
             error[:2000] if error else None, SOURCE, cursor["set"]))

    # -- finishing --------------------------------------------------------------------

    def result(self, cursor: dict, caught_up: bool) -> dict:
        until = cursor["last_complete_until"]
        return {"set": cursor["set"], "pages": self.pages, "windows_completed": self.windows,
                "caught_up": caught_up, "retries": self.retries,
                "last_complete_until": until.isoformat() if until else None, **self.totals}

    def fail(self, cursor: dict | None, reason: str) -> str:
        reason = documents.clean_text(reason, 1000) or "failed"
        with self.conn.transaction():
            self.own()
            if cursor is not None:
                self.save(cursor, error=reason)
            queue.fail(self.conn, self.schema, self.job["id"], self.worker.id, f"The arXiv sync failed: {reason}")
        log.warning("arXiv sync %s failed: %s", self.job["id"], reason)
        return "failed"

    def complete(self, cursor: dict, caught_up: bool) -> str:
        with self.conn.transaction():
            done = queue.complete(self.conn, self.schema, self.job["id"], self.worker.id,
                                  self.result(cursor, caught_up))
        return "succeeded" if done else "lost (lease lapsed)"

    # -- the walk ---------------------------------------------------------------------

    def params(self, cursor: dict) -> dict:
        if cursor["token"]:
            return {"verb": "ListRecords", "resumptionToken": cursor["token"]}
        return {"verb": "ListRecords", "metadataPrefix": METADATA_PREFIX, "set": cursor["set"],
                "from": cursor["window_from"].isoformat(), "until": cursor["window_until"].isoformat()}

    def run(self) -> str:
        try:
            set_spec, start, max_pages = check_payload(self.job["payload"])
            cursor = self.open_cursor(set_spec, start)
        except (ValueError, SyncError) as exc:
            return self.fail(None, str(exc))
        deadline = time.monotonic() + self.spec.timeout_seconds
        restarts = 0
        try:
            while True:
                today = dt.datetime.now(dt.timezone.utc).date()
                if cursor["window_from"] is None:
                    begin = cursor["last_complete_until"] or cursor["start_date"]
                    cursor.update(window_from=begin, window_until=min(begin + dt.timedelta(days=WINDOW_DAYS), today),
                                  token=None, token_expires=None)
                    restarts = 0
                if cursor["token"] and cursor["token_expires"] \
                        and cursor["token_expires"] <= dt.datetime.now(dt.timezone.utc):
                    cursor.update(token=None, token_expires=None)  # expired: restart the window
                if max_pages is not None and self.pages >= max_pages:
                    return self.complete(cursor, caught_up=False)
                wait = self.take_slot()
                if time.monotonic() + wait >= deadline:
                    return self.complete(cursor, caught_up=False)
                self.worker.sleep(self.job, wait)
                page = self.fetch(cursor)
                if page is None:
                    continue  # to be retried
                if page.error_code == "badResumptionToken":
                    restarts += 1
                    if restarts > MAX_TOKEN_RESTARTS:
                        raise SyncError("arXiv refused this window's resumption token too many times")
                    cursor.update(token=None, token_expires=None)
                    continue
                if page.error_code not in (None, "noRecordsMatch"):
                    raise SyncError(f"arXiv answered {page.error_code}: {page.error_message or ''}".strip())
                if self.record(cursor, page, today):
                    return self.complete(cursor, caught_up=True)
        except SyncError as exc:
            return self.fail(cursor, str(exc))
        except documents.PageError as exc:
            return self.fail(cursor, f"arXiv's answer did not pass the checks: {exc}")
        except psycopg.DatabaseError as exc:
            if isinstance(exc, psycopg.OperationalError):
                raise
            return self.fail(cursor, f"the database refused a page: {type(exc).__name__}")

    def fetch(self, cursor: dict) -> documents.Page | None:
        """One request: the checked page, or None after a failed request
        that will be retried. Raises SyncError when it will not be."""
        payload = {"base_url": self.worker.settings.arxiv_oai_url, "params": self.params(cursor),
                   "timeout_seconds": REQUEST_TIMEOUT_SECONDS}
        try:
            out = self.worker.run_child(self.job, self.spec.target, payload, self.spec.child_timeout_seconds,
                                        self.spec.child_output_max_bytes)
        except child.JobFailed as exc:
            self.retry(f"the request failed: {exc}", None)
            return None
        status, retry_after = response_status(out)
        if status in RETRY_STATUSES:
            self.retry(f"arXiv answered HTTP {status}", retry_after)
            return None
        if status != 200:
            raise SyncError(f"arXiv answered HTTP {status}")
        self.push_slot()
        self.retries_in_a_row = 0
        return documents.clean_page(out.get("page"))

    def retry(self, reason: str, retry_after: int | None) -> None:
        self.retries += 1
        self.retries_in_a_row += 1
        if self.retries_in_a_row > MAX_RETRIES:
            raise SyncError(f"{reason} ({MAX_RETRIES} retries in a row)")
        delay = RETRY_SECONDS if retry_after is None else min(max(retry_after, 1), MAX_RETRY_AFTER_SECONDS)
        log.info("arXiv sync %s: %s; retrying in %d s", self.job["id"], reason, delay)
        self.push_slot(delay)

    def record(self, cursor: dict, page: documents.Page, today: dt.date) -> bool:
        """Record a page and move the cursor in one transaction; True once
        the window that reaches today is complete."""
        reached_end = page.resumption_token is None
        server_day = page.response_date.date()
        new = dict(cursor)
        if reached_end:
            until = max(new["start_date"], min(new["window_until"], server_day))
            if new["last_complete_until"] is None or until > new["last_complete_until"]:
                new["last_complete_until"] = until
            new.update(window_from=None, window_until=None, token=None, token_expires=None)
        else:
            new.update(token=page.resumption_token, token_expires=page.token_expires)
        with self.conn.transaction():
            self.own()
            counts = documents.upsert_page(self.conn, self.schema, page)
            self.save(new, page_counts=counts)
        caught_up = reached_end and cursor["window_until"] >= min(today, server_day)
        cursor.clear()
        cursor.update(new)
        self.pages += 1
        self.windows += reached_end
        for key in COUNTS:
            self.totals[key] += counts[key]
        log.info("arXiv sync %s: page %d of set %s recorded (%d records, %d new)", self.job["id"], self.pages,
                 cursor["set"], counts["records"], counts["inserted"])
        return caught_up


def run(worker, job: dict, spec) -> str:
    """The arxiv_sync driver (backend/ai/kinds.py); returns how the job ended."""
    return _Sync(worker, job, spec).run()
