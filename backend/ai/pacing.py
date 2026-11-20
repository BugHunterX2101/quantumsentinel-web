"""What every AI driver that talks to a remote service shares.

* ``own``: inside a transaction, lock the job row or raise Interrupted if
  the job is no longer this worker's, so a worker that lost its lease never
  records anything.
* ``take_slot``/``push_slot``: the rate slot of a source (table ``sources``,
  migration 0014). Every worker takes the next slot before a request and
  moves it on by the source's ``min_interval_seconds``, so all of them
  together keep the source's rate limit; an answer, or a Retry-After, pushes
  it further. Each runs in a short transaction of its own, never across a
  request.
* ``response_status``: the HTTP status and Retry-After a job process
  reported, checked.
"""
from __future__ import annotations

from psycopg import Connection, sql

from .worker import Interrupted


class SourceError(Exception):
    """The driver cannot go on; its job fails with this reason."""


def _t(schema: str, name: str) -> sql.Composable:
    return sql.Identifier(schema, name)


def own(conn: Connection, schema: str, job_id: str, worker_id: str) -> None:
    row = conn.execute(sql.SQL(
        "SELECT 1 FROM {} WHERE id = %s AND worker_id = %s AND status = 'running' FOR UPDATE"
    ).format(_t(schema, "jobs")), (job_id, worker_id)).fetchone()
    if row is None:
        raise Interrupted("lost (lease lapsed)", release=False)


def take_slot(conn: Connection, schema: str, source: str) -> float:
    """Take the next request slot of ``source``; returns the seconds until it."""
    with conn.transaction():
        row = conn.execute(sql.SQL(
            "UPDATE {} SET next_allowed_at = greatest(next_allowed_at, now())"
            " + make_interval(secs => min_interval_seconds::float8) WHERE name = %s"
            " RETURNING extract(epoch FROM next_allowed_at"
            " - make_interval(secs => min_interval_seconds::float8) - now())::float8"
        ).format(_t(schema, "sources")), (source,)).fetchone()
    if row is None:
        raise SourceError(f"source {source} is missing from the sources table")
    return max(0.0, row[0])


def push_slot(conn: Connection, schema: str, source: str, seconds: float = 0) -> None:
    """After an answer: no request to ``source`` before its
    min_interval_seconds (or ``seconds``, if longer) from now."""
    with conn.transaction():
        conn.execute(sql.SQL(
            "UPDATE {} SET next_allowed_at = greatest(next_allowed_at,"
            " now() + make_interval(secs => greatest(min_interval_seconds::float8, %s))) WHERE name = %s"
        ).format(_t(schema, "sources")), (float(seconds), source))


def response_status(out) -> tuple[int, int | None]:
    """The HTTP status and Retry-After a job process reported; raises SourceError."""
    if not isinstance(out, dict):
        raise SourceError("the job process returned no result object")
    status, retry_after = out.get("http_status"), out.get("retry_after")
    if isinstance(status, bool) or not isinstance(status, int) or not 100 <= status <= 599:
        raise SourceError("the job process returned no valid HTTP status")
    if retry_after is not None and (isinstance(retry_after, bool) or not isinstance(retry_after, int)
                                    or retry_after < 0):
        retry_after = None
    return status, retry_after
