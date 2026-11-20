"""The AI job queue (tables ``jobs`` and ``workers`` in the AI schema).

Only the application enqueues: ``enqueue`` runs as qs_app, which holds
INSERT on ``jobs`` and no other write there. Everything after that runs in
the AI worker, as qs_ai_worker:

* ``claim_next`` takes the oldest queued job whose stage is switched on,
  with FOR UPDATE SKIP LOCKED;
* ``renew`` extends the claim's lease while the job runs, and tells the
  worker when the job is no longer its own or its stage has been switched off;
* ``complete`` and ``fail`` finish it;
* ``release`` hands it back, uncounted, when the worker stops or the stage
  is switched off;
* ``reap_expired`` recovers the jobs of workers that died.

Every function takes the AI schema's name and a psycopg connection, and
runs its statements in the caller's transaction. Callers keep each
transaction short: no transaction stays open while a job runs.
"""
from __future__ import annotations

import json
import uuid

from psycopg import Connection, errors, sql
from psycopg.types.json import Jsonb

from . import flags
from .kinds import KINDS

WORKER_GONE_ERROR = "The AI worker stopped responding while running this job."
# A worker that has not been seen for this long is presumed gone.
STALE_WORKER_SECONDS = 300
MAX_PAYLOAD_BYTES = 64 * 1024


class EnqueueError(ValueError):
    pass


class StageOff(EnqueueError):
    """The job's stage, or the whole AI tier, is switched off."""


def _t(schema: str, name: str) -> sql.Composable:
    return sql.Identifier(schema, name)


def enqueue(conn: Connection, schema: str, kind: str, payload: dict, requested_by: str) -> str:
    """Queue a job of ``kind``; returns its id. Refused while its stage is off."""
    spec = KINDS.get(kind)
    if spec is None:
        raise EnqueueError(f"unknown AI job kind {kind!r}")
    requested_by = (requested_by or "").strip()
    if not requested_by:
        raise EnqueueError("say who requests the job")
    if not isinstance(payload, dict) or len(json.dumps(payload)) > MAX_PAYLOAD_BYTES:
        raise EnqueueError(f"an AI job's payload is a JSON object of at most {MAX_PAYLOAD_BYTES} bytes")
    if not flags.enabled(conn, schema, spec.stage):
        raise StageOff(f"the AI stage {spec.stage!r} is switched off "
                       f"(it runs only while {flags.MASTER} and {spec.stage} are both on)")
    job_id = str(uuid.uuid4())
    try:
        with conn.transaction():  # a savepoint: a refusal leaves the caller's transaction usable
            conn.execute(sql.SQL("INSERT INTO {} (id, kind, payload, requested_by, max_attempts)"
                                 " VALUES (%s, %s, %s, %s, %s)").format(_t(schema, "jobs")),
                         (job_id, kind, Jsonb(payload), requested_by[:320], spec.max_attempts))
    except errors.UniqueViolation:
        # jobs_one_arxiv_sync (migration 0014) and any later one-at-a-time index.
        raise EnqueueError(f"an AI job of kind {kind!r} is already queued or running") from None
    return job_id


def claim_next(conn: Connection, schema: str, worker_id: str, lease_seconds: float) -> dict | None:
    """Claim the oldest queued job whose kind this code knows and whose
    stage is switched on, or return None.

    The flags are read inside this statement, so a job whose stage is off
    stays queued, and is claimed by the first poll after it is switched on.
    """
    names = list(KINDS)
    stages = [KINDS[name].stage for name in names]
    row = conn.execute(sql.SQL("""
        UPDATE {jobs} j SET status = 'running', attempts = j.attempts + 1, worker_id = %(worker)s,
               started_at = now(), lease_expires_at = now() + make_interval(secs => %(lease)s)
        FROM (SELECT q.id FROM {jobs} q
              JOIN unnest(%(kinds)s::text[], %(stages)s::text[]) AS k(kind, stage) ON k.kind = q.kind
              WHERE q.status = 'queued'
                AND EXISTS (SELECT 1 FROM {flags} m WHERE m.name = %(master)s AND m.enabled)
                AND EXISTS (SELECT 1 FROM {flags} s WHERE s.name = k.stage AND s.enabled)
              ORDER BY q.created_at, q.id
              LIMIT 1
              FOR UPDATE OF q SKIP LOCKED) AS pick
        WHERE j.id = pick.id
        RETURNING j.id, j.kind, j.payload, j.attempts, j.max_attempts
    """).format(jobs=_t(schema, "jobs"), flags=_t(schema, "flags")),
        {"worker": worker_id, "lease": float(lease_seconds), "kinds": names, "stages": stages,
         "master": flags.MASTER}).fetchone()
    if row is None:
        return None
    job_id, kind, payload, attempts, max_attempts = row
    return {"id": job_id, "kind": kind, "payload": payload, "attempts": attempts, "max_attempts": max_attempts}


def renew(conn: Connection, schema: str, job_id: str, worker_id: str, lease_seconds: float) -> str:
    """Extend the lease of a running job.

    Returns "ok"; "lost" if the job is no longer this worker's (its lease
    lapsed and it was reaped); or "off" if its stage has been switched off,
    in which case the lease is left as it is and the worker stops the job.
    """
    row = conn.execute(sql.SQL("SELECT kind FROM {} WHERE id = %s AND worker_id = %s AND status = 'running'"
                               " FOR UPDATE").format(_t(schema, "jobs")), (job_id, worker_id)).fetchone()
    if row is None:
        return "lost"
    spec = KINDS.get(row[0])
    if spec is None or not flags.enabled(conn, schema, spec.stage):
        return "off"
    conn.execute(sql.SQL("UPDATE {} SET lease_expires_at = now() + make_interval(secs => %s)"
                         " WHERE id = %s").format(_t(schema, "jobs")), (float(lease_seconds), job_id))
    return "ok"


def _finish(conn: Connection, schema: str, job_id: str, worker_id: str, status: str,
            result: dict | None, error: str | None) -> bool:
    cursor = conn.execute(sql.SQL(
        "UPDATE {} SET status = %s, result = %s, error = %s, finished_at = now(), lease_expires_at = NULL,"
        " worker_id = NULL WHERE id = %s AND worker_id = %s AND status = 'running'").format(_t(schema, "jobs")),
        (status, Jsonb(result) if result is not None else None, error, job_id, worker_id))
    return cursor.rowcount == 1


def complete(conn: Connection, schema: str, job_id: str, worker_id: str, result: dict) -> bool:
    """Record a job's result; False if the job was no longer this worker's."""
    return _finish(conn, schema, job_id, worker_id, "succeeded", result, None)


def fail(conn: Connection, schema: str, job_id: str, worker_id: str, error: str) -> bool:
    """Record that a job failed; False if the job was no longer this worker's."""
    return _finish(conn, schema, job_id, worker_id, "failed", None, (error or "failed")[:2000])


def release(conn: Connection, schema: str, job_id: str, worker_id: str) -> bool:
    """Hand a running job back to the queue without counting the attempt
    (its worker is stopping, or its stage was switched off)."""
    cursor = conn.execute(sql.SQL(
        "UPDATE {} SET status = 'queued', attempts = greatest(attempts - 1, 0), worker_id = NULL,"
        " lease_expires_at = NULL, started_at = NULL WHERE id = %s AND worker_id = %s AND status = 'running'"
    ).format(_t(schema, "jobs")), (job_id, worker_id))
    return cursor.rowcount == 1


def reap_expired(conn: Connection, schema: str) -> tuple[int, int]:
    """Requeue running jobs whose lease lapsed, or fail those out of
    attempts; returns (requeued, failed)."""
    jobs = _t(schema, "jobs")
    requeued = conn.execute(sql.SQL(
        "UPDATE {} SET status = 'queued', worker_id = NULL, lease_expires_at = NULL, started_at = NULL"
        " WHERE status = 'running' AND lease_expires_at < now() AND attempts < max_attempts").format(jobs)).rowcount
    failed = conn.execute(sql.SQL(
        "UPDATE {} SET status = 'failed', error = %s, finished_at = now(), worker_id = NULL,"
        " lease_expires_at = NULL WHERE status = 'running' AND lease_expires_at < now()").format(jobs),
        (WORKER_GONE_ERROR,)).rowcount
    return requeued, failed


def purge_finished(conn: Connection, schema: str, retention_days: float) -> int:
    return conn.execute(sql.SQL(
        "DELETE FROM {} WHERE status IN ('succeeded', 'failed')"
        " AND finished_at < now() - make_interval(secs => %s)").format(_t(schema, "jobs")),
        (float(retention_days) * 86400,)).rowcount


def heartbeat(conn: Connection, schema: str, worker_id: str, hostname: str, pid: int, started_at) -> None:
    workers = _t(schema, "workers")
    conn.execute(sql.SQL(
        "INSERT INTO {} (id, hostname, pid, started_at, last_seen_at) VALUES (%s, %s, %s, %s, now())"
        " ON CONFLICT (id) DO UPDATE SET last_seen_at = now()").format(workers),
        (worker_id, hostname[:255], pid, started_at))
    conn.execute(sql.SQL("DELETE FROM {} WHERE last_seen_at < now() - make_interval(secs => %s)").format(workers),
                 (float(STALE_WORKER_SECONDS),))


def unregister(conn: Connection, schema: str, worker_id: str) -> None:
    conn.execute(sql.SQL("DELETE FROM {} WHERE id = %s").format(_t(schema, "workers")), (worker_id,))


def get(conn: Connection, schema: str, job_id: str) -> dict | None:
    row = conn.execute(sql.SQL(
        "SELECT id, kind, status, payload, result, error, requested_by, attempts, max_attempts,"
        " created_at, started_at, finished_at FROM {} WHERE id = %s").format(_t(schema, "jobs")),
        (job_id,)).fetchone()
    if row is None:
        return None
    keys = ("id", "kind", "status", "payload", "result", "error", "requested_by", "attempts", "max_attempts",
            "created_at", "started_at", "finished_at")
    return dict(zip(keys, row))
