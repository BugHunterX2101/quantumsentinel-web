"""The AI worker: ``python -m backend.ai.worker`` with QS_PROCESS_ROLE=ai-worker.

It connects as qs_ai_worker (AI_DATABASE_URL) and runs AI jobs one at a
time, each in a job process of its own (backend/ai/child.py). It imports
nothing of the trusted tier and needs none of its secrets: in production it
refuses to start if it is given any (backend/ai/settings.py). It signs
nothing and writes no audit event; the application audits what operators
do to the AI tier (``manage set-flag``, ``manage ai-selftest``).

No transaction stays open while a job runs. Each step (claim, lease
renewal, finish, heartbeat) is a short transaction of its own. While a job
runs, the worker renews its lease every AI_WORKER_HEARTBEAT_SECONDS and
checks the job's stage. If the stage has been switched off it stops the job
and hands it back to the queue, uncounted, so a switch takes effect at
once, not when the job ends. A job that runs past its kind's timeout is
killed and fails.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid

import psycopg
from psycopg import sql

from . import child, queue
from .kinds import KINDS
from .settings import Settings, SettingsError, load

log = logging.getLogger("backend.ai.worker")
HOUSEKEEPING_SECONDS = 30
PURGE_SECONDS = 3600


class AiWorker:
    def __init__(self, settings: Settings, hostname: str | None = None):
        self.settings = settings
        self.schema = settings.schema
        self.id = f"{hostname or socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        self.started_at = dt.datetime.now(dt.timezone.utc)
        self.conn: psycopg.Connection | None = None
        self._stopping = threading.Event()
        self._last_housekeeping = self._last_purge = float("-inf")

    # -- connection ------------------------------------------------------------

    def connect(self) -> None:
        self.close()
        conn = psycopg.connect(self.settings.database_url, autocommit=True, connect_timeout=10,
                               application_name="quantumsentinel-ai-worker")
        conn.execute("SELECT set_config('TimeZone', 'UTC', false), set_config('search_path', %s, false)",
                     (sql.Identifier(self.schema).as_string(conn),))
        self.conn = conn

    def close(self) -> None:
        if self.conn is not None:
            try:
                self.conn.close()
            finally:
                self.conn = None

    def schema_ready(self) -> bool:
        return self.conn.execute("SELECT to_regclass(%s) IS NOT NULL",
                                 (sql.Identifier(self.schema, "jobs").as_string(self.conn),)).fetchone()[0]

    # -- loop -------------------------------------------------------------------

    def stop(self) -> None:
        self._stopping.set()

    def _heartbeat(self) -> None:
        queue.heartbeat(self.conn, self.schema, self.id, socket.gethostname(), os.getpid(), self.started_at)

    def housekeeping(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_housekeeping < HOUSEKEEPING_SECONDS:
            return
        with self.conn.transaction():
            self._heartbeat()
            requeued, failed = queue.reap_expired(self.conn, self.schema)
        if requeued or failed:
            log.warning("reaped AI jobs of workers that stopped responding: %d requeued, %d failed",
                        requeued, failed)
        if now - self._last_purge >= PURGE_SECONDS:
            with self.conn.transaction():
                queue.purge_finished(self.conn, self.schema, self.settings.retention_days)
            self._last_purge = now
        self._last_housekeeping = now

    def run_once(self) -> str | None:
        """Claim and run one job; returns how it ended, or None if none was claimable."""
        self.housekeeping()
        with self.conn.transaction():
            job = queue.claim_next(self.conn, self.schema, self.id, self.settings.lease_seconds)
        if job is None:
            return None
        log.info("AI job %s (%s) claimed, attempt %d", job["id"], job["kind"], job["attempts"])
        ended = self.execute(job)
        log.info("AI job %s %s", job["id"], ended)
        return ended

    def execute(self, job: dict) -> str:
        spec = KINDS[job["kind"]]
        process = child.start(self.settings.job_memory_bytes, spec.timeout_seconds)
        deadline = time.monotonic() + spec.timeout_seconds
        data: bytes | None = child.request(spec.target, job["payload"])
        while True:
            wait = max(0.0, min(self.settings.heartbeat_seconds, deadline - time.monotonic()))
            try:
                stdout, stderr = process.communicate(input=data, timeout=wait)
                break
            except subprocess.TimeoutExpired:
                data = None  # sent with the first call
            if time.monotonic() >= deadline:
                _kill(process)
                with self.conn.transaction():
                    queue.fail(self.conn, self.schema, job["id"], self.id,
                               f"The AI job ran longer than {spec.timeout_seconds:g} s and was stopped.")
                return "timed out"
            if self._stopping.is_set():
                _kill(process)
                with self.conn.transaction():
                    queue.release(self.conn, self.schema, job["id"], self.id)
                return "released (worker stopping)"
            with self.conn.transaction():
                state = queue.renew(self.conn, self.schema, job["id"], self.id, self.settings.lease_seconds)
                self._heartbeat()
            if state == "lost":
                _kill(process)
                return "lost (lease lapsed)"
            if state == "off":
                _kill(process)
                with self.conn.transaction():
                    queue.release(self.conn, self.schema, job["id"], self.id)
                return "released (stage switched off)"
        try:
            result = child.outcome(process.returncode, stdout, stderr, self.settings.result_max_bytes)
        except child.JobFailed as exc:
            with self.conn.transaction():
                queue.fail(self.conn, self.schema, job["id"], self.id, f"The AI job failed: {exc}")
            return "failed"
        with self.conn.transaction():
            done = queue.complete(self.conn, self.schema, job["id"], self.id, result)
        return "succeeded" if done else "lost (lease lapsed)"

    def run_forever(self) -> None:
        log.info("AI worker %s started (schema %s)", self.id, self.schema)
        try:
            while not self._stopping.is_set():
                try:
                    if self.conn is None or self.conn.closed:
                        self.connect()
                        if not self.schema_ready():
                            log.info("waiting for the AI schema %s to be migrated", self.schema)
                            self.close()
                            self._stopping.wait(self.settings.poll_seconds)
                            continue
                        self.housekeeping(force=True)
                    if self.run_once() is None:
                        self._stopping.wait(self.settings.poll_seconds)
                except psycopg.OperationalError as exc:
                    log.warning("AI worker lost its database connection: %s", exc)
                    self.close()
                    self._stopping.wait(2)
        finally:
            if self.conn is not None and not self.conn.closed:
                try:
                    with self.conn.transaction():
                        queue.unregister(self.conn, self.schema, self.id)
                except psycopg.Error:
                    pass
            self.close()
            log.info("AI worker %s stopped", self.id)


def _kill(process: subprocess.Popen) -> None:
    process.kill()
    try:
        process.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        settings = load()
    except SettingsError as exc:
        print(f"ai-worker: {exc}", file=sys.stderr)
        return 2
    worker = AiWorker(settings)
    for name in ("SIGTERM", "SIGINT"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), lambda *_: worker.stop())
    worker.run_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
