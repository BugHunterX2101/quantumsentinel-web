"""Research job worker.

    python -m backend.worker

Claims queued research jobs (see services/research_jobs.py) and runs each in
a child process, so the API process never computes research. The child is
reused between jobs and replaced when a job times out, is cancelled, crashes
it, or after RECYCLE_AFTER_JOBS jobs. The worker renews its job's lease while
the child computes; run several workers to run several jobs at once.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import logging
import multiprocessing as mp
import os
import secrets
import signal
import socket
import sys
import threading
import time
import traceback

from .config import (PROCESS_ROLE, RESEARCH_JOB_LEASE_SECONDS, RESEARCH_JOB_TIMEOUT_SECONDS,
                     RESEARCH_WORKER_POLL_SECONDS)
from .database import SessionLocal, head_revision, schema_is_current

log = logging.getLogger("backend.worker")

RECYCLE_AFTER_JOBS = 50
# A new job process must import the research code within this long; its
# job's own time limit only starts once it has (see _child_main).
CHILD_BOOT_SECONDS = 120
_TASKS_MODULE = "backend.services.research_tasks"
_READY = "ready"
PURGE_INTERVAL_SECONDS = 3600
WORKER_HEARTBEAT_SECONDS = 5
GENERIC_FAILURE = "Research job failed"
# What a job process keeps of its worker's environment: what the platform and
# the research libraries need (paths, locale, temporary and cache
# directories, proxies, CA bundles, thread counts) and ENVIRONMENT. Nothing
# else: not the database URL, not a key, not a *_FILE pointer to a secret.
# Research code runs on users' inputs and needs none of them.
_JOB_ENVIRONMENT = frozenset({
    "PATH", "HOME", "USER", "LOGNAME", "LANG", "LANGUAGE", "TZ", "TMPDIR", "TEMP", "TMP",
    "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "MPLCONFIGDIR",
    "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "PATHEXT", "USERPROFILE", "APPDATA",
    "LOCALAPPDATA", "PROGRAMDATA",
    "__CF_USER_TEXT_ENCODING",  # macOS sets it in every process: the user's text encoding
    "VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH", "PYTHONUTF8", "PYTHONIOENCODING", "PYTHONUNBUFFERED",
    "PYTHONDONTWRITEBYTECODE", "PYTHONHASHSEED",
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "ALL_PROXY",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
    "ENVIRONMENT",
})


def job_environment(environ) -> dict[str, str]:
    """The environment a job process starts with, from its worker's."""
    kept = {name: value for name, value in environ.items()
            if name.upper() in _JOB_ENVIRONMENT or name.upper().startswith("LC_")}
    # Production's startup checks ask a job process for no secrets (config.py).
    kept["QS_PROCESS_ROLE"] = "research-job"
    return kept


@contextlib.contextmanager
def _job_process_environment():
    """multiprocessing starts a child with this process's environment as it
    is at that moment, and has no way to pass another, so os.environ holds the
    job process's environment while it starts. The worker's other thread (the
    lifeline watcher) never reads it."""
    saved = dict(os.environ)
    os.environ.clear()
    os.environ.update(job_environment(saved))
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


# --------------------------------------------------------------------------
# Child process: runs one task at a time, never touches the database
# --------------------------------------------------------------------------

def _resolve(target: str):
    module_name, _, attr = target.partition(":")
    return getattr(importlib.import_module(module_name), attr)


def _run_target(target: str, params: dict) -> tuple:
    """Run a task; return ("ok", result, audit, trials) or ("error", status, detail)."""
    from .services.research_jobs import json_safe
    from .services.research_tasks import TaskError
    try:
        output = _resolve(target)(params)
    except TaskError as exc:
        return ("error", exc.status_code, exc.detail)
    except Exception:
        # Details stay in the worker log; users get a generic message.
        log.error("research task %s raised:\n%s", target, traceback.format_exc())
        return ("error", 500, GENERIC_FAILURE)
    trials = None
    if output.trials is not None:
        trials = {"family": output.trials.family, "configs": json_safe(output.trials.configs),
                  "source": output.trials.source}
    return ("ok", json_safe(output.result), json_safe(output.audit), trials)


def _child_main(conn) -> None:
    # Interrupts go to the worker, which decides whether to stop the child.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s job-process %(message)s")
    # Import the research code (numpy, scipy, sklearn: seconds) before saying
    # ready, so a job's time limit measures the job, not this process booting.
    importlib.import_module(_TASKS_MODULE)
    conn.send(_READY)
    while True:
        try:
            message = conn.recv()
        except EOFError:
            return
        if message is None:
            return
        target, params = message
        conn.send(_run_target(target, params))


class JobProcess:
    """A reusable child process that runs tasks with a hard time limit."""

    def __init__(self, recycle_after: int = RECYCLE_AFTER_JOBS):
        self._ctx = mp.get_context("spawn")
        self._proc = None
        self._conn = None
        self._jobs = 0
        self._recycle_after = recycle_after

    def _start(self) -> None:
        parent, child = self._ctx.Pipe()
        self._proc = self._ctx.Process(target=_child_main, args=(child,), daemon=True,
                                       name="qs-research-job")
        with _job_process_environment():
            self._proc.start()
        child.close()
        self._conn = parent
        self._jobs = 0

    def stop(self) -> None:
        if self._proc is None:
            return
        try:
            self._conn.send(None)
            self._proc.join(5)
        except (OSError, EOFError, BrokenPipeError):
            pass
        self.kill()

    def kill(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            self._proc.kill()
            self._proc.join(10)
        if self._conn is not None:
            self._conn.close()
        self._proc = self._conn = None

    def run(self, target: str, params: dict, timeout: float, on_tick, tick_seconds: float = 1.0):
        """Run ``target(params)`` in the child.

        ``on_tick()`` is called about every ``tick_seconds`` while it runs; if
        it returns a string, the child is killed and that string returned
        ("cancelled", "lost", "shutdown"). Otherwise returns the task's
        ("ok", ...) / ("error", status, detail) tuple, ("timeout",) or
        ("crashed", exitcode).
        """
        deadline = None  # starts when the child is ready to run the job
        if self._proc is None or not self._proc.is_alive():
            self._start()
            boot_deadline = time.monotonic() + CHILD_BOOT_SECONDS
        else:
            deadline = time.monotonic() + timeout
        self._conn.send((target, params))
        while True:
            if self._conn.poll(tick_seconds):
                try:
                    outcome = self._conn.recv()
                except (EOFError, OSError):
                    return self._crashed()
                if outcome == _READY:
                    deadline = time.monotonic() + timeout
                    continue
                self._jobs += 1
                if self._jobs >= self._recycle_after:
                    self.stop()
                return outcome
            if not self._proc.is_alive():
                return self._crashed()
            if deadline is None and time.monotonic() >= boot_deadline:
                log.error("job process did not start within %ss", CHILD_BOOT_SECONDS)
                return self._crashed()
            if deadline is not None and time.monotonic() >= deadline:
                self.kill()
                return ("timeout",)
            stop = on_tick()
            if stop:
                self.kill()
                return (stop,)

    def _crashed(self):
        exitcode = self._proc.exitcode if self._proc is not None else None
        self.kill()
        return ("crashed", exitcode)


# --------------------------------------------------------------------------
# Worker loop
# --------------------------------------------------------------------------

class Worker:
    def __init__(self, job_process: JobProcess | None = None,
                 lease_seconds: float = RESEARCH_JOB_LEASE_SECONDS,
                 timeout_seconds: float = RESEARCH_JOB_TIMEOUT_SECONDS,
                 poll_seconds: float = RESEARCH_WORKER_POLL_SECONDS,
                 session_factory=SessionLocal):
        self.hostname = socket.gethostname()
        self.pid = os.getpid()
        # Fits research_workers.id (128) whatever the hostname's length.
        self.id = f"{self.hostname[:100]}:{self.pid}:{secrets.token_hex(3)}"
        self.job_process = job_process or JobProcess()
        self.lease_seconds = lease_seconds
        self.timeout_seconds = timeout_seconds
        self.poll_seconds = poll_seconds
        self.session_factory = session_factory
        self.stop_event = threading.Event()
        self._last_heartbeat = 0.0
        self._last_purge = 0.0

    # -- bookkeeping --------------------------------------------------------
    def _heartbeat(self, db, current_job_id: str | None, force: bool = False) -> None:
        from .services import research_jobs
        now = time.monotonic()
        if force or now - self._last_heartbeat >= WORKER_HEARTBEAT_SECONDS:
            research_jobs.heartbeat_worker(db, self.id, self.hostname, self.pid, current_job_id)
            self._last_heartbeat = now

    def _housekeeping(self, db) -> None:
        from .services import research_jobs
        reaped = research_jobs.reap_expired(db)
        if reaped:
            log.warning("reclaimed %d research job(s) from unresponsive workers", reaped)
        now = time.monotonic()
        if now - self._last_purge >= PURGE_INTERVAL_SECONDS:
            self._last_purge = now
            purged = research_jobs.purge_finished(db)
            if purged:
                log.info("purged %d finished research job(s) past retention", purged)

    # -- one iteration --------------------------------------------------------
    def run_once(self) -> bool:
        """Claim and run at most one job. Returns True if a job was run."""
        from .services import research_jobs
        db = self.session_factory()
        try:
            self._heartbeat(db, None)
            self._housekeeping(db)
            job = research_jobs.claim_next(db, self.id, self.lease_seconds)
            if job is None:
                return False
            self._execute(db, job)
            return True
        finally:
            db.close()

    def _execute(self, db, job) -> None:
        from .services import research_jobs
        job_id, attempt = job.id, job.attempts
        kind = research_jobs.KINDS.get(job.kind)
        log.info("running research job %s (%s), attempt %d", job.id, job.kind, attempt)
        if kind is None:
            research_jobs.fail(db, job, self.id, attempt, 500, "Unknown research job kind")
            return
        self._heartbeat(db, job.id, force=True)
        renew_every = max(1.0, self.lease_seconds / 3)
        last_renew = time.monotonic()

        def on_tick():
            nonlocal last_renew
            if self.stop_event.is_set():
                return "shutdown"
            now = time.monotonic()
            if now - last_renew < renew_every:
                # Between renewals only look for a cancel request, so a
                # cancel takes about one tick rather than a third of a lease.
                return "cancelled" if research_jobs.cancel_requested(db, job_id) else None
            last_renew = now
            state = research_jobs.renew_lease(db, job, self.id, attempt, self.lease_seconds)
            self._heartbeat(db, job.id)
            if state == "cancel":
                return "cancelled"
            if state == "lost":
                return "lost"
            return None

        started = time.monotonic()
        outcome = self.job_process.run(kind.target, job.params_json, self.timeout_seconds, on_tick)
        elapsed = time.monotonic() - started
        status = outcome[0]
        if status == "ok":
            _, result, audit, trials = outcome
            if not research_jobs.complete(db, job, self.id, attempt, result, audit, trials):
                log.warning("research job %s finished after its lease was lost; result discarded", job.id)
        elif status == "error":
            _, code, detail = outcome
            research_jobs.fail(db, job, self.id, attempt, code, detail)
        elif status == "timeout":
            research_jobs.fail(db, job, self.id, attempt, 504,
                               f"The research job exceeded its {self.timeout_seconds:.0f}-second time limit.")
        elif status == "crashed":
            log.error("research job %s: job process exited with code %s", job.id, outcome[1])
            research_jobs.fail(db, job, self.id, attempt, 500,
                               "The research job process stopped unexpectedly.")
        elif status == "cancelled":
            research_jobs.mark_cancelled(db, job, self.id, attempt)
        elif status == "shutdown":
            research_jobs.release(db, job, self.id, attempt)
        elif status == "lost":
            log.warning("research job %s: lease lost; stopped its computation", job.id)
        log.info("research job %s -> %s in %.1f s", job.id, status, elapsed)
        self._heartbeat(db, None, force=True)

    # -- main loop --------------------------------------------------------------
    def serve(self) -> None:
        from .services import research_jobs
        log.info("research worker %s started", self.id)
        try:
            while not self.stop_event.is_set():
                try:
                    ran = self.run_once()
                except Exception:
                    log.exception("research worker iteration failed")
                    ran = False
                if not ran:
                    self.stop_event.wait(self.poll_seconds)
        finally:
            self.job_process.stop()
            db = self.session_factory()
            try:
                research_jobs.remove_worker(db, self.id)
            except Exception:
                log.exception("could not deregister research worker %s", self.id)
            finally:
                db.close()
            log.info("research worker %s stopped", self.id)


def _wait_for_schema(stop_event: threading.Event) -> bool:
    """The API process migrates the schema; wait until it is at this code's revision."""
    waited = 0.0
    while not stop_event.is_set():
        try:
            if schema_is_current():
                return True
        except Exception as exc:
            log.warning("database not reachable yet: %s", exc)
        if waited % 30 == 0:
            log.info("waiting for the database schema to reach revision %s", head_revision())
        stop_event.wait(2)
        waited += 2
    return False


def _prepare_identity() -> None:
    """Sign audit events with the deployment's pinned server key, as the API does."""
    from .config import TRUSTED_SERVER_DSA_FINGERPRINT
    from .services import security_service
    security_service.enforce_identity_pin(TRUSTED_SERVER_DSA_FINGERPRINT)
    db = SessionLocal()
    try:
        security_service.server_identity.register_in_db(db)
    finally:
        db.close()


def _lifeline_closed_probe():
    """A non-blocking check for EOF on stdin.

    Never a blocking read: on Windows a synchronous read pending on stdin
    stalls every DLL load in the process (each new C runtime queries the
    standard handles), which hangs imports such as numpy's.
    """
    if os.name == "nt":
        import _winapi
        import msvcrt
        handle = msvcrt.get_osfhandle(sys.stdin.fileno())

        def closed() -> bool:
            try:
                _winapi.PeekNamedPipe(handle, 0)
                return False
            except OSError:  # ERROR_BROKEN_PIPE: the write end is closed
                return True
    else:
        import select
        fd = sys.stdin.fileno()

        def closed() -> bool:
            readable, _, _ = select.select([fd], [], [], 0)
            return bool(readable) and os.read(fd, 4096) == b""
    return closed


def _watch_lifeline(stop_event: threading.Event, interval: float = 1.0) -> None:
    """Embedded mode: stdin is a pipe whose write end only the API process
    holds. It closes when that process exits, however it exits, and the
    worker then stops (releasing its job) instead of running on as an orphan."""
    closed = _lifeline_closed_probe()
    while not stop_event.wait(interval):
        if closed():
            log.info("API process gone; stopping the embedded research worker")
            stop_event.set()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m backend.worker",
                                     description="Run research jobs from the queue.")
    parser.add_argument("--lifeline", action="store_true",
                        help="stop when stdin closes (set by the API for its embedded worker)")
    args = parser.parse_args(argv)
    if PROCESS_ROLE == "research-job":
        # That role's settings pass production's checks without a signing key.
        parser.error("a research worker cannot run as QS_PROCESS_ROLE=research-job")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    worker = Worker()

    def _stop(signum, _frame):
        log.info("signal %s received; stopping after releasing the current job", signum)
        worker.stop_event.set()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    if args.lifeline:
        threading.Thread(target=_watch_lifeline, args=(worker.stop_event,), daemon=True,
                         name="lifeline").start()
    if not _wait_for_schema(worker.stop_event):
        return 0
    _prepare_identity()
    worker.serve()
    return 0


if __name__ == "__main__":
    sys.exit(main())
