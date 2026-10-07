"""QuantumSentinel — research job queue.

Research is CPU-heavy and slow (market-data downloads, parameter searches),
so the API only records a job here and returns; a worker process
(``python -m backend.worker``) claims it, runs the computation in a child
process and stores the outcome. Clients poll ``GET /api/research/jobs/{id}``.

Every state change is a compare-and-set UPDATE, so concurrent workers and
API processes never apply conflicting transitions:

    queued --claim--> running --complete--> succeeded
                         |     --fail-----> failed
                         |     --cancel---> cancelled
                         +--lease expired--> queued again (or failed after
                                             max_attempts runs)
    queued --cancel--> cancelled

A worker owns a running job only while its lease is current. Each write it
makes is conditional on (worker_id, attempts, status='running'), so a worker
that was presumed dead and replaced can never overwrite the newer attempt.
"""
from __future__ import annotations

import datetime as dt
import math
import threading
from dataclasses import dataclass

import numpy as np
from fastapi.encoders import jsonable_encoder
from sqlalchemy import delete, func, select, text, update
from sqlalchemy.orm import Session

from .. import models, schemas
from ..config import (DATABASE_URL, RESEARCH_JOB_MAX_ATTEMPTS, RESEARCH_JOB_RETENTION_DAYS,
                      RESEARCH_MAX_ACTIVE_JOBS_PER_USER)
from . import research_trials, security_service

QUEUED, RUNNING, SUCCEEDED, FAILED, CANCELLED = "queued", "running", "succeeded", "failed", "cancelled"
ACTIVE = (QUEUED, RUNNING)
TERMINAL = (SUCCEEDED, FAILED, CANCELLED)

# A worker that has not heartbeated for this long is not counted as online.
WORKER_ONLINE_SECONDS = 30

_is_postgres = DATABASE_URL.startswith(("postgresql://", "postgresql+"))

# Conditional UPDATE/DELETE statements here compare stored datetimes. By
# default SQLAlchemy re-evaluates such criteria in Python against objects
# already loaded in the session, and SQLite returns those datetimes naive, so
# the comparison raises. Every caller commits right after, which expires the
# loaded objects anyway, so skip that in-memory synchronisation.
_UNSYNCED = {"synchronize_session": False}


@dataclass(frozen=True)
class JobKind:
    schema: type
    target: str          # "module:function" run in the job process
    audit_action: str


_TASKS = "backend.services.research_tasks"
KINDS: dict[str, JobKind] = {
    "ma_backtest": JobKind(schemas.BacktestRequest, f"{_TASKS}:ma_backtest", "BACKTEST_COMPLETED"),
    "advanced_backtest": JobKind(schemas.AdvancedBacktestRequest, f"{_TASKS}:advanced_backtest",
                                 "ADVANCED_BACKTEST"),
    "walk_forward": JobKind(schemas.WalkForwardRequest, f"{_TASKS}:walk_forward", "WALK_FORWARD"),
    "event_backtest": JobKind(schemas.EventBacktestRequest, f"{_TASKS}:event_backtest", "EVENT_BACKTEST"),
    "alpha": JobKind(schemas.AlphaResearchRequest, f"{_TASKS}:alpha_research", "ALPHA_RESEARCH"),
    "factor_model": JobKind(schemas.FactorModelRequest, f"{_TASKS}:factor_model", "FACTOR_MODEL"),
    "correlation": JobKind(schemas.CorrelationRequest, f"{_TASKS}:correlation", "CORRELATION_ANALYSIS"),
    "optimize": JobKind(schemas.PortfolioOptRequest, f"{_TASKS}:portfolio_optimization", "PORTFOLIO_OPT"),
    "regime": JobKind(schemas.RegimeDetectionRequest, f"{_TASKS}:regime_detection", "REGIME_DETECTION"),
    "neutral_strategy": JobKind(schemas.NeutralStrategyRequest, f"{_TASKS}:neutral_strategy",
                                "NEUTRAL_STRATEGY"),
    "pairs_trading": JobKind(schemas.PairsTradingRequest, f"{_TASKS}:pairs_trading", "PAIRS_TRADING"),
    "latency_benchmark": JobKind(schemas.LatencyBenchmarkRequest, f"{_TASKS}:latency_benchmark",
                                 "LATENCY_BENCHMARK"),
    "report": JobKind(schemas.ReportRequest, f"{_TASKS}:research_report", "RESEARCH_REPORT"),
}


class QueueFull(Exception):
    """The user already has the maximum number of active jobs."""

    def __init__(self, active: int, limit: int):
        super().__init__(f"You already have {active} research jobs queued or running "
                         f"(limit {limit}). Wait for one to finish or cancel one.")
        self.active, self.limit = active, limit


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _aware(value: dt.datetime | None) -> dt.datetime | None:
    # SQLite returns naive datetimes; every stored value is UTC.
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=dt.timezone.utc)
    return value


def _iso(value: dt.datetime | None) -> str | None:
    value = _aware(value)
    return value.isoformat() if value else None


def json_safe(value):
    """Encode a task result exactly as the API used to (FastAPI's encoder),
    plus numpy types, with non-finite floats as null (JSON has no NaN)."""
    encoded = jsonable_encoder(value, custom_encoder={
        np.ndarray: lambda a: a.tolist(),
        np.generic: lambda v: v.item(),
    })
    return _finite(encoded)


def _finite(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite(v) for v in value]
    return value


# --------------------------------------------------------------------------
# API side
# --------------------------------------------------------------------------

# Orders the active-job count and insert between threads of one process;
# on PostgreSQL the user-row lock below extends that across processes.
_enqueue_lock = threading.Lock()


def enqueue(db: Session, user_id: str, kind: str, params: dict) -> models.ResearchJob:
    """Queue a job, or raise QueueFull when the user is at their active limit."""
    if kind not in KINDS:
        raise ValueError(f"unknown research job kind {kind!r}")
    with _enqueue_lock:
        try:
            if _is_postgres:
                db.execute(text("SELECT id FROM users WHERE id = :uid FOR UPDATE"), {"uid": user_id})
            active = int(db.execute(
                select(func.count()).select_from(models.ResearchJob).where(
                    models.ResearchJob.user_id == user_id,
                    models.ResearchJob.status.in_(ACTIVE))
            ).scalar() or 0)
            if active >= RESEARCH_MAX_ACTIVE_JOBS_PER_USER:
                raise QueueFull(active, RESEARCH_MAX_ACTIVE_JOBS_PER_USER)
            job = models.ResearchJob(user_id=user_id, kind=kind, params_json=json_safe(params),
                                     status=QUEUED, max_attempts=RESEARCH_JOB_MAX_ATTEMPTS)
            db.add(job)
            db.commit()
        except Exception:
            db.rollback()
            raise
    db.refresh(job)
    return job


def get_for_user(db: Session, user_id: str, job_id: str) -> models.ResearchJob | None:
    job = db.get(models.ResearchJob, job_id)
    return job if job is not None and job.user_id == user_id else None


def list_for_user(db: Session, user_id: str, limit: int = 20) -> list[models.ResearchJob]:
    return list(db.execute(
        select(models.ResearchJob).where(models.ResearchJob.user_id == user_id)
        .order_by(models.ResearchJob.created_at.desc()).limit(limit)
    ).scalars())


def request_cancel(db: Session, user_id: str, job_id: str) -> models.ResearchJob | None:
    """Cancel a queued job now; ask a running one's worker to stop it.

    A running job stops at the worker's next check (within seconds). If it
    finishes before that check, it keeps its result.
    """
    job = get_for_user(db, user_id, job_id)
    if job is None:
        return None
    if job.status == QUEUED:
        done = db.execute(
            update(models.ResearchJob).execution_options(**_UNSYNCED)
            .where(models.ResearchJob.id == job_id, models.ResearchJob.status == QUEUED)
            .values(status=CANCELLED, finished_at=utcnow())
        ).rowcount
        db.commit()
        db.refresh(job)
        if done:
            return job
    if job.status == RUNNING:
        db.execute(update(models.ResearchJob).execution_options(**_UNSYNCED)
                   .where(models.ResearchJob.id == job_id, models.ResearchJob.status == RUNNING)
                   .values(cancel_requested=True))
        db.commit()
        db.refresh(job)
    return job


def workers_online(db: Session) -> int:
    cutoff = utcnow() - dt.timedelta(seconds=WORKER_ONLINE_SECONDS)
    return int(db.execute(
        select(func.count()).select_from(models.ResearchWorker)
        .where(models.ResearchWorker.last_seen_at >= cutoff)
    ).scalar() or 0)


def view(db: Session, job: models.ResearchJob, include_result: bool = True) -> dict:
    """The job as the API reports it."""
    out = {
        "job_id": job.id,
        "kind": job.kind,
        "status": job.status,
        "cancel_requested": bool(job.cancel_requested),
        "attempts": job.attempts,
        "created_at": _iso(job.created_at),
        "started_at": _iso(job.started_at),
        "finished_at": _iso(job.finished_at),
        "poll_url": f"/api/research/jobs/{job.id}",
    }
    if job.status in ACTIVE:
        out["workers_online"] = workers_online(db)
    if job.status == QUEUED:
        out["queue_position"] = 1 + int(db.execute(
            select(func.count()).select_from(models.ResearchJob).where(
                models.ResearchJob.status == QUEUED,
                models.ResearchJob.created_at < job.created_at)
        ).scalar() or 0)
    if job.status == FAILED:
        out["error"] = {"status_code": job.error_status or 500,
                        "detail": job.error_detail or "Research job failed"}
    if job.status == SUCCEEDED and include_result:
        out["result"] = job.result_json
    return out


def queue_stats(db: Session) -> dict:
    counts = dict(db.execute(
        select(models.ResearchJob.status, func.count()).where(models.ResearchJob.status.in_(ACTIVE))
        .group_by(models.ResearchJob.status)
    ).all())
    oldest = _aware(db.execute(
        select(func.min(models.ResearchJob.created_at)).where(models.ResearchJob.status == QUEUED)
    ).scalar())
    return {
        "queued": int(counts.get(QUEUED, 0)),
        "running": int(counts.get(RUNNING, 0)),
        "oldest_queued_seconds": round((utcnow() - oldest).total_seconds(), 1) if oldest else None,
        "workers_online": workers_online(db),
    }


# --------------------------------------------------------------------------
# Worker side
# --------------------------------------------------------------------------

def claim_next(db: Session, worker_id: str, lease_seconds: float) -> models.ResearchJob | None:
    """Atomically take the oldest queued job, or return None."""
    for _ in range(5):
        query = (select(models.ResearchJob.id).where(models.ResearchJob.status == QUEUED)
                 .order_by(models.ResearchJob.created_at).limit(1))
        if _is_postgres:
            query = query.with_for_update(skip_locked=True)
        job_id = db.execute(query).scalar()
        if job_id is None:
            db.rollback()
            return None
        now = utcnow()
        claimed = db.execute(
            update(models.ResearchJob).execution_options(**_UNSYNCED)
            .where(models.ResearchJob.id == job_id, models.ResearchJob.status == QUEUED)
            .values(status=RUNNING, worker_id=worker_id, attempts=models.ResearchJob.attempts + 1,
                    started_at=now, lease_expires_at=now + dt.timedelta(seconds=lease_seconds))
        ).rowcount
        db.commit()
        if claimed:
            job = db.get(models.ResearchJob, job_id)
            db.refresh(job)
            return job
        # Another worker claimed it between the read and the update: retry.
    return None


def _owned(job: models.ResearchJob, worker_id: str, attempt: int):
    return (models.ResearchJob.id == job.id, models.ResearchJob.worker_id == worker_id,
            models.ResearchJob.attempts == attempt, models.ResearchJob.status == RUNNING)


def renew_lease(db: Session, job: models.ResearchJob, worker_id: str, attempt: int,
                lease_seconds: float) -> str:
    """Extend the lease. Returns "ok", "cancel" (cancellation requested) or
    "lost" (the job is no longer this worker's)."""
    renewed = db.execute(
        update(models.ResearchJob).execution_options(**_UNSYNCED).where(*_owned(job, worker_id, attempt))
        .values(lease_expires_at=utcnow() + dt.timedelta(seconds=lease_seconds))
    ).rowcount
    db.commit()
    if not renewed:
        return "lost"
    return "cancel" if cancel_requested(db, job.id) else "ok"


def cancel_requested(db: Session, job_id: str) -> bool:
    """Whether the job's owner asked to stop it. A read only, cheap enough
    for the worker to check every tick between lease renewals."""
    requested = db.execute(select(models.ResearchJob.cancel_requested)
                           .where(models.ResearchJob.id == job_id)).scalar()
    db.commit()  # end the read transaction; never sit idle in one
    return bool(requested)


def _finish(db: Session, job: models.ResearchJob, worker_id: str, attempt: int, **values) -> bool:
    return bool(db.execute(
        update(models.ResearchJob).execution_options(**_UNSYNCED).where(*_owned(job, worker_id, attempt))
        .values(finished_at=utcnow(), lease_expires_at=None, **values)
    ).rowcount)


def complete(db: Session, job: models.ResearchJob, worker_id: str, attempt: int,
             result: dict, audit: dict, trials: dict | None) -> bool:
    """Store a successful result with its side effects. False if the job is
    no longer this worker's (nothing is stored then, apart from trials: the
    configurations were evaluated whether or not the result is kept)."""
    kind = KINDS[job.kind]
    result = dict(result)
    if trials:
        result["trial_family"] = trials["family"]
        result["trials_in_family"] = research_trials.record(
            db, job.user_id, trials["family"], trials["configs"], trials["source"])
    resource_type, resource_id = "research", None
    try:
        if job.kind == "ma_backtest":
            # The dashboard backtest keeps its history row, as before.
            result = json_safe(result)
            record = models.Backtest(
                user_id=job.user_id, initial_capital=result["initial_capital"],
                final_capital=result["final_capital"], sharpe_ratio=result["sharpe_ratio"],
                max_drawdown=result["max_drawdown"], win_rate=result["win_rate"],
                total_trades=result["total_trades"], result_json=result)
            db.add(record)
            db.flush()
            result = {"id": record.id, **result}
            resource_type, resource_id = "backtest", record.id
        if not _finish(db, job, worker_id, attempt, status=SUCCEEDED, result_json=json_safe(result)):
            db.rollback()
            return False
        db.commit()
    except Exception:
        db.rollback()
        raise
    security_service.write_audit_log(db, job.user_id, kind.audit_action, resource_type, resource_id,
                                     {**json_safe(audit), "job_id": job.id})
    return True


def fail(db: Session, job: models.ResearchJob, worker_id: str, attempt: int,
         status_code: int, detail: str) -> bool:
    done = _finish(db, job, worker_id, attempt, status=FAILED,
                   error_status=int(status_code), error_detail=str(detail)[:2000])
    db.commit()
    return done


def mark_cancelled(db: Session, job: models.ResearchJob, worker_id: str, attempt: int) -> bool:
    done = _finish(db, job, worker_id, attempt, status=CANCELLED)
    db.commit()
    return done


def release(db: Session, job: models.ResearchJob, worker_id: str, attempt: int) -> bool:
    """Hand an interrupted job back to the queue (worker shutting down);
    the interrupted run does not count as an attempt."""
    done = bool(db.execute(
        update(models.ResearchJob).execution_options(**_UNSYNCED).where(*_owned(job, worker_id, attempt))
        .values(status=QUEUED, worker_id=None, lease_expires_at=None,
                attempts=models.ResearchJob.attempts - 1)
    ).rowcount)
    db.commit()
    return done


WORKER_LOST_DETAIL = "The research worker stopped responding while running this job."


def reap_expired(db: Session) -> int:
    """Requeue (or fail, after max_attempts) running jobs whose lease lapsed."""
    now = utcnow()
    expired = db.execute(
        select(models.ResearchJob).where(models.ResearchJob.status == RUNNING,
                                         models.ResearchJob.lease_expires_at < now)
    ).scalars().all()
    reaped = 0
    for job in expired:
        stale = (models.ResearchJob.id == job.id, models.ResearchJob.status == RUNNING,
                 models.ResearchJob.worker_id == job.worker_id,
                 models.ResearchJob.attempts == job.attempts,
                 models.ResearchJob.lease_expires_at < now)
        if job.cancel_requested:
            values = dict(status=CANCELLED, finished_at=now, lease_expires_at=None)
        elif job.attempts < job.max_attempts:
            values = dict(status=QUEUED, worker_id=None, lease_expires_at=None)
        else:
            values = dict(status=FAILED, finished_at=now, lease_expires_at=None,
                          error_status=500, error_detail=WORKER_LOST_DETAIL)
        reaped += db.execute(update(models.ResearchJob).execution_options(**_UNSYNCED).where(*stale).values(**values)).rowcount
    db.commit()
    return reaped


def heartbeat_worker(db: Session, worker_id: str, hostname: str, pid: int,
                     current_job_id: str | None) -> None:
    now = utcnow()
    touched = db.execute(
        update(models.ResearchWorker).execution_options(**_UNSYNCED).where(models.ResearchWorker.id == worker_id)
        .values(last_seen_at=now, current_job_id=current_job_id)
    ).rowcount
    if not touched:
        db.add(models.ResearchWorker(id=worker_id, hostname=hostname[:255], pid=pid,
                                     started_at=now, last_seen_at=now, current_job_id=current_job_id))
    db.commit()


def remove_worker(db: Session, worker_id: str) -> None:
    db.execute(delete(models.ResearchWorker).execution_options(**_UNSYNCED).where(models.ResearchWorker.id == worker_id))
    db.commit()


def purge_finished(db: Session, retention_days: float = RESEARCH_JOB_RETENTION_DAYS) -> int:
    """Delete finished jobs past retention and workers gone for a day."""
    now = utcnow()
    purged = db.execute(
        delete(models.ResearchJob).execution_options(**_UNSYNCED).where(
            models.ResearchJob.status.in_(TERMINAL),
            models.ResearchJob.finished_at < now - dt.timedelta(days=retention_days))
    ).rowcount
    db.execute(delete(models.ResearchWorker).execution_options(**_UNSYNCED).where(
        models.ResearchWorker.last_seen_at < now - dt.timedelta(days=1)))
    db.commit()
    return purged
