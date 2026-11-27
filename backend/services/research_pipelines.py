"""QuantumSentinel — research pipelines: bounded runs of research jobs.

A pipeline is a budget. Its owner names how many jobs it may queue
(``max_jobs``); each job queued in it (research_jobs.enqueue with
``pipeline_id``) spends one, whatever then becomes of the job, and a spent
job is never handed back: a failed, cancelled or purged job still counts.
On top of every pipeline's own budget, one user may queue at most
RESEARCH_PIPELINE_DAILY_JOBS pipeline jobs per UTC day.

Searching more configurations makes a false discovery more likely, so the
breadth of an automated search has to be bounded before it runs, not
explained after (the Deflated Sharpe discount and the MinBTL bound grow
with the number of trials). A closed pipeline queues nothing more.

Charging runs inside the enqueue transaction, after the user's row lock:
the pipeline row and the day row are locked in that order, so concurrent
enqueues of one user are serialised and the counters can never pass their
budgets (migration 0018's CHECKs refuse it besides).
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from .. import models

OPEN, CLOSED = "open", "closed"
MAX_JOBS_LIMIT = 10_000
NAME_MAX = 120

_UNSYNCED = {"synchronize_session": False}


class JobRefused(Exception):
    """A job (or pipeline change) the queue refuses, with the HTTP status
    and the user-safe reason the API answers with."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code, self.detail = status_code, detail


class PipelineNotFound(JobRefused):
    def __init__(self):
        super().__init__(404, "Research pipeline not found")


class PipelineClosed(JobRefused):
    def __init__(self):
        super().__init__(409, "The research pipeline is closed: it queues no more jobs")


class PipelineBudgetSpent(JobRefused):
    def __init__(self, max_jobs: int):
        super().__init__(409, f"The research pipeline has queued all {max_jobs} jobs its budget allows")


class DailyBudgetSpent(JobRefused):
    def __init__(self, limit: int):
        super().__init__(429, f"You have queued {limit} pipeline jobs today, the daily limit "
                              f"(RESEARCH_PIPELINE_DAILY_JOBS); try again after 00:00 UTC"
                         if limit > 0 else "Pipeline jobs are disabled (RESEARCH_PIPELINE_DAILY_JOBS is 0)")


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _aware(value: dt.datetime | None) -> dt.datetime | None:
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=dt.timezone.utc)
    return value


def create(db: Session, user_id: str, name: str, max_jobs: int) -> models.ResearchPipeline:
    name = (name or "").strip()
    if not 1 <= len(name) <= NAME_MAX:
        raise JobRefused(422, f"A pipeline name has 1 to {NAME_MAX} characters")
    if isinstance(max_jobs, bool) or not isinstance(max_jobs, int) or not 1 <= max_jobs <= MAX_JOBS_LIMIT:
        raise JobRefused(422, f"max_jobs is a whole number from 1 to {MAX_JOBS_LIMIT}")
    pipeline = models.ResearchPipeline(user_id=user_id, name=name, max_jobs=max_jobs, jobs_queued=0,
                                       status=OPEN, created_at=utcnow())
    db.add(pipeline)
    db.commit()
    db.refresh(pipeline)
    return pipeline


def get_for_user(db: Session, user_id: str, pipeline_id: str) -> models.ResearchPipeline | None:
    pipeline = db.get(models.ResearchPipeline, pipeline_id)
    return pipeline if pipeline is not None and pipeline.user_id == user_id else None


def list_for_user(db: Session, user_id: str, limit: int = 50) -> list[models.ResearchPipeline]:
    return list(db.execute(
        select(models.ResearchPipeline).where(models.ResearchPipeline.user_id == user_id)
        .order_by(models.ResearchPipeline.created_at.desc()).limit(limit)
    ).scalars())


def close(db: Session, user_id: str, pipeline_id: str) -> models.ResearchPipeline | None:
    """Close the pipeline (it queues nothing more); closing a closed one is a
    no-op. Jobs already queued in it still run."""
    pipeline = get_for_user(db, user_id, pipeline_id)
    if pipeline is None:
        return None
    db.execute(update(models.ResearchPipeline).execution_options(**_UNSYNCED)
               .where(models.ResearchPipeline.id == pipeline_id, models.ResearchPipeline.status == OPEN)
               .values(status=CLOSED, closed_at=utcnow()))
    db.commit()
    db.refresh(pipeline)
    return pipeline


def charge(db: Session, user_id: str, pipeline_id: str, daily_limit: int, today: dt.date) -> None:
    """Spend one job of the pipeline's budget and of the user's daily budget,
    or raise JobRefused. Runs in the caller's transaction, which must hold
    the user's row lock; nothing is committed here."""
    # populate_existing: the locked row's counters, not ones this session
    # may have loaded before it waited for the lock.
    pipeline = db.execute(select(models.ResearchPipeline).where(models.ResearchPipeline.id == pipeline_id)
                          .with_for_update().execution_options(populate_existing=True)).scalar_one_or_none()
    if pipeline is None or pipeline.user_id != user_id:
        raise PipelineNotFound()
    if pipeline.status != OPEN:
        raise PipelineClosed()
    if pipeline.jobs_queued >= pipeline.max_jobs:
        raise PipelineBudgetSpent(pipeline.max_jobs)
    # A daily limit of 0 (or less) refuses below too: nothing spent yet is
    # still not under it.
    db.execute(pg_insert(models.ResearchPipelineDay)
               .values(user_id=user_id, day=today, jobs_queued=0)
               .on_conflict_do_nothing(index_elements=["user_id", "day"]))
    spent_today = db.execute(
        select(models.ResearchPipelineDay.jobs_queued)
        .where(models.ResearchPipelineDay.user_id == user_id, models.ResearchPipelineDay.day == today)
        .with_for_update()).scalar_one()
    if spent_today >= daily_limit:
        raise DailyBudgetSpent(daily_limit)
    db.execute(update(models.ResearchPipeline).execution_options(**_UNSYNCED)
               .where(models.ResearchPipeline.id == pipeline_id)
               .values(jobs_queued=models.ResearchPipeline.jobs_queued + 1))
    db.execute(update(models.ResearchPipelineDay).execution_options(**_UNSYNCED)
               .where(models.ResearchPipelineDay.user_id == user_id, models.ResearchPipelineDay.day == today)
               .values(jobs_queued=models.ResearchPipelineDay.jobs_queued + 1))


def view(db: Session, pipeline: models.ResearchPipeline, include_jobs: bool = False) -> dict:
    out = {
        "pipeline_id": pipeline.id,
        "name": pipeline.name,
        "status": pipeline.status,
        "max_jobs": pipeline.max_jobs,
        "jobs_queued": pipeline.jobs_queued,
        "jobs_remaining": pipeline.max_jobs - pipeline.jobs_queued if pipeline.status == OPEN else 0,
        "created_at": _aware(pipeline.created_at).isoformat(),
        "closed_at": _aware(pipeline.closed_at).isoformat() if pipeline.closed_at else None,
    }
    if include_jobs:
        # Jobs still kept: a finished job is purged after the retention
        # period, while jobs_queued keeps counting it.
        jobs = db.execute(select(models.ResearchJob)
                          .where(models.ResearchJob.pipeline_id == pipeline.id,
                                 models.ResearchJob.user_id == pipeline.user_id)
                          .order_by(models.ResearchJob.created_at)).scalars()
        out["jobs"] = [{"job_id": job.id, "kind": job.kind, "stage": job.stage, "status": job.status,
                        "poll_url": f"/api/research/jobs/{job.id}"} for job in jobs]
    return out
