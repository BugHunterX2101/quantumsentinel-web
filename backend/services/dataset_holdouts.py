"""QuantumSentinel — sealed holdouts of dataset families.

A family's holdout start (``dataset_holdouts``, migration 0005) splits its
data in two. Research tasks see only the bars and corporate actions dated
before it; the data from it on is held back for a final evaluation, so that a
strategy chosen after many research runs can be judged on data none of those
runs saw. The research worker reads the starts when it runs a job, hands them
to the job process, and market_data applies them to every dataset the task
reads; the worker then checks that the result names the starts it was given
(dataset_registry.record_market_data). A family with no row has no holdout.

An operator sets a start with ``python -m backend.manage set-holdout``, and it
can only move later (the migration's trigger enforces this for every writer).
Moving it later releases the old holdout to research, which is how a holdout
is rolled forward once it has been used.

Not yet: the data from the start on is still inside the dataset blobs the
worker reads, and is kept from tasks by this code path, not by storage
permissions. A final-evaluation job that alone reads it, under a role that
alone can, comes with immutable strategy versions (plan R1-5b, after R2).
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy import func, select
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

from .. import models
from .dataset_providers import FAMILIES

AUDIT_ACTION = "DATASET_HOLDOUT_SET"


class HoldoutError(ValueError):
    """A holdout start the registry refuses; the message is safe to show."""


def current(db: Session) -> dict[str, dt.date]:
    """Each family's holdout start: its latest row."""
    return dict(db.execute(select(models.DatasetHoldout.family, func.max(models.DatasetHoldout.holdout_start))
                           .group_by(models.DatasetHoldout.family)).all())


def set_start(db: Session, family: str, start: dt.date, *, set_by: str, reason: str) -> models.DatasetHoldout:
    """Record that ``family``'s holdout starts on ``start``, and audit it.

    Refused unless ``family`` is one a provider produces and ``start`` is
    later than the family's current start. The trigger re-checks under a lock,
    so a concurrent writer cannot slip an earlier start in."""
    from . import security_service

    if family not in FAMILIES:
        raise HoldoutError(f"unknown dataset family {family!r}; families: {', '.join(sorted(FAMILIES))}")
    if not isinstance(start, dt.date) or isinstance(start, dt.datetime):
        raise HoldoutError("the holdout start must be a date")
    set_by, reason = (set_by or "").strip(), (reason or "").strip()
    if not set_by or not reason:
        raise HoldoutError("say who sets the holdout and why")
    latest = current(db).get(family)
    if latest is not None and start <= latest:
        db.rollback()
        raise HoldoutError(f"the holdout of {family} starts on {latest.isoformat()}; it can only move later")
    row = models.DatasetHoldout(family=family, holdout_start=start, set_by=set_by[:320], reason=reason,
                                created_at=dt.datetime.now(dt.timezone.utc))
    db.add(row)
    try:
        db.flush()
    except DBAPIError as exc:
        db.rollback()
        if getattr(exc.orig, "sqlstate", None) == "23001":  # restrict_violation: the trigger refused it
            raise HoldoutError(f"the holdout of {family} can only move later") from exc
        raise
    db.commit()
    security_service.write_audit_log(db, None, AUDIT_ACTION, "dataset_family", family, {
        "holdout_start": start.isoformat(), "previous_start": latest.isoformat() if latest else None,
        "set_by": row.set_by, "reason": reason})
    return row
