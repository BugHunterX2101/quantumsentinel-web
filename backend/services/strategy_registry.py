"""QuantumSentinel — the registry of strategy versions.

A strategy version is a strategy spec (services/strategy_spec.py) stored as
its canonical JSON under its content hash. Rows never change (migration
0006): a changed strategy is a new version. Registering a spec that already
exists, by another user or again, keeps the first row and only records that
this user holds it, so one user never learns from the registry who else
wrote the same spec.

A tag is a user's own name for a version they hold, such as ``1.2.0``. It is
an alias for reading: it belongs to that user, always names the version it
was first given to, and means nothing to any gate.

The research worker runs a strategy backtest job with two functions here:
``backtest_inputs`` loads the version's spec before the job process starts,
and ``finalize_backtest_job`` accepts the result and records its evaluation.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .. import models
from . import strategy_spec
from .strategy_spec import STRATEGY_VERSION_ID, StrategySpec

LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
log = logging.getLogger(__name__)


class StrategyRegistryError(ValueError):
    """A registry request that cannot be honoured."""


class UnknownStrategyVersion(StrategyRegistryError):
    """No version by that id is held by the user."""


class TagConflict(StrategyRegistryError):
    """The label already names another of the user's versions."""


@dataclass(frozen=True)
class Held:
    """A version as one user holds it: when *they* registered it (never the
    version's own ``created_at``, the first registrant's time, which would
    tell them someone else wrote the same spec earlier) and their tags."""
    version: models.StrategyVersion
    held_since: dt.datetime
    tags: list[str]


def register(db: Session, value: Any, *, user_id: str) -> models.StrategyVersion:
    """Validate a spec (parsed JSON), record it and that ``user_id`` holds it,
    in the caller's transaction. Raises pydantic's ValidationError for a spec
    that is not DSL v1."""
    spec = strategy_spec.parse(value)
    canonical = strategy_spec.canonical(spec)
    version_id = strategy_spec.version_id(spec)
    now = dt.datetime.now(dt.timezone.utc)
    db.execute(insert(models.StrategyVersion).values(
        version_id=version_id, dsl=spec.dsl, template=spec.signal.template, family=spec.universe.family,
        spec_canonical=canonical, created_at=now).on_conflict_do_nothing(index_elements=["version_id"]))
    version = db.execute(select(models.StrategyVersion).where(
        models.StrategyVersion.version_id == version_id)).scalar_one()
    # The database checks the stored bytes hash to the id, so equal ids hold
    # equal bytes; checked here too, as the registry promises the caller.
    if bytes(version.spec_canonical) != canonical:
        raise StrategyRegistryError(f"{version_id}: different bytes are already registered")
    db.execute(insert(models.UserStrategyVersion).values(user_id=user_id, version_id=version_id, created_at=now)
               .on_conflict_do_nothing(index_elements=["user_id", "version_id"]))
    return version


def spec_of(version: models.StrategyVersion) -> StrategySpec:
    """The spec a stored version holds, validated again by today's DSL code.

    A version written by an earlier release that today's validators refuse,
    or that no longer encodes to its own id, is refused rather than run as
    something other than what was registered."""
    raw = bytes(version.spec_canonical)
    try:
        spec = strategy_spec.parse(json.loads(raw))
    except ValueError as exc:  # pydantic's ValidationError and JSONDecodeError are ValueErrors
        raise StrategyRegistryError(f"{version.version_id}: the stored spec no longer validates: {exc}") from exc
    if strategy_spec.canonical(spec) != raw or strategy_spec.version_id(spec) != version.version_id:
        raise StrategyRegistryError(f"{version.version_id}: the stored spec no longer encodes to its id")
    return spec


def get_for_user(db: Session, user_id: str, version_id: str) -> models.StrategyVersion | None:
    """The version, if ``user_id`` holds it."""
    return db.execute(
        select(models.StrategyVersion)
        .join(models.UserStrategyVersion, models.UserStrategyVersion.version_id == models.StrategyVersion.version_id)
        .where(models.UserStrategyVersion.user_id == user_id,
               models.StrategyVersion.version_id == version_id)).scalar_one_or_none()


def _held(db: Session, user_id: str, *criteria, limit: int | None = None) -> list[Held]:
    rows = db.execute(
        select(models.StrategyVersion, models.UserStrategyVersion.created_at)
        .join(models.UserStrategyVersion, models.UserStrategyVersion.version_id == models.StrategyVersion.version_id)
        .where(models.UserStrategyVersion.user_id == user_id, *criteria)
        .order_by(models.UserStrategyVersion.created_at.desc(), models.StrategyVersion.version_id)
        .limit(limit)).all()
    tags: dict[str, list[str]] = {}
    if rows:
        for row in db.execute(select(models.UserStrategyTag).where(
                models.UserStrategyTag.user_id == user_id,
                models.UserStrategyTag.version_id.in_([version.version_id for version, _ in rows]))
                .order_by(models.UserStrategyTag.created_at, models.UserStrategyTag.label)).scalars():
            tags.setdefault(row.version_id, []).append(row.label)
    return [Held(version, held_since, tags.get(version.version_id, [])) for version, held_since in rows]


def list_for_user(db: Session, user_id: str, limit: int = 50) -> list[Held]:
    """The versions ``user_id`` holds, last registered first, each with the
    user's tags for it in the order given."""
    return _held(db, user_id, limit=limit)


def held(db: Session, user_id: str, version_id: str) -> Held | None:
    """One version as ``user_id`` holds it, if they do."""
    found = _held(db, user_id, models.StrategyVersion.version_id == version_id)
    return found[0] if found else None


def view(held_version: Held) -> dict:
    """A held version as the API reports it."""
    version = held_version.version
    return {
        "version_id": version.version_id,
        "dsl": version.dsl,
        "template": version.template,
        "family": version.family,
        "spec": json.loads(bytes(version.spec_canonical)),
        "tags": list(held_version.tags),
        "held_since": held_version.held_since.astimezone(dt.timezone.utc).isoformat(),
    }


def tag(db: Session, user_id: str, version_id: str, label: str) -> models.UserStrategyTag:
    """Give ``label`` to a version ``user_id`` holds, in the caller's
    transaction. Giving a label again to the version it names is a no-op; a
    label that names another version is refused, as tags never move."""
    if not isinstance(label, str) or not LABEL.fullmatch(label):
        raise StrategyRegistryError("a tag is 1-64 letters, digits, '.', '_' or '-', starting with a letter or digit")
    if get_for_user(db, user_id, version_id) is None:
        raise UnknownStrategyVersion("no such strategy version")
    existing = db.get(models.UserStrategyTag, (user_id, label))
    if existing is None:
        row = models.UserStrategyTag(user_id=user_id, label=label, version_id=version_id,
                                     created_at=dt.datetime.now(dt.timezone.utc))
        try:
            with db.begin_nested():
                db.add(row)
            return row
        except IntegrityError:
            # A concurrent request gave the label first.
            existing = db.get(models.UserStrategyTag, (user_id, label))
    if existing is None or existing.version_id != version_id:
        raise TagConflict(f"the tag {label!r} already names another version")
    return existing


def resolve(db: Session, user_id: str, ref: str) -> models.StrategyVersion | None:
    """A version ``user_id`` holds, named by its id or by one of their tags."""
    if STRATEGY_VERSION_ID.fullmatch(ref):
        return get_for_user(db, user_id, ref)
    found = db.get(models.UserStrategyTag, (user_id, ref)) if LABEL.fullmatch(ref) else None
    return get_for_user(db, user_id, found.version_id) if found else None


# --------------------------------------------------------------------------
# Strategy backtest jobs (research_jobs.KINDS["strategy_backtest"]), run by
# the research worker
# --------------------------------------------------------------------------

def _job_version(db: Session, job: models.ResearchJob) -> models.StrategyVersion:
    from .research_tasks import TaskError

    version_id = (job.params_json or {}).get("strategy_version_id")
    version = get_for_user(db, job.user_id, version_id) if isinstance(version_id, str) else None
    if version is None:
        raise TaskError(404, "Strategy version not found")
    return version


def backtest_inputs(db: Session, job: models.ResearchJob) -> dict:
    """What a strategy backtest's job process is given: the spec of the
    version the job names, which the job's user must hold, as today's DSL
    code reads it (``spec_of``). The job's params only name the version, so a
    task never runs a spec the API did not resolve for that user."""
    from .research_tasks import TaskError

    try:
        spec = spec_of(_job_version(db, job))
    except StrategyRegistryError as exc:
        raise TaskError(422, str(exc)) from exc
    return {"spec": json.loads(strategy_spec.canonical(spec))}


def finalize_backtest_job(db: Session, job: models.ResearchJob, result: dict, audit: dict):
    """Accept a strategy backtest's result, after the worker has accepted the
    market data it read (dataset_registry.record_market_data): every dataset
    that supplied its prices must be of the version's family (the spec's
    ``universe.family`` selects the data), and what it measured is recorded
    as an evaluation of its trial (research_trials.record_evaluation), in
    the transaction that stores the result."""
    from .. import schemas
    from . import research_trials
    from .research_tasks import TaskError, strategy_trial_config

    version = _job_version(db, job)
    datasets = list(result["market_data"]["datasets"])
    families = sorted(set(db.execute(select(models.DatasetVersion.family).where(
        models.DatasetVersion.dataset_id.in_(datasets))).scalars()))
    if families != [version.family]:
        raise TaskError(422, f"The strategy trades {version.family} data, but its prices came from "
                             f"{', '.join(families) or 'no'} datasets")
    try:
        spec = spec_of(version)
    except StrategyRegistryError as exc:
        raise TaskError(422, str(exc)) from exc
    family = research_trials.family_hash(spec.signal.template, spec.universe.symbols)
    if result.get("trial_family") != family:
        raise TaskError(500, "The research job recorded its trial in another family")
    try:
        research_trials.record_evaluation(
            db, job_id=job.id, user_id=job.user_id, family=family,
            config=strategy_trial_config(schemas.StrategyBacktestJob.model_validate(job.params_json)),
            strategy_version_id=version.version_id, metrics=result["evaluation"], dataset_ids=datasets)
    except (KeyError, TypeError, ValueError) as exc:
        log.error("strategy backtest job %s: its evaluation was refused: %s", job.id, exc)
        raise TaskError(500, "The research job reported an invalid evaluation") from exc
    return result, ("strategy_version", version.version_id), audit
