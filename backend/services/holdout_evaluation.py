"""QuantumSentinel — the one-shot holdout evaluation of a validated strategy (R1-5b).

A strategy validation (strategy_validation.py) judges a candidate on data
from before its dataset family's holdout start (dataset_holdouts). A
candidate whose verdict passed may then be graded once on the held-out data
from that start on. That grade is the only time any code reads the holdout.

An operator queues it with ``python -m backend.manage final-evaluate``. The
``final_evaluation`` job (research_jobs.KINDS) is the only evaluator kind, so
only the research evaluator runs it, as ``qs_evaluator``, under three rules:

* It spends the attempt first. ``inputs`` (the kind's prepare step, in the
  worker process) checks the job against the verdict, then writes a
  ``holdout_attempts`` row and commits it before the job process reads any
  data. A crash, a data error or a refused result still spends it. The job
  has one attempt (max_attempts 1), and the evaluator never hands it back
  when it shuts down. The table allows one attempt per user, version,
  dataset family and holdout start, so a version is graded again only after
  ``set-holdout`` moves the start later.
* It grades only a candidate chosen without the holdout. The verdict's
  signed manifest must name the current holdout start. The job process
  first re-runs the evaluation cut at that start, and it must reproduce the
  evaluation's return series exactly. Only then does it run the strategy on
  the uncut data, from the evaluation's own first session through the end
  of its datasets, and report the returns from the start on. Data before the
  start is warm-up only.
* It reveals only pass or fail. ``finalize_job`` computes the grade itself
  from the reported returns, under the holdout policy (``grade``). It stores
  the grade, its figures and its signed manifest in ``holdout_evaluations``,
  which the research worker cannot read. The job's result and audit event,
  which the research worker can read, carry ids and pass/fail only, and no
  error message from the holdout run carries a figure.

Each attempt counts as one more trial in the candidate's research family
(trial_matrix.family_state). So a strategy picked after a holdout grade is
deflated for that grade, but the holdout run is never part of a family's
trial matrix.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import math
from dataclasses import asdict, dataclass

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .. import models
from . import dataset_format

log = logging.getLogger(__name__)

KIND = "final_evaluation"
AUDIT_ACTION = "HOLDOUT_EVALUATION"
POLICY_SCHEME = "qs-hp-sha256-v1"
MANIFEST_VERSION = 3
MANIFEST_KIND = "qs-holdout-evaluation"
MANIFEST_SCHEME = "qs-hm-sha256-v1"
GRADE_SCHEME = "qs-grade-json-sha256-v1"

# The job-facing message for every failure of the holdout run itself: its
# real cause could carry a figure of the holdout, and the job's error is
# readable by the research worker. The evaluator's own log has the cause.
HOLDOUT_RUN_FAILED = "The holdout run could not be completed; the attempt is spent."
SHUTDOWN_DETAIL = ("The research evaluator stopped before the holdout evaluation finished; "
                   "the attempt is spent.")


class HoldoutEvaluationError(ValueError):
    """A holdout evaluation that cannot be queued; the message is safe to show."""


@dataclass(frozen=True)
class HoldoutPolicy:
    """What a holdout grade requires. Every value is an integer or a string,
    so the canonical JSON the id hashes is exact."""
    name: str
    # Fewer holdout sessions than this fails closed.
    min_sessions: int
    # The floor on the holdout's annualised Sharpe ratio, in thousandths.
    min_sharpe_annual_milli: int
    # Sessions a year by dataset family, as (family, sessions) pairs ordered
    # by family. A family not listed fails closed.
    sessions_per_year: tuple[tuple[str, int], ...]

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("a holdout policy needs a name")
        for field in ("min_sessions", "min_sharpe_annual_milli"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or not 2 <= value <= 100_000:
                raise ValueError(f"{field} must be an integer in 2..100000")
        pairs = self.sessions_per_year
        if (not isinstance(pairs, tuple) or not pairs
                or not all(isinstance(p, tuple) and len(p) == 2 and isinstance(p[0], str) and p[0]
                           and isinstance(p[1], int) and not isinstance(p[1], bool) and 1 <= p[1] <= 366
                           for p in pairs)
                or [p[0] for p in pairs] != sorted({p[0] for p in pairs})):
            raise ValueError("sessions_per_year is (family, sessions in 1..366) pairs, one per family, in order")


def _v1() -> HoldoutPolicy:
    from .validation_policy import POLICY_V1
    # A quarter of exchange sessions, and the haircut Sharpe floor of the
    # validation policy, on the same calendars.
    return HoldoutPolicy(name="qs-holdout-policy-v1", min_sessions=63, min_sharpe_annual_milli=500,
                         sessions_per_year=POLICY_V1.sessions_per_year)


HOLDOUT_POLICY_V1 = _v1()
CURRENT_POLICY = HOLDOUT_POLICY_V1


def policy_canonical(policy: HoldoutPolicy) -> bytes:
    return dataset_format.canonical_json(json.loads(json.dumps(asdict(policy))))


def policy_id(policy: HoldoutPolicy) -> str:
    return f"{POLICY_SCHEME}:{hashlib.sha256(policy_canonical(policy)).hexdigest()}"


# --------------------------------------------------------------------------
# The grade
# --------------------------------------------------------------------------

def grade(dates: list[str], quanta: list[int], holdout_start: dt.date, dataset_family: str,
          policy: HoldoutPolicy = CURRENT_POLICY) -> dict:
    """Grade the holdout returns ``quanta`` (research_trials.quantise, one
    per session in ``dates``, every one on or after ``holdout_start``).

    It passes if and only if there are at least ``min_sessions`` sessions
    and the annualised Sharpe ratio (validation_stats.sharpe, ddof 1, times
    the square root of the family's sessions a year) is at least the floor.
    Anything on which the grade is not defined fails closed with a reason."""
    from . import validation_stats as vs

    out = {"policy": policy_id(policy), "dataset_family": dataset_family,
           "holdout_start": holdout_start.isoformat(), "sessions": len(dates),
           "first_session": dates[0] if dates else None, "last_session": dates[-1] if dates else None,
           "min_sessions": policy.min_sessions, "min_sharpe_annual": policy.min_sharpe_annual_milli / 1000,
           "sharpe_annual": None, "passed": False, "reason": None}
    if len(dates) < policy.min_sessions:
        out["reason"] = f"{len(dates)} holdout sessions; the policy needs at least {policy.min_sessions}"
        return out
    per_year = dict(policy.sessions_per_year).get(dataset_family)
    if per_year is None:
        out["reason"] = f"the policy gives no sessions a year for the {dataset_family!r} dataset family"
        return out
    try:
        sharpe = vs.sharpe(quanta)
    except vs.StatsError as exc:
        out["reason"] = f"no Sharpe ratio: {exc}"
        return out
    annual = sharpe * math.sqrt(per_year)
    if not math.isfinite(annual):
        out["reason"] = "the Sharpe ratio is not finite"
        return out
    out["sharpe_annual"] = annual
    out["passed"] = annual * 1000 >= policy.min_sharpe_annual_milli
    if not out["passed"]:
        out["reason"] = "the annualised Sharpe ratio is below the policy's floor"
    return out


# --------------------------------------------------------------------------
# Queueing (manage final-evaluate)
# --------------------------------------------------------------------------

def _candidate(db: Session, user_id: str, validation_job_id: str, period: str, initial_capital: float):
    """The passing verdict, its evaluation, the version's spec, its dataset
    family and that family's holdout start, after every check that needs no
    holdout data. Raises HoldoutEvaluationError with a message safe to show."""
    from . import dataset_holdouts, strategy_registry, strategy_validation, trial_matrix, validation_manifest

    validation = strategy_validation.get_for_user(db, user_id, validation_job_id)
    if validation is None:
        raise HoldoutEvaluationError("Validation not found")
    if not validation.passed:
        raise HoldoutEvaluationError("The validation did not pass; only a passing strategy is graded on its holdout")
    evaluation = trial_matrix.evaluation_for_user(db, user_id, validation.evaluation_job_id)
    if evaluation is None:
        raise HoldoutEvaluationError("The validation's evaluation is not recorded")
    if not trial_matrix.ran_with(evaluation, period, initial_capital):
        raise HoldoutEvaluationError("The period and initial capital are not the ones the evaluation ran with")
    version = strategy_registry.get_for_user(db, user_id, validation.strategy_version_id)
    if version is None:
        raise HoldoutEvaluationError(f"The strategy version {validation.strategy_version_id} is not held")
    try:
        spec = strategy_registry.spec_of(version)
    except strategy_registry.StrategyRegistryError as exc:
        raise HoldoutEvaluationError(str(exc)) from exc
    family = spec.universe.family
    start = dataset_holdouts.current(db).get(family)
    if start is None:
        raise HoldoutEvaluationError(f"The {family} dataset family has no holdout to grade on")
    manifest_row = validation_manifest.get(db, validation.job_id)
    if manifest_row is None:
        raise HoldoutEvaluationError("The validation has no signed manifest, so the holdout start it was reached "
                                     "under is unknown; validate the evaluation again")
    manifest = json.loads(bytes(manifest_row.canonical))
    if (not validation_manifest.signature_valid(db, manifest_row)
            or not validation_manifest.matches(manifest_row, validation, manifest)):
        raise HoldoutEvaluationError("The validation's manifest does not verify")
    if manifest.get("dataset_holdout_start") != start.isoformat():
        raise HoldoutEvaluationError(f"The validation was reached under another holdout start than the current "
                                     f"one of {family} ({start.isoformat()}); validate the evaluation again")
    return validation, evaluation, spec, family, start


def spent(db: Session, user_id: str, strategy_version_id: str, dataset_family: str, start: dt.date) -> bool:
    return db.execute(select(func.count()).select_from(models.HoldoutAttempt).where(
        models.HoldoutAttempt.user_id == user_id,
        models.HoldoutAttempt.strategy_version_id == strategy_version_id,
        models.HoldoutAttempt.dataset_family == dataset_family,
        models.HoldoutAttempt.holdout_start == start)).scalar_one() > 0


def queue(db: Session, validation_job_id: str, period: str, initial_capital: float) -> models.ResearchJob:
    """Queue the holdout evaluation of a passing verdict, for its owner.
    Every check the evaluator repeats before it spends the attempt is made
    here too, so that a job that would be refused is not queued."""
    from .. import schemas
    from . import research_jobs

    validation = db.get(models.StrategyValidation, validation_job_id) if isinstance(validation_job_id, str) else None
    if validation is None:
        raise HoldoutEvaluationError("Validation not found")
    try:
        params = schemas.FinalEvaluationJob(validation=validation_job_id, period=period,
                                            initial_capital=initial_capital).model_dump()
    except ValueError as exc:
        raise HoldoutEvaluationError(str(exc)) from exc
    _, _, spec, family, start = _candidate(db, validation.user_id, validation_job_id, params["period"],
                                           params["initial_capital"])
    if spent(db, validation.user_id, validation.strategy_version_id, family, start):
        raise HoldoutEvaluationError(f"{validation.strategy_version_id} has already been graded on the holdout of "
                                     f"{family} from {start.isoformat()}")
    db.rollback()
    return research_jobs.enqueue(db, validation.user_id, KIND, params, same_as=("validation",))


# --------------------------------------------------------------------------
# The final_evaluation job (research_jobs.KINDS[KIND]), run by the evaluator
# --------------------------------------------------------------------------

def inputs(db: Session, job: models.ResearchJob) -> dict:
    """The kind's prepare step, in the evaluator's worker process: check the
    job, then spend the attempt (committed) and return what the job process
    is given. Every read comes first; the attempt is written last."""
    from .. import schemas
    from . import strategy_spec
    from .research_tasks import TaskError

    req = schemas.FinalEvaluationJob.model_validate(job.params_json)
    try:
        validation, evaluation, spec, family, start = _candidate(db, job.user_id, req.validation, req.period,
                                                                 req.initial_capital)
    except HoldoutEvaluationError as exc:
        raise TaskError(422, str(exc)) from exc
    db.add(models.HoldoutAttempt(job_id=job.id, user_id=job.user_id, validation_job_id=validation.job_id,
                                 strategy_version_id=validation.strategy_version_id,
                                 family_hash=evaluation.family_hash, dataset_family=family, holdout_start=start,
                                 created_at=dt.datetime.now(dt.timezone.utc)))
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise TaskError(409, f"{validation.strategy_version_id} has already been graded on the holdout of "
                             f"{family} from {start.isoformat()}") from exc
    log.info("holdout evaluation %s: attempt spent (%s on %s from %s)", job.id, validation.strategy_version_id,
             family, start.isoformat())
    return {"strategy_version_id": validation.strategy_version_id,
            "spec": json.loads(strategy_spec.canonical(spec)),
            "evaluation": {"job_id": evaluation.job_id, "returns_sha256": evaluation.returns_sha256,
                           "dataset_ids": sorted(evaluation.dataset_ids)},
            "period": req.period, "initial_capital": req.initial_capital,
            "dataset_family": family, "holdout_start": start.isoformat()}


def finalize_job(db: Session, job: models.ResearchJob, result: dict, audit: dict):
    """Grade the holdout returns the job process reported, and record the
    grade with its signed manifest. Returns a result and an audit event with
    ids and pass/fail only."""
    from . import research_trials
    from .research_tasks import TaskError

    attempt = db.get(models.HoldoutAttempt, job.id)
    validation = db.get(models.StrategyValidation, attempt.validation_job_id) if attempt is not None else None
    evaluation = db.get(models.ResearchTrialEvaluation, validation.evaluation_job_id) if validation else None
    if evaluation is None:
        log.error("holdout evaluation %s: no attempt, verdict or evaluation recorded", job.id)
        raise TaskError(500, HOLDOUT_RUN_FAILED)
    report = result.get("holdout") if isinstance(result, dict) else None
    try:
        dates, quanta = list(report["dates"]), [int(q) for q in report["returns"]]
        ok = (report["holdout_start"] == attempt.holdout_start.isoformat()
              and report["reproduced"] == evaluation.returns_sha256
              and sorted(report["datasets"]) == sorted(evaluation.dataset_ids)
              and len(dates) == len(quanta) and dates == sorted(set(dates))
              and all(d >= attempt.holdout_start.isoformat() for d in dates)
              and all(not isinstance(q, bool) for q in report["returns"]))
    except (KeyError, TypeError, ValueError) as exc:
        log.error("holdout evaluation %s: malformed report: %s", job.id, exc)
        ok = False
    if not ok:
        log.error("holdout evaluation %s: the job process's report was refused", job.id)
        raise TaskError(500, HOLDOUT_RUN_FAILED)

    policy = CURRENT_POLICY
    grade_ = grade(dates, quanta, attempt.holdout_start, attempt.dataset_family, policy)
    grade_["returns_sha256"] = research_trials.quantised_returns_id(quanta) if quanta else None
    grade_["window_first_session"] = report.get("window_first_session")
    try:
        json.dumps(grade_, allow_nan=False)
    except ValueError as exc:
        log.error("holdout evaluation %s: grade not finite: %s", job.id, exc)
        raise TaskError(500, HOLDOUT_RUN_FAILED) from exc
    now = dt.datetime.now(dt.timezone.utc)
    try:
        record(db, attempt=attempt, evaluation=evaluation, policy=policy, grade_=grade_, created_at=now)
    except ManifestUnavailable as exc:
        log.error("holdout evaluation %s: its manifest could not be signed: %s", job.id, exc)
        raise TaskError(500, "The holdout evaluation's manifest could not be signed; the attempt is spent.") from exc
    log.info("holdout evaluation %s: %s", job.id, "passed" if grade_["passed"] else "failed")
    summary = {"holdout_evaluation": job.id, "validation": attempt.validation_job_id,
               "strategy_version_id": attempt.strategy_version_id, "dataset_family": attempt.dataset_family,
               "holdout_start": attempt.holdout_start.isoformat(), "policy": grade_["policy"],
               "passed": grade_["passed"]}
    return dict(summary), ("strategy_version", attempt.strategy_version_id), dict(summary)


# --------------------------------------------------------------------------
# The grade's signed manifest
# --------------------------------------------------------------------------

class ManifestUnavailable(Exception):
    """The grade cannot be signed as production requires."""


def grade_sha256(grade_: dict) -> str:
    from .validation_manifest import verdict_sha256
    return verdict_sha256(grade_)


def build_manifest(*, attempt: models.HoldoutAttempt, evaluation: models.ResearchTrialEvaluation,
                   validation_manifest_id: str | None, policy_id_: str, grade_: dict, created_at: dt.datetime,
                   signing_key_id: str | None) -> dict:
    """The manifest of one grade (every value a string, bool, int or None)."""
    from .event_simulator import ENGINE_VERSION
    from .validation_manifest import code_commit, dependency_lock_sha256

    return {
        "manifest_version": MANIFEST_VERSION,
        "kind": MANIFEST_KIND,
        "holdout_job_id": attempt.job_id,
        "created_at": created_at.astimezone(dt.timezone.utc).isoformat(),
        "validation_job_id": attempt.validation_job_id,
        "validation_manifest_id": validation_manifest_id,
        "strategy_version_id": attempt.strategy_version_id,
        "evaluation": {
            "job_id": evaluation.job_id,
            "research_family": evaluation.family_hash,
            "returns_sha256": evaluation.returns_sha256,
            "dataset_ids": sorted(evaluation.dataset_ids),
        },
        "dataset_family": attempt.dataset_family,
        "holdout_start": attempt.holdout_start.isoformat(),
        "policy_id": policy_id_,
        "passed": grade_["passed"],
        "grade": {"scheme": GRADE_SCHEME, "sha256": grade_sha256(grade_)},
        "holdout_returns_sha256": grade_.get("returns_sha256"),
        "engine_version": ENGINE_VERSION,
        "dependency_lock_sha256": dependency_lock_sha256(),
        "code_commit": code_commit(),
        "signing_key_id": signing_key_id,
    }


def manifest_id(canonical: bytes) -> str:
    return f"{MANIFEST_SCHEME}:{hashlib.sha256(canonical).hexdigest()}"


def record(db: Session, *, attempt: models.HoldoutAttempt, evaluation: models.ResearchTrialEvaluation,
           policy: HoldoutPolicy, grade_: dict, created_at: dt.datetime) -> models.HoldoutEvaluation:
    """Build, sign and store the grade and its manifest, in the caller's
    transaction (signed as validation manifests are)."""
    from . import validation_manifest

    try:
        with db.begin_nested():  # a failed registration must not abort the transaction
            key_id = validation_manifest._signing_key_id(db)
    except Exception as exc:
        if validation_manifest._production():
            raise ManifestUnavailable("the server's ML-DSA key is unavailable") from exc
        log.warning("holdout evaluation %s: ML-DSA unavailable, signing its manifest with HMAC: %s",
                    attempt.job_id, exc)
        key_id = None
    verdict_manifest = validation_manifest.get(db, attempt.validation_job_id)
    manifest = build_manifest(attempt=attempt, evaluation=evaluation,
                              validation_manifest_id=verdict_manifest.manifest_id if verdict_manifest else None,
                              policy_id_=policy_id(policy), grade_=grade_, created_at=created_at,
                              signing_key_id=key_id)
    canonical = dataset_format.canonical_json(manifest)
    row = models.HoldoutEvaluation(job_id=attempt.job_id, policy_id=policy_id(policy),
                                   policy_canonical=policy_canonical(policy), passed=grade_["passed"], grade=grade_,
                                   manifest_id=manifest_id(canonical), canonical=canonical,
                                   signature=validation_manifest._sign(canonical, key_id), signing_key_id=key_id,
                                   created_at=created_at)
    db.add(row)
    db.flush()
    return row


def signature_valid(db: Session, row: models.HoldoutEvaluation) -> bool:
    """Whether the grade's manifest is signed as validation manifests are."""
    from . import validation_manifest
    return validation_manifest.signature_valid(db, row)


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------

def holdout_queries(db: Session, user_id: str, family_hash: str) -> int:
    """How many holdout attempts the user has spent in the research family."""
    return int(db.execute(select(func.count()).select_from(models.HoldoutAttempt).where(
        models.HoldoutAttempt.user_id == user_id, models.HoldoutAttempt.family_hash == family_hash)).scalar() or 0)


def owner_view(db: Session, user_id: str, limit: int = 50) -> list[dict]:
    """The user's holdout attempts and, where graded, whether each passed:
    no figure of the holdout."""
    rows = db.execute(
        select(models.HoldoutAttempt, models.HoldoutEvaluation.passed, models.HoldoutEvaluation.policy_id,
               models.HoldoutEvaluation.manifest_id)
        .outerjoin(models.HoldoutEvaluation, models.HoldoutEvaluation.job_id == models.HoldoutAttempt.job_id)
        .where(models.HoldoutAttempt.user_id == user_id)
        .order_by(models.HoldoutAttempt.created_at.desc(), models.HoldoutAttempt.job_id).limit(limit)).all()
    return [{"job_id": a.job_id, "validation": a.validation_job_id, "strategy_version_id": a.strategy_version_id,
             "dataset_family": a.dataset_family, "holdout_start": a.holdout_start.isoformat(),
             "graded": passed is not None, "passed": passed, "policy": policy, "manifest_id": manifest,
             "created_at": a.created_at.isoformat()}
            for a, passed, policy, manifest in rows]
