"""QuantumSentinel — strategy validations (R3-4).

A strategy validation judges one strategy evaluation against the validation
policy (``validation_policy.evaluate``): the risk gate on the candidate's
re-run, and the statistical gates on its research family's trial matrix
(``trial_matrix``).

The ``strategy_validation`` job runs in two parts. The job process builds
the matrix exactly as the ``validation_matrix`` job does (it re-runs every
strategy version evaluated in the family on the candidate's datasets) and
reports the candidate's risk measurements beside it. The worker then
accepts the matrix as it accepts a validation matrix
(``trial_matrix.accept_matrix``: the evaluation's datasets, a family that did
not change meanwhile, a well-formed matrix with every evaluated version),
checks that the candidate's column is its evaluation's return series, and
computes the verdict itself from the matrix it accepted, the family's trials
as the ledger holds them, and the candidate's spec as the registry holds it.
Nothing the job process says about the verdict is kept; only its
measurements are used, and those it cannot misstate without the matrix's id
or the evaluation's hash giving it away, apart from the risk measurements,
which the risk gate takes as the engine reported them.

The verdict and the matrix are stored in the transaction that stores the
job's result (migration 0008): the matrix once under its id, the verdict once
per job, both append-only. The job records no trial. A verdict is
pre-holdout: passing it makes a candidate eligible for the one-shot holdout
evaluation, which is not part of it.
"""
from __future__ import annotations

import datetime as dt
import json
import logging

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from .. import models
from . import trial_matrix

log = logging.getLogger(__name__)


def finalize_validation_job(db: Session, job: models.ResearchJob, result: dict, audit: dict):
    """Accept the job's matrix, judge the candidate against the validation
    policy, and record the matrix and the verdict."""
    from . import strategy_registry, validation_policy
    from .research_tasks import TaskError

    evaluation, state, matrix = trial_matrix.accept_matrix(db, job, result)
    version_id = evaluation.strategy_version_id
    column = matrix["columns"][trial_matrix.column_of(matrix, version_id)]
    if column["returns_sha256"] != evaluation.returns_sha256:
        raise TaskError(500, "The research job reported a matrix in which the candidate is not its evaluation")
    version = strategy_registry.get_for_user(db, job.user_id, version_id)
    if version is None:
        raise TaskError(422, f"The candidate's strategy version {version_id} is not held")
    try:
        spec = strategy_registry.spec_of(version)
    except strategy_registry.StrategyRegistryError as exc:
        raise TaskError(422, str(exc)) from exc

    policy = validation_policy.POLICY_V1
    verdict = validation_policy.evaluate(matrix, version_id, state, spec.universe.family, spec.risk_limits,
                                         result.get("risk"), policy)
    try:
        json.dumps(verdict, allow_nan=False)
    except ValueError as exc:
        log.error("strategy validation job %s: its verdict is not finite: %s", job.id, exc)
        raise TaskError(500, "The validation's verdict could not be recorded") from exc

    now = dt.datetime.now(dt.timezone.utc)
    canonical = trial_matrix.canonical(matrix)
    db.execute(insert(models.TrialMatrix).values(matrix_id=matrix["id"], canonical=canonical, created_at=now)
               .on_conflict_do_nothing(index_elements=["matrix_id"]))
    db.add(models.StrategyValidation(
        job_id=job.id, user_id=job.user_id, evaluation_job_id=evaluation.job_id, strategy_version_id=version_id,
        family_hash=evaluation.family_hash, matrix_id=matrix["id"], policy_id=verdict["policy"],
        policy_canonical=validation_policy.policy_canonical(policy), passed=verdict["passed"], verdict=verdict,
        created_at=now))
    db.flush()

    result = {key: value for key, value in result.items() if key not in ("matrix", "risk")}
    result.update(matrix_id=matrix["id"], validation=verdict)
    audit = {**audit, "policy": verdict["policy"], "passed": verdict["passed"],
             "failed_gates": sorted(name for name, gate in verdict["gates"].items() if not gate["passed"])}
    return result, ("strategy_version", version_id), audit


# --------------------------------------------------------------------------
# Reading verdicts
# --------------------------------------------------------------------------

def list_for_user(db: Session, user_id: str, limit: int) -> list[models.StrategyValidation]:
    """The user's verdicts, newest first."""
    return list(db.execute(select(models.StrategyValidation)
                           .where(models.StrategyValidation.user_id == user_id)
                           .order_by(models.StrategyValidation.created_at.desc(),
                                     models.StrategyValidation.job_id)
                           .limit(limit)).scalars())


def get_for_user(db: Session, user_id: str, job_id: str) -> models.StrategyValidation | None:
    row = db.get(models.StrategyValidation, job_id) if isinstance(job_id, str) else None
    return row if row is not None and row.user_id == user_id else None


def view(row: models.StrategyValidation) -> dict:
    return {"job_id": row.job_id, "evaluation": row.evaluation_job_id,
            "strategy_version_id": row.strategy_version_id, "family": row.family_hash,
            "matrix_id": row.matrix_id, "policy": {"id": row.policy_id, **json.loads(row.policy_canonical)},
            "passed": row.passed, "verdict": row.verdict, "created_at": row.created_at.isoformat()}


def matrix_of(db: Session, row: models.StrategyValidation) -> dict:
    """The trial matrix a verdict judged, as trial_matrix.build gave it."""
    stored = db.get(models.TrialMatrix, row.matrix_id)
    return {"id": stored.matrix_id, **json.loads(stored.canonical)}
