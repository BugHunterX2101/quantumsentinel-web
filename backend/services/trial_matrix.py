"""QuantumSentinel — the trial matrix (R3-2).

The statistical gates of the validation policy (R3-3) judge a candidate
strategy version against every variant tried in its research family: the
deflated Sharpe ratio needs how many independent trials there were, and the
probability of backtest overfitting needs their returns, date by date. This
module builds that T x N matrix (T sessions, N trials) by re-running every
strategy version evaluated in the family on the candidate's own data, with
today's engine, rather than storing each run's series when it ran.

Which versions. The columns are the distinct strategy versions among the
user's evaluations in the candidate's family
(``research_trial_evaluations``): a version that was registered but never
run is not a trial. The ledger (``research_trials``) also counts runs that
left no evaluation (results the worker refused, and the moving-average,
advanced and walk-forward backtests that share a template's family); those
have no strategy version to re-run, so the matrix says how many of the
family's trials it covers (``ledger``) and is complete only when every trial
has an evaluation. A later gate that needs every trial falls back to the raw
count otherwise. Building a matrix records no trial: re-running versions
that were already counted must not inflate the count.

What it runs on. The request names the candidate's evaluation (its job id)
with the period and capital it ran with; the server checks those give the
evaluation's trial (its config hash), and every version re-runs on exactly
the datasets that evaluation read, over that period, with that capital. The
candidate's own re-run must reproduce the evaluation's return series (its
``returns_sha256``); if it does not (the engine changed since, or a holdout
moved the window), the matrix is refused and the candidate must be
backtested again. A version that cannot be re-run refuses the whole matrix:
a column left out would understate the number of trials.

The matrix. Each column is one distinct return series: versions that differ
only in fields the engine does not read (``risk_limits``,
``universe.family``) produce the same series and share a column, which
lists them all. Returns are stored as the integers round(r * 10^12) that the
return-series id hashes (``research_trials.quantise``), so a column's values
and its ``returns_sha256`` agree exactly in any language. Columns are
ordered by that id. The matrix id is ``qs-tm-sha256-v1:<hex>``, the SHA-256
of the canonical JSON (``dataset_format.canonical_json``) of the scheme, the
dates and the columns; tests/test_trial_matrix.py holds a golden vector.
The matrix lives in the job's result; a strategy validation also stores it,
beyond the job's retention (services/strategy_validation.py).
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any, Mapping, Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .. import models
from . import research_trials
from .dataset_format import DatasetError, canonical_json

MATRIX_SCHEME = "qs-tm-sha256-v1"
MATRIX_ID = re.compile(rf"{MATRIX_SCHEME}:[0-9a-f]{{64}}")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
log = logging.getLogger(__name__)


class MatrixError(ValueError):
    """Runs or a stored matrix that do not form a valid trial matrix."""


# --------------------------------------------------------------------------
# The matrix (no database)
# --------------------------------------------------------------------------

def build(dates: Sequence[str], runs: Mapping[str, Sequence[float]]) -> dict:
    """The matrix of ``runs`` (strategy version id -> its daily returns, one
    per date in ``dates``), with its id."""
    dates = list(dates)
    if not runs:
        raise MatrixError("a matrix needs at least one run")
    columns: dict[str, dict] = {}
    for version_id, returns in runs.items():
        quantised = research_trials.quantise(returns)
        if len(quantised) != len(dates):
            raise MatrixError(f"{version_id}: {len(quantised)} returns for {len(dates)} sessions")
        series_id = research_trials.quantised_returns_id(quantised)
        column = columns.setdefault(series_id, {"returns_sha256": series_id, "strategy_version_ids": [],
                                                "returns": quantised})
        column["strategy_version_ids"].append(version_id)
    matrix = {"scheme": MATRIX_SCHEME, "dates": dates,
              "columns": [{**columns[key], "strategy_version_ids": sorted(columns[key]["strategy_version_ids"])}
                          for key in sorted(columns)]}
    return {"id": matrix_id(matrix), **matrix}


def matrix_id(matrix: Mapping[str, Any]) -> str:
    """The id of a matrix, after checking it is one (``canonical``)."""
    return f"{MATRIX_SCHEME}:{hashlib.sha256(canonical(matrix)).hexdigest()}"


def canonical(matrix: Mapping[str, Any]) -> bytes:
    """The canonical JSON of a matrix's scheme, dates and columns, which its
    id hashes, after checking it is a matrix: dates strictly increasing,
    every column as long as the dates with values that hash to its
    ``returns_sha256``, columns ordered by that id, and each version in
    exactly one column."""
    if matrix.get("scheme") != MATRIX_SCHEME:
        raise MatrixError(f"not a {MATRIX_SCHEME} matrix")
    dates, columns = matrix.get("dates"), matrix.get("columns")
    if not isinstance(dates, list) or not dates or not all(isinstance(d, str) and _DATE.fullmatch(d)
                                                           for d in dates):
        raise MatrixError("a matrix needs its dates, as YYYY-MM-DD")
    if any(a >= b for a, b in zip(dates, dates[1:])):
        raise MatrixError("the dates must be strictly increasing")
    if not isinstance(columns, list) or not columns:
        raise MatrixError("a matrix needs at least one column")
    versions: list[str] = []
    for column in columns:
        if not isinstance(column, Mapping) or set(column) != {"returns_sha256", "strategy_version_ids", "returns"}:
            raise MatrixError("a column has its returns_sha256, strategy_version_ids and returns")
        returns, held = column["returns"], column["strategy_version_ids"]
        if not isinstance(returns, list) or len(returns) != len(dates):
            raise MatrixError(f"a column has {len(returns) if isinstance(returns, list) else 'no'} returns "
                              f"for {len(dates)} sessions")
        try:
            series_id = research_trials.quantised_returns_id(returns)
        except ValueError as exc:
            raise MatrixError(str(exc)) from exc
        if column["returns_sha256"] != series_id:
            raise MatrixError("a column's returns do not hash to its returns_sha256")
        if not isinstance(held, list) or not held or held != sorted(set(held)):
            raise MatrixError("a column lists its distinct strategy versions in order")
        versions.extend(held)
    ids = [column["returns_sha256"] for column in columns]
    if ids != sorted(set(ids)):
        raise MatrixError("the columns are distinct and ordered by returns_sha256")
    if len(versions) != len(set(versions)):
        raise MatrixError("a strategy version is in more than one column")
    body = {"scheme": MATRIX_SCHEME, "dates": dates,
            "columns": [{key: column[key] for key in ("returns_sha256", "strategy_version_ids", "returns")}
                        for column in columns]}
    try:
        return canonical_json(body)
    except DatasetError as exc:  # a return beyond +/-(2**53 - 1) quanta
        raise MatrixError(f"the matrix cannot be encoded: {exc}") from exc


def column_of(matrix: Mapping[str, Any], version_id: str) -> int:
    """The index of the column that holds ``version_id``."""
    for index, column in enumerate(matrix["columns"]):
        if version_id in column["strategy_version_ids"]:
            return index
    raise MatrixError(f"{version_id} is not in the matrix")


# --------------------------------------------------------------------------
# The family's trials (database)
# --------------------------------------------------------------------------

def evaluation_for_user(db: Session, user_id: str, job_id: str) -> models.ResearchTrialEvaluation | None:
    """The evaluation recorded by ``job_id``, if it is ``user_id``'s."""
    row = db.get(models.ResearchTrialEvaluation, job_id) if isinstance(job_id, str) else None
    return row if row is not None and row.user_id == user_id else None


def ran_with(evaluation: models.ResearchTrialEvaluation, period: str, initial_capital: float) -> bool:
    """Whether ``period`` and ``initial_capital`` are what the evaluation's
    run used: with its version, they give its trial's config hash."""
    from .. import schemas
    from .research_tasks import strategy_trial_config
    config = strategy_trial_config(schemas.StrategyBacktestJob(
        strategy_version_id=evaluation.strategy_version_id, period=period, initial_capital=initial_capital))
    return research_trials.config_hash(config) == evaluation.config_hash


def family_state(db: Session, user_id: str, family: str) -> dict:
    """The family's trials as the ledger holds them now: how many, how many
    have an evaluation, and the strategy versions evaluated.

    ``trials``, the count every statistic deflates by, is the ledger's
    research trials plus the holdout attempts spent in the family
    (holdout_evaluation): a strategy picked after a holdout grade was picked
    from one more look at the data. The holdout runs have no column in a
    trial matrix, so completeness (``every_trial_evaluated``) is judged on
    the research trials alone."""
    from .holdout_evaluation import holdout_queries
    evaluated_configs = select(models.ResearchTrialEvaluation.config_hash).where(
        models.ResearchTrialEvaluation.user_id == user_id,
        models.ResearchTrialEvaluation.family_hash == family)
    evaluated = int(db.execute(select(func.count()).select_from(models.ResearchTrial).where(
        models.ResearchTrial.user_id == user_id, models.ResearchTrial.family_hash == family,
        models.ResearchTrial.config_hash.in_(evaluated_configs))).scalar() or 0)
    research = research_trials.count(db, user_id, family)
    holdout = holdout_queries(db, user_id, family)
    versions = sorted(db.execute(select(models.ResearchTrialEvaluation.strategy_version_id).distinct().where(
        models.ResearchTrialEvaluation.user_id == user_id,
        models.ResearchTrialEvaluation.family_hash == family)).scalars())
    return {"trials": research + holdout, "research_trials": research, "holdout_queries": holdout,
            "evaluated_trials": evaluated, "every_trial_evaluated": evaluated == research,
            "strategy_version_ids": versions}


# --------------------------------------------------------------------------
# Validation matrix jobs (research_jobs.KINDS["validation_matrix"]), run by
# the research worker
# --------------------------------------------------------------------------

def _job_evaluation(db: Session, job: models.ResearchJob):
    from .. import schemas
    from .research_tasks import TaskError

    req = schemas.ValidationMatrixJob.model_validate(job.params_json)
    evaluation = evaluation_for_user(db, job.user_id, req.evaluation)
    if evaluation is None:
        raise TaskError(404, "Evaluation not found")
    if not ran_with(evaluation, req.period, req.initial_capital):
        raise TaskError(422, "The period and initial capital are not the ones the evaluation ran with")
    if sorted(req.dataset_ids or ()) != sorted(evaluation.dataset_ids):
        raise TaskError(422, "The datasets are not the ones the evaluation read")
    return req, evaluation


def matrix_inputs(db: Session, job: models.ResearchJob) -> dict:
    """What a validation matrix job's process is given: the candidate's
    evaluation, the family's trials, and the spec of every strategy version
    evaluated in the family, as today's DSL code reads it. A version that
    can no longer be read refuses the job: it cannot be left out."""
    from . import strategy_registry, strategy_spec
    from .research_tasks import TaskError

    req, evaluation = _job_evaluation(db, job)
    state = family_state(db, job.user_id, evaluation.family_hash)
    specs = {}
    for version_id in state["strategy_version_ids"]:
        version = strategy_registry.get_for_user(db, job.user_id, version_id)
        if version is None:
            raise TaskError(422, f"The family's strategy version {version_id} is not held")
        try:
            specs[version_id] = json.loads(strategy_spec.canonical(strategy_registry.spec_of(version)))
        except strategy_registry.StrategyRegistryError as exc:
            raise TaskError(422, f"The family's trials cannot all be re-run: {exc}") from exc
    return {"candidate": {"evaluation": evaluation.job_id, "strategy_version_id": evaluation.strategy_version_id,
                          "returns_sha256": evaluation.returns_sha256},
            "family": evaluation.family_hash, "ledger": state, "specs": specs}


def finalize_matrix_job(db: Session, job: models.ResearchJob, result: dict, audit: dict):
    """Accept a validation matrix (``accept_matrix``)."""
    evaluation, _, _ = accept_matrix(db, job, result)
    return result, ("strategy_version", evaluation.strategy_version_id), audit


def accept_matrix(db: Session, job: models.ResearchJob, result: dict):
    """Accept the matrix a job reported, after the worker has accepted the
    market data it read: the prices came from exactly the evaluation's
    datasets, the family's trials are still the ones it was built from, and
    the matrix is well formed and holds every version evaluated in the
    family. Returns the evaluation, the family's trials and the matrix."""
    from .research_tasks import TaskError

    _, evaluation = _job_evaluation(db, job)
    if sorted(result["market_data"]["datasets"]) != sorted(evaluation.dataset_ids):
        raise TaskError(422, "The prices came from other datasets than the ones the evaluation read")
    state = family_state(db, job.user_id, evaluation.family_hash)
    if state != result.get("ledger"):
        raise TaskError(409, "The family's trials changed while the matrix was built; build it again")
    matrix = result.get("matrix") or {}
    try:
        valid = matrix_id(matrix) == matrix.get("id")
        held = sorted(v for column in matrix["columns"] for v in column["strategy_version_ids"])
    except (MatrixError, KeyError, TypeError) as exc:
        log.error("validation matrix job %s: its matrix was refused: %s", job.id, exc)
        valid, held = False, None
    if not valid or held != state["strategy_version_ids"]:
        raise TaskError(500, "The research job reported an invalid matrix")
    return evaluation, state, matrix
