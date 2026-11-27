"""QuantumSentinel — research sweeps: one strategy over a parameter grid.

A sweep is the deterministic generator of a research pipeline (R6-2a). It
takes one of the user's strategy versions and a grid over its signal
parameters, makes every grid point a candidate spec, registers each valid
one as a strategy version, and backtests them all on the same frozen
datasets, period and capital, as jobs of the pipeline (stage ``sweep``).
No model is involved: the grid is the whole search, and it is fixed before
anything runs.

Bounding the search. Every candidate is a trial of the base version's
family (its template on its symbols, research_trials.family_hash), so a
sweep is admitted only if the family's trials, with this sweep's, stay
within the minimum backtest length bound for the data the candidates will
read (``minbtl_cap``). At the planning Sharpe ratio of 1.0 (the user's
decision of 2026-11-27) that allows 7 trials on two years of data, 45 on
five and 724 on ten. The cap governs sweep admission only: a backtest queued
by hand in the family is not refused, it raises the count the next sweep is
held to.

What counts. A family's trials (``family_trials``) are its ledger trials
(research_trials), its holdout attempts (holdout_evaluation) and every grid
point a sweep refused as a spec: a candidate that failed is still a
configuration that was tried (``n_rejected``, which the sweep row keeps
forever). Admission also counts each earlier sweep's valid candidates that
have no trial yet, so two sweeps created back to back cannot both pass on
the same count: a candidate that never runs keeps holding its place.
A candidate that fails the risk gate still records its trial (the backtest
reports the gate, it does not refuse the run), and a spec that parses
always compiles and runs: a window longer than the data runs flat. So a
backtest that fails leaves no trial only for a cause every candidate on the
same panel shares (too few sessions, a crash, a cancellation); it showed no
result to select on, and admission keeps counting it all the same.

Paying and queueing. Creating a sweep charges its pipeline for every valid
candidate at once (research_pipelines.charge), all or nothing. The
candidates are then queued as the user's active-job limit allows
(``advance``): when the sweep is created and on each POST .../advance.
Nothing queues them in the background, so a sweep no client advances
stalls. A closed pipeline queues nothing more; its candidates stay unqueued
and their budget stays spent.
"""
from __future__ import annotations

import datetime as dt
import itertools
import json
import math
from typing import Mapping, Sequence

from pydantic import ValidationError
from sqlalchemy import distinct, exists, func, select, update
from sqlalchemy.orm import Session

from .. import models, schemas
from . import dataset_holdouts, dataset_registry, research_jobs, research_pipelines, research_trials
from . import strategy_registry, strategy_spec, validation_stats
from .research_pipelines import JobRefused

STAGE = "sweep"
JOB_KIND = "strategy_backtest"
# The annualised Sharpe ratio a sweep's family is planned for (MinBTL).
PLANNING_SHARPE = 1.0
# Where the search for the cap stops (about 99 years of data at Sharpe 1.0
# would allow far more); no budget comes near it.
MINBTL_CEILING = 1_000_000
REJECTION_MAX = 500
_DAYS_PER_YEAR = 365.25
_TEMPLATES = ("ma_crossover", "momentum", "mean_reversion")

_UNSYNCED = {"synchronize_session": False}


class SweepNotFound(JobRefused):
    def __init__(self):
        super().__init__(404, "Research sweep not found")


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _aware(value: dt.datetime | None) -> dt.datetime | None:
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=dt.timezone.utc)
    return value


# --------------------------------------------------------------------------
# The bound (no database)
# --------------------------------------------------------------------------

def period_years(period: str) -> float:
    """A trailing period (``6mo``, ``2y``) in years."""
    from .market_data import PERIOD

    match = PERIOD.fullmatch(period) if isinstance(period, str) else None
    if match is None:
        raise ValueError(f"not a period: {period!r}")
    count, unit = int(match.group(1)), match.group(2)
    return count / 12 if unit == "mo" else float(count)


def minbtl_cap(years: float, planning_sharpe: float = PLANNING_SHARPE) -> int:
    """The most trials a family may hold for ``years`` of data: the largest N
    whose minimum backtest length is at most that long when the best of N
    skill-less strategies is expected to show ``planning_sharpe``
    (validation_stats.min_backtest_length). In years the bound is
    (E[max_N] / SR_annual)^2, whatever the number of sessions in a year, so
    it needs only the calendar span. One trial is always allowed: the
    expected maximum of a single draw is 0."""
    if isinstance(years, bool) or not isinstance(years, (int, float)) or not math.isfinite(years) or years < 0:
        raise ValueError("the data span is a non-negative number of years")
    if not planning_sharpe > 0:
        raise ValueError("the planning Sharpe ratio must be positive")

    def fits(n: int) -> bool:
        return (validation_stats.expected_max(n) / planning_sharpe) ** 2 <= years

    low, high = 1, MINBTL_CEILING
    if fits(high):
        return high
    while high - low > 1:  # fits(low) and not fits(high); E[max_N] rises with N
        mid = (low + high) // 2
        low, high = (mid, high) if fits(mid) else (low, mid)
    return low


def data_years(period: str, datasets: Sequence[models.DatasetVersion], holdout_start: dt.date | None) -> float:
    """The years of data a sweep's backtests read: the trailing ``period``,
    cut to the span every dataset covers (their panel is aligned by date) and
    to the family's research data, which ends before its holdout start."""
    start = max(d.coverage_start for d in datasets)
    end = min(d.coverage_end for d in datasets)
    if holdout_start is not None:
        end = min(end, holdout_start - dt.timedelta(days=1))
    span = max(0, (end - start).days) / _DAYS_PER_YEAR
    return min(period_years(period), span)


# --------------------------------------------------------------------------
# The grid (no database)
# --------------------------------------------------------------------------

def grid_keys(spec: strategy_spec.StrategySpec) -> tuple[str, ...]:
    """The parameters a sweep may vary: the base template's signal
    parameters. Anything else would move a candidate to another family
    (template, symbols) or make versions the engine runs identically
    (risk limits, data family), which would count as trials and teach
    nothing."""
    return tuple(sorted(name for name in type(spec.signal).model_fields if name != "template"))


def check_grid(spec: strategy_spec.StrategySpec, grid: Mapping[str, Sequence[int]]) -> None:
    allowed = grid_keys(spec)
    unknown = sorted(set(grid) - set(allowed))
    if unknown:
        raise JobRefused(422, f"A {spec.signal.template} sweep varies only {', '.join(allowed)}; "
                              f"not {', '.join(unknown)}")


def expand(grid: Mapping[str, Sequence[int]]) -> list[dict]:
    """The grid's points in order: keys sorted, each key's values ascending,
    the last key varying fastest. Each point is the parameters it sets."""
    keys = sorted(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(sorted(grid[k]) for k in keys))]


def candidate_spec(base: strategy_spec.StrategySpec, params: Mapping[str, int]) -> dict:
    """The base spec with the point's signal parameters set, as parsed JSON."""
    value = json.loads(strategy_spec.canonical(base))
    value["signal"].update(params)
    return value


def rejection_reason(exc: ValidationError) -> str:
    """Why a candidate spec was refused, without echoing its input."""
    parts = []
    for error in exc.errors():
        loc = ".".join(str(p) for p in error.get("loc", ()) if p not in _TEMPLATES)
        parts.append(f"{loc}: {error.get('msg')}" if loc else str(error.get("msg")))
    return "; ".join(parts)[:REJECTION_MAX] or "invalid spec"


def _trial_config_hash(version_id: str, period: str, initial_capital: float) -> str:
    from .research_tasks import strategy_trial_config
    return research_trials.config_hash(strategy_trial_config(schemas.StrategyBacktestJob(
        strategy_version_id=version_id, period=period, initial_capital=initial_capital)))


# --------------------------------------------------------------------------
# A family's trials (database)
# --------------------------------------------------------------------------

def rejections(db: Session, user_id: str, family: str) -> int:
    """The grid points the user's sweeps in the family refused as specs."""
    return int(db.execute(select(func.coalesce(func.sum(models.ResearchSweep.n_rejected), 0)).where(
        models.ResearchSweep.user_id == user_id, models.ResearchSweep.family_hash == family)).scalar() or 0)


def family_trials(db: Session, user_id: str, family: str) -> dict:
    """The number of trials every statistic of the family deflates by
    (``trials``) and its parts: the ledger's research trials, the holdout
    attempts spent in the family and the candidates sweeps refused as specs.
    The one place N is summed (trial_matrix.family_state, the stat-test)."""
    from .holdout_evaluation import holdout_queries
    research = research_trials.count(db, user_id, family)
    holdout = holdout_queries(db, user_id, family)
    rejected = rejections(db, user_id, family)
    return {"trials": research + holdout + rejected, "research_trials": research,
            "holdout_queries": holdout, "sweep_rejections": rejected}


def _pending_configs(db: Session, user_id: str, family: str) -> set[str]:
    """The trial configs of the user's earlier sweep candidates in the family
    that the ledger does not hold yet: queued, running or never queued."""
    in_ledger = exists().where(models.ResearchTrial.user_id == user_id,
                               models.ResearchTrial.family_hash == family,
                               models.ResearchTrial.config_hash == models.ResearchSweepCandidate.config_hash)
    return set(db.execute(
        select(distinct(models.ResearchSweepCandidate.config_hash))
        .join(models.ResearchSweep, models.ResearchSweep.id == models.ResearchSweepCandidate.sweep_id)
        .where(models.ResearchSweep.user_id == user_id, models.ResearchSweep.family_hash == family,
               models.ResearchSweepCandidate.config_hash.is_not(None), ~in_ledger)).scalars())


def _ledger_configs(db: Session, user_id: str, family: str, hashes: set[str]) -> set[str]:
    if not hashes:
        return set()
    return set(db.execute(select(models.ResearchTrial.config_hash).where(
        models.ResearchTrial.user_id == user_id, models.ResearchTrial.family_hash == family,
        models.ResearchTrial.config_hash.in_(hashes))).scalars())


# --------------------------------------------------------------------------
# Creating and advancing a sweep (database)
# --------------------------------------------------------------------------

def _datasets(db: Session, user_id: str, dataset_ids: Sequence[str], spec: strategy_spec.StrategySpec):
    datasets = []
    for dataset_id in dataset_ids:
        dataset = dataset_registry.get_for_user(db, user_id, dataset_id)
        if dataset is None:
            raise JobRefused(404, f"Dataset not found: {dataset_id}")
        if dataset.family != spec.universe.family:
            raise JobRefused(422, f"The strategy trades {spec.universe.family} data, but {dataset_id} is "
                                  f"{dataset.family} data")
        datasets.append(dataset)
    covered = {str(s).upper() for d in datasets for s in (d.symbols or ())}
    missing = sorted(set(spec.universe.symbols) - covered)
    if missing:
        raise JobRefused(422, f"The datasets hold no prices for {', '.join(missing)}")
    return datasets


def get_for_user(db: Session, user_id: str, sweep_id: str) -> models.ResearchSweep | None:
    sweep = db.get(models.ResearchSweep, sweep_id) if isinstance(sweep_id, str) else None
    return sweep if sweep is not None and sweep.user_id == user_id else None


def create(db: Session, user_id: str, pipeline_id: str, req: schemas.ResearchSweepRequest, *,
           idempotency_key: str | None = None, request_hash: str | None = None
           ) -> tuple[models.ResearchSweep, bool]:
    """Create a sweep in the user's pipeline and queue what the active-job
    limit allows; return it and whether it is new. A retried request (same
    ``idempotency_key`` and ``request_hash``) gets the sweep the first one
    created, spending nothing; another request under a used key is refused.
    Raises JobRefused for a request the sweep cannot honour."""
    if idempotency_key is not None:
        if not research_jobs.IDEMPOTENCY_KEY.fullmatch(idempotency_key) or not request_hash:
            raise JobRefused(400, "invalid Idempotency-Key")
    else:
        request_hash = None
    base = strategy_registry.resolve(db, user_id, req.strategy)
    if base is None:
        raise JobRefused(404, "Strategy version not found")
    try:
        spec = strategy_registry.spec_of(base)
    except strategy_registry.StrategyRegistryError as exc:
        raise JobRefused(422, str(exc)) from None
    check_grid(spec, req.grid)
    datasets = _datasets(db, user_id, req.dataset_ids, spec)
    family = research_trials.family_hash(spec.signal.template, spec.universe.symbols)
    holdout_start = dataset_holdouts.current(db).get(spec.universe.family)
    cap = minbtl_cap(data_years(req.period, datasets, holdout_start))

    # Every point, judged before anything is locked or written.
    points = []
    for params in expand(req.grid):
        value = candidate_spec(spec, params)
        try:
            candidate = strategy_spec.parse(value)
        except ValidationError as exc:
            points.append({"params": params, "rejection": rejection_reason(exc)})
            continue
        version_id = strategy_spec.version_id(candidate)
        points.append({"params": params, "value": value, "version_id": version_id,
                       "config_hash": _trial_config_hash(version_id, req.period, req.initial_capital)})
    valid = [p for p in points if "version_id" in p]
    n_rejected = len(points) - len(valid)
    if not valid:
        raise JobRefused(422, "No point of the grid is a valid strategy spec")

    with research_jobs.queue_transaction(db, user_id):
        if idempotency_key is not None:
            earlier = db.execute(select(models.ResearchSweep).where(
                models.ResearchSweep.user_id == user_id,
                models.ResearchSweep.idempotency_key == idempotency_key)).scalar_one_or_none()
            if earlier is not None:
                if earlier.request_hash != request_hash or earlier.pipeline_id != pipeline_id:
                    raise JobRefused(409, "Idempotency-Key was already used for a different research sweep")
                return earlier, False
        # Under the user's lock: no other sweep or enqueue of this user can
        # change the family's count until this transaction ends.
        before = family_trials(db, user_id, family)["trials"]
        pending = _pending_configs(db, user_id, family)
        new_hashes = {p["config_hash"] for p in valid}
        known = _ledger_configs(db, user_id, family, new_hashes) | pending
        before += len(pending)
        trials_after = before + len(new_hashes - known) + n_rejected
        if trials_after > cap:
            raise JobRefused(409, f"The family would hold {trials_after} trials, more than the {cap} its data "
                                  f"allows at a planning Sharpe ratio of {PLANNING_SHARPE:g} (minimum backtest "
                                  f"length); it holds {before} before this sweep")
        research_pipelines.charge(db, user_id, pipeline_id, research_jobs.RESEARCH_PIPELINE_DAILY_JOBS,
                                  utcnow().date(), jobs=len(valid))
        for point in valid:
            version = strategy_registry.register(db, point["value"], user_id=user_id)
            if version.version_id != point["version_id"]:
                raise AssertionError("a candidate registered under another id")
        sweep = models.ResearchSweep(
            user_id=user_id, pipeline_id=pipeline_id, base_version_id=base.version_id, family_hash=family,
            grid={k: sorted(v) for k, v in sorted(req.grid.items())}, period=req.period,
            initial_capital=req.initial_capital, dataset_ids=sorted(req.dataset_ids),
            n_candidates=len(points), n_rejected=n_rejected, trials_after=trials_after, minbtl_cap=cap,
            idempotency_key=idempotency_key, request_hash=request_hash, created_at=utcnow())
        db.add(sweep)
        db.flush()
        db.add_all(models.ResearchSweepCandidate(
            sweep_id=sweep.id, ordinal=ordinal, params=point["params"],
            strategy_version_id=point.get("version_id"), config_hash=point.get("config_hash"),
            rejection=point.get("rejection")) for ordinal, point in enumerate(points))
        db.flush()
    advance(db, user_id, sweep.id)
    return sweep, True


def advance(db: Session, user_id: str, sweep_id: str) -> int:
    """Queue the sweep's next candidates, as many as the user's active-job
    limit leaves room for, in grid order; return how many were queued. A
    closed pipeline queues none. Each candidate is queued once: its job is
    recorded only where none is yet (migration 0019's trigger refuses any
    other change)."""
    with research_jobs.queue_transaction(db, user_id):
        sweep = get_for_user(db, user_id, sweep_id)
        if sweep is None:
            raise SweepNotFound()
        if research_pipelines.lock_open(db, sweep.pipeline_id) is None:
            return 0
        room = research_jobs.RESEARCH_MAX_ACTIVE_JOBS_PER_USER - research_jobs.active_jobs(db, user_id)
        if room <= 0:
            return 0
        candidates = db.execute(
            select(models.ResearchSweepCandidate)
            .where(models.ResearchSweepCandidate.sweep_id == sweep.id,
                   models.ResearchSweepCandidate.job_id.is_(None),
                   models.ResearchSweepCandidate.strategy_version_id.is_not(None))
            .order_by(models.ResearchSweepCandidate.ordinal).limit(room)
            .with_for_update()).scalars().all()
        queued = 0
        for candidate in candidates:
            params = schemas.StrategyBacktestJob(
                strategy_version_id=candidate.strategy_version_id, period=sweep.period,
                initial_capital=sweep.initial_capital, dataset_ids=list(sweep.dataset_ids)).model_dump()
            job = research_jobs.insert_prepaid(db, user_id, JOB_KIND, params, sweep.pipeline_id, STAGE)
            done = db.execute(update(models.ResearchSweepCandidate).execution_options(**_UNSYNCED).where(
                models.ResearchSweepCandidate.sweep_id == sweep.id,
                models.ResearchSweepCandidate.ordinal == candidate.ordinal,
                models.ResearchSweepCandidate.job_id.is_(None)).values(job_id=job.id, queued_at=utcnow()))
            if done.rowcount != 1:
                raise AssertionError("a sweep candidate was queued twice")
            queued += 1
    return queued


def view(db: Session, sweep: models.ResearchSweep) -> dict:
    """A sweep as the API reports it: its bound, its candidates and, for each
    queued one, its job (while kept) and what its run measured."""
    candidates = db.execute(select(models.ResearchSweepCandidate)
                            .where(models.ResearchSweepCandidate.sweep_id == sweep.id)
                            .order_by(models.ResearchSweepCandidate.ordinal)
                            .execution_options(populate_existing=True)).scalars().all()
    job_ids = [c.job_id for c in candidates if c.job_id]
    statuses = dict(db.execute(select(models.ResearchJob.id, models.ResearchJob.status).where(
        models.ResearchJob.id.in_(job_ids))).all()) if job_ids else {}
    evaluations = {e.job_id: e for e in db.execute(select(models.ResearchTrialEvaluation).where(
        models.ResearchTrialEvaluation.job_id.in_(job_ids))).scalars()} if job_ids else {}
    pipeline = db.get(models.ResearchPipeline, sweep.pipeline_id)
    out = []
    for c in candidates:
        evaluation = evaluations.get(c.job_id)
        out.append({
            "ordinal": c.ordinal, "params": c.params, "strategy_version_id": c.strategy_version_id,
            "rejection": c.rejection, "job_id": c.job_id,
            "queued_at": _aware(c.queued_at).isoformat() if c.queued_at else None,
            "job_status": statuses.get(c.job_id) if c.job_id else None,
            "evaluation": {"sharpe": evaluation.sharpe, "n_obs": evaluation.n_obs,
                           "returns_sha256": evaluation.returns_sha256} if evaluation else None,
        })
    valid = sum(1 for c in candidates if c.strategy_version_id)
    queued = sum(1 for c in candidates if c.job_id)
    return {
        "sweep_id": sweep.id, "pipeline_id": sweep.pipeline_id,
        "pipeline_status": pipeline.status if pipeline else None,
        "base_version_id": sweep.base_version_id, "family_hash": sweep.family_hash,
        "grid": sweep.grid, "period": sweep.period, "initial_capital": sweep.initial_capital,
        "dataset_ids": sweep.dataset_ids, "stage": STAGE,
        "n_candidates": sweep.n_candidates, "n_rejected": sweep.n_rejected, "n_queued": queued,
        "n_unqueued": valid - queued,
        "minbtl": {"planning_sharpe": PLANNING_SHARPE, "cap": sweep.minbtl_cap,
                   "trials_after": sweep.trials_after},
        "created_at": _aware(sweep.created_at).isoformat(),
        "candidates": out,
    }
