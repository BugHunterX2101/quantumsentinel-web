"""Research sweeps (R6-2a): one strategy version over a grid of its signal
parameters, as backtests of a research pipeline.

A sweep turns every grid point into a candidate spec, registers each valid
one as a strategy version and backtests it on the same frozen datasets
(stage ``sweep``). It is admitted only if the family's trials, with its own,
stay within the minimum backtest length bound for the data at a planning
Sharpe ratio of 1.0; points refused as specs count among them. The schema
comes from the migrations, so migration 0019's CHECKs and triggers are the
ones under test.
"""
import datetime as dt
import json
import types

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import func, select, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import sessionmaker

from backend import main, models, schemas, worker
from backend.services import (dataset_format, dataset_holdouts, research_jobs, research_pipelines,
                              research_sweeps, research_trials, strategy_spec, trial_matrix, validation_stats)
from market_helpers import fake_market
from research_job_helpers import InlineJobProcess, run_queued
from test_strategy_spec import spec

SYMBOLS = ["AAA", "BBB", "CCC"]
FAMILY = research_trials.family_hash("ma_crossover", SYMBOLS)
START, END = dt.date(2018, 1, 2), dt.date(2024, 3, 30)


# --------------------------------------------------------------------------
# The bound and the grid (no database)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("period,cap", [("1mo", 1), ("6mo", 2), ("1y", 3), ("2y", 7), ("5y", 45),
                                        ("10y", 724)])
def test_the_cap_is_the_approved_minbtl_table(period, cap):
    """The user's decision of 2026-11-27: 7, 45 and 724 trials for two, five
    and ten years at a planning Sharpe ratio of 1.0."""
    assert research_sweeps.minbtl_cap(research_sweeps.period_years(period)) == cap


@pytest.mark.parametrize("years", [0.0, 0.3, 1.0, 1.24, 2.0, 3.7, 5.0, 10.0])
def test_the_cap_is_the_largest_n_whose_minimum_backtest_length_fits(years):
    cap = research_sweeps.minbtl_cap(years)
    assert cap == 1 or validation_stats.expected_max(cap) ** 2 <= years
    assert validation_stats.expected_max(cap + 1) ** 2 > years
    # A lower planning Sharpe ratio needs more data per trial.
    assert research_sweeps.minbtl_cap(years, 2.0) >= cap >= research_sweeps.minbtl_cap(years, 0.5)


def test_the_cap_stops_at_its_ceiling_and_refuses_what_is_not_a_span():
    assert research_sweeps.minbtl_cap(99.0) == research_sweeps.MINBTL_CEILING
    for bad in (-1.0, float("nan"), float("inf"), True, "2"):
        with pytest.raises(ValueError):
            research_sweeps.minbtl_cap(bad)
    for sharpe in (0.0, -1.0):
        with pytest.raises(ValueError):
            research_sweeps.minbtl_cap(2.0, sharpe)
    for bad in ("2", "2w", "", None):
        with pytest.raises(ValueError):
            research_sweeps.period_years(bad)


def _dataset(start, end):
    return types.SimpleNamespace(coverage_start=start, coverage_end=end)


def test_the_data_span_is_the_period_cut_to_what_every_dataset_covers_before_the_holdout():
    d = dt.date
    years = research_sweeps.data_years
    both = [_dataset(d(2015, 1, 1), d(2024, 1, 1)), _dataset(d(2020, 1, 1), d(2025, 1, 1))]
    assert years("10y", both, None) == pytest.approx((d(2024, 1, 1) - d(2020, 1, 1)).days / 365.25)
    assert years("2y", both, None) == 2.0
    # The research data ends the day before the holdout starts.
    assert years("10y", both, d(2022, 1, 1)) == pytest.approx((d(2021, 12, 31) - d(2020, 1, 1)).days / 365.25)
    # Datasets that share no date, or a holdout before the data, leave nothing.
    assert years("5y", [_dataset(d(2020, 1, 1), d(2021, 1, 1)), _dataset(d(2022, 1, 1), d(2023, 1, 1))], None) == 0
    assert years("5y", both, d(2019, 1, 1)) == 0
    assert research_sweeps.minbtl_cap(0) == 1


def test_the_grid_expands_in_a_fixed_order():
    points = research_sweeps.expand({"slow": [30, 20], "fast": [5, 2]})
    assert points == [{"fast": 2, "slow": 20}, {"fast": 2, "slow": 30}, {"fast": 5, "slow": 20},
                      {"fast": 5, "slow": 30}]
    assert research_sweeps.expand({"fast": [5, 2], "slow": [30, 20]}) == points


@pytest.mark.parametrize("signal,keys", [
    ({"template": "ma_crossover", "fast": 5, "slow": 20}, ("fast", "slow")),
    ({"template": "momentum", "lookback": 20, "n_long": 1, "n_short": 0}, ("lookback", "n_long", "n_short")),
    ({"template": "mean_reversion", "window": 20, "n_std_centi": 200}, ("n_std_centi", "window")),
])
def test_a_sweep_varies_only_the_templates_signal_parameters(signal, keys):
    base = strategy_spec.parse(spec(signal=signal))
    assert research_sweeps.grid_keys(base) == keys
    research_sweeps.check_grid(base, {k: [2] for k in keys})
    for other in ("template", "fraction_bps", "max_drawdown_bps", "symbols"):
        with pytest.raises(research_pipelines.JobRefused) as caught:
            research_sweeps.check_grid(base, {keys[0]: [2], other: [3]})
        assert caught.value.status_code == 422 and other in caught.value.detail


def test_a_candidate_is_the_base_spec_with_the_points_parameters():
    base = strategy_spec.parse(spec())
    value = research_sweeps.candidate_spec(base, {"fast": 7})
    assert value["signal"] == {"template": "ma_crossover", "fast": 7, "slow": 50}
    assert value["universe"] == json.loads(strategy_spec.canonical(base))["universe"]
    assert strategy_spec.parse(value).signal.fast == 7


def test_a_rejection_says_why_without_echoing_the_input():
    for params, says in (({"fast": 987_654}, "signal.fast"), ({"fast": 60}, "slow must be larger than fast")):
        with pytest.raises(ValidationError) as caught:
            strategy_spec.parse(research_sweeps.candidate_spec(strategy_spec.parse(spec()), params))
        reason = research_sweeps.rejection_reason(caught.value)
        assert says in reason and str(params["fast"]) not in reason and "ma_crossover" not in reason
        assert len(reason) <= research_sweeps.REJECTION_MAX


DATASET = f"{dataset_format.DATASET_SCHEME}:{'0' * 64}"


def request(**fields):
    return schemas.ResearchSweepRequest(**{"strategy": "base", "grid": {"fast": [5, 10]},
                                           "dataset_ids": [DATASET], **fields})


def test_the_request_accepts_a_grid_it_can_sweep():
    assert request().grid == {"fast": [5, 10]}


@pytest.mark.parametrize("fields", [
    {"grid": {}},
    {"grid": {"fast": []}},
    {"grid": {"fast": [5, 5]}},
    {"grid": {"fast": [5.0]}},
    {"grid": {"fast": ["5"]}},
    {"grid": {"fast": [True]}},
    {"grid": {"Fast": [5]}},
    {"grid": {"a": [1], "b": [1], "c": [1], "d": [1], "e": [1]}},
    {"grid": {"fast": list(range(2, 53))}},
    {"grid": {"fast": list(range(2, 27)), "slow": list(range(30, 51))}},
    {"dataset_ids": []},
    {"dataset_ids": ["d1"]},
    {"period": "2w"},
    {"initial_capital": 10.0},
    {"strategy": ""},
    {"pipeline_id": "p"},
])
def test_the_request_refuses_a_grid_or_field_it_cannot_sweep(fields):
    with pytest.raises(ValidationError):
        request(**fields)


def test_the_request_needs_its_datasets_and_accepts_the_largest_grid():
    with pytest.raises(ValidationError):
        schemas.ResearchSweepRequest(strategy="base", grid={"fast": [5]})
    largest = request(grid={"fast": list(range(2, 27)), "slow": list(range(30, 50))})
    assert len(research_sweeps.expand(largest.grid)) == schemas.SWEEP_MAX_CANDIDATES


# --------------------------------------------------------------------------
# Sweeps, end to end
# --------------------------------------------------------------------------

@pytest.fixture
def Session(make_engine):
    return sessionmaker(bind=make_engine(migrated=True))


@pytest.fixture
def db(Session):
    session = Session()
    yield session
    session.close()


@pytest.fixture
def market(monkeypatch, tmp_path):
    """About six years of sessions, so a five-year sweep may hold 45 trials."""
    days = pd.bdate_range(START, "2024-03-29")
    rng = np.random.default_rng(31)
    data = {}
    for i, t in enumerate(SYMBOLS):
        data[("Close", t)] = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, len(days))))
        data[("Volume", t)] = 1e6 + i
    return fake_market(monkeypatch, tmp_path, pd.DataFrame(data, index=days))


def make_user(db, email="researcher@example.com"):
    user = models.User(email=email, password_hash="x", role="user")
    db.add(user)
    db.commit()
    return user


def body(response):
    return json.loads(response.body)


def snapshot(db, user, symbols=SYMBOLS) -> list[str]:
    view = run_queued(db, main.snapshot_dataset(schemas.DatasetSnapshotRequest(
        symbols=list(symbols), start=START, end=END), user=user, db=db))
    assert view["status"] == "succeeded", view
    return [d["dataset_id"] for d in view["result"]["datasets"]]


def register(db, user, tag="base", **changes):
    value = spec(**{"signal.fast": 5, "signal.slow": 20, "universe.symbols": list(SYMBOLS), **changes})
    return main.register_strategy(schemas.StrategyRegisterRequest(spec=value, tag=tag), user=user,
                                  db=db)["version_id"]


def pipeline(db, user, max_jobs=20):
    return research_pipelines.create(db, user.id, "sweep", max_jobs).id


def sweep(db, user, pid, grid, datasets, period="5y", routing=None, **fields):
    req = schemas.ResearchSweepRequest(strategy=fields.pop("strategy", "base"), grid=grid, period=period,
                                       dataset_ids=datasets, **fields)
    return main.create_research_sweep(pid, req, user=user, db=db, routing=routing or main.JobRouting())


def refused(status_code, call, *args, **kwargs):
    with pytest.raises(HTTPException) as caught:
        call(*args, **kwargs)
    assert caught.value.status_code == status_code, caught.value.detail
    return caught.value.detail


def run_all(db) -> int:
    runner = worker.Worker(job_process=InlineJobProcess(), session_factory=sessionmaker(bind=db.get_bind()),
                           lease_seconds=60, timeout_seconds=300, poll_seconds=0.1)
    ran = 0
    while runner.run_once():
        ran += 1
    db.expire_all()
    return ran


def drain(db, user, sweep_id) -> dict:
    """Run and advance the sweep until every valid candidate has run."""
    while True:
        run_all(db)
        if main.advance_research_sweep(sweep_id, user=user, db=db)["queued"] == 0:
            return main.get_research_sweep(sweep_id, user=user, db=db)


def counters(db, pid, user):
    db.expire_all()
    queued = db.get(models.ResearchPipeline, pid).jobs_queued
    today = db.execute(select(func.coalesce(func.sum(models.ResearchPipelineDay.jobs_queued), 0))
                       .where(models.ResearchPipelineDay.user_id == user.id)).scalar_one()
    return queued, today


def count(db, model):
    db.expire_all()
    return db.execute(select(func.count()).select_from(model)).scalar_one()


def written(db):
    """What a refused sweep must not have written."""
    return (count(db, models.ResearchSweep), count(db, models.ResearchSweepCandidate),
            count(db, models.StrategyVersion), count(db, models.ResearchJob), count(db, models.AuditLog))


GRID = {"fast": [5, 10, 30], "slow": [20, 40]}  # (30, 20) is no spec: slow must exceed fast


def test_a_sweep_registers_charges_queues_and_runs_its_candidates(db, market):
    user = make_user(db)
    datasets = snapshot(db, user)
    base = register(db, user)
    pid = pipeline(db, user, max_jobs=10)
    response = sweep(db, user, pid, GRID, datasets)
    assert response.status_code == 201
    created = body(response)
    sid = created["sweep_id"]
    assert response.headers["location"] == f"/api/research/sweeps/{sid}"
    assert (created["n_candidates"], created["n_rejected"], created["n_queued"], created["n_unqueued"]) == \
        (6, 1, 3, 2)
    assert created["minbtl"] == {"planning_sharpe": 1.0, "cap": 45, "trials_after": 6}
    assert (created["base_version_id"], created["family_hash"], created["stage"]) == (base, FAMILY, "sweep")
    assert created["grid"] == {"fast": [5, 10, 30], "slow": [20, 40]} and created["dataset_ids"] == sorted(datasets)
    assert [c["params"] for c in created["candidates"]] == research_sweeps.expand(GRID)
    [rejected] = [c for c in created["candidates"] if c["rejection"]]
    assert rejected["params"] == {"fast": 30, "slow": 20} and rejected["strategy_version_id"] is None
    assert "slow must be larger than fast" in rejected["rejection"] and rejected["job_id"] is None
    # The base itself is the (5, 20) point: registering it again is the same version.
    assert created["candidates"][0]["strategy_version_id"] == base
    assert count(db, models.StrategyVersion) == 5
    # The pipeline paid for all five valid candidates; three are queued.
    assert counters(db, pid, user) == (5, 5)
    jobs = db.execute(select(models.ResearchJob).where(models.ResearchJob.kind == "strategy_backtest")
                      .order_by(models.ResearchJob.created_at)).scalars().all()
    assert [(j.pipeline_id, j.stage) for j in jobs] == [(pid, "sweep")] * 3
    assert {j.id for j in jobs} == {c["job_id"] for c in created["candidates"][:3]}
    assert jobs[0].params_json["dataset_ids"] == sorted(datasets) and jobs[0].params_json["period"] == "5y"
    [audit] = db.execute(select(models.AuditLog).where(models.AuditLog.action == "SWEEP_CREATED")).scalars()
    assert (audit.resource_type, audit.resource_id) == ("research_sweep", sid)
    assert audit.metadata_json == {"pipeline_id": pid, "base_version_id": base, "n_candidates": 6,
                                   "n_rejected": 1, "trials_after": 6, "minbtl_cap": 45}

    # Nothing more is queued while the active-job limit is reached.
    assert main.advance_research_sweep(sid, user=user, db=db)["queued"] == 0
    assert run_all(db) == 3
    advanced = main.advance_research_sweep(sid, user=user, db=db)
    assert advanced["queued"] == 2 and advanced["n_unqueued"] == 0
    done = drain(db, user, sid)
    assert counters(db, pid, user) == (5, 5)
    valid = [c for c in done["candidates"] if c["strategy_version_id"]]
    assert all(c["job_status"] == "succeeded" and c["evaluation"]["n_obs"] > 0 for c in valid), valid
    assert len({c["evaluation"]["returns_sha256"] for c in valid}) == 5

    # The family holds the five runs and the refused point: what the sweep was admitted at.
    assert research_sweeps.family_trials(db, user.id, FAMILY) == {
        "trials": 6, "research_trials": 5, "holdout_queries": 0, "sweep_rejections": 1}
    state = trial_matrix.family_state(db, user.id, FAMILY)
    assert (state["trials"], state["sweep_rejections"], state["evaluated_trials"]) == (6, 1, 5)
    assert state["every_trial_evaluated"] is True
    assert sorted(state["strategy_version_ids"]) == sorted(c["strategy_version_id"] for c in valid)
    rows = db.execute(select(models.ResearchTrialEvaluation)).scalars().all()
    assert all(sorted(r.dataset_ids) == sorted(datasets) for r in rows) and len(rows) == 5

    # A stat test deflates by every one of them; the trial matrix reports them.
    stat = main.statistical_tests(schemas.StatTestRequest(returns=[0.01, -0.02, 0.003, 0.004, -0.001],
                                                          trial_family=FAMILY), user=user, db=db)
    assert db.get(models.ResearchJob, body(stat)["job_id"]).params_json["server_counted"] == 6
    queued = main.validation_matrix(schemas.ValidationMatrixRequest(
        evaluation=valid[0]["job_id"], period="5y", initial_capital=100_000.0), user=user, db=db)
    assert run_all(db) == 2  # the stat test, then the matrix
    matrix = research_jobs.view(db, db.get(models.ResearchJob, body(queued)["job_id"]))
    assert matrix["status"] == "succeeded", matrix
    ledger = matrix["result"]["ledger"]
    assert (ledger["trials"], ledger["sweep_rejections"], ledger["every_trial_evaluated"]) == (6, 1, True)


def test_the_cap_refuses_a_sweep_that_would_exceed_it_and_writes_nothing(db, market):
    """Two years of data allow seven trials. Six valid candidates fit; with
    two refused points they would be eight."""
    user = make_user(db)
    datasets = snapshot(db, user)
    register(db, user)
    pid = pipeline(db, user)
    before = written(db)
    detail = refused(409, sweep, db, user, pid, {"fast": [2, 3, 4, 5, 6, 7, 25, 30]}, datasets, period="2y")
    assert "8 trials, more than the 7" in detail and "holds 0 before" in detail
    assert written(db) == before and counters(db, pid, user) == (0, 0)
    assert research_sweeps.family_trials(db, user.id, FAMILY)["trials"] == 0
    fits = body(sweep(db, user, pid, {"fast": [2, 3, 4, 5, 6, 7, 30]}, datasets, period="2y"))
    assert fits["minbtl"] == {"planning_sharpe": 1.0, "cap": 7, "trials_after": 7} and fits["n_rejected"] == 1
    assert refused(409, sweep, db, user, pid, {"fast": [8]}, datasets, period="2y")
    assert counters(db, pid, user) == (6, 6)


def test_back_to_back_sweeps_count_each_others_unrun_candidates_once(db, market):
    user = make_user(db)
    datasets = snapshot(db, user)
    register(db, user)
    pid = pipeline(db, user)
    first = body(sweep(db, user, pid, {"fast": [2, 3, 4, 5]}, datasets, period="2y"))
    assert first["minbtl"]["trials_after"] == 4
    # Four more would make eight: the first sweep's unrun candidates hold their places.
    assert "holds 4 before" in refused(409, sweep, db, user, pid, {"fast": [6, 7, 8, 9]}, datasets, period="2y")
    # A sweep repeating three of them adds only the new one.
    second = body(sweep(db, user, pid, {"fast": [2, 3, 4, 6]}, datasets, period="2y"))
    assert second["minbtl"]["trials_after"] == 5
    drain(db, user, first["sweep_id"])
    drain(db, user, second["sweep_id"])
    assert research_trials.count(db, user.id, FAMILY) == 5
    # Once run, they count as the ledger's trials, still once.
    third = body(sweep(db, user, pid, {"fast": [2, 3]}, datasets, period="2y"))
    assert third["minbtl"]["trials_after"] == 5
    drain(db, user, third["sweep_id"])
    assert research_sweeps.family_trials(db, user.id, FAMILY)["trials"] == 5


def test_a_trial_the_ledger_holds_is_not_counted_again(db, market):
    user = make_user(db)
    datasets = snapshot(db, user)
    register(db, user)
    by_hand = run_queued(db, main.strategy_backtest(schemas.StrategyBacktestRequest(
        strategy="base", period="2y", dataset_ids=datasets), user=user, db=db))
    assert by_hand["status"] == "succeeded" and research_trials.count(db, user.id, FAMILY) == 1
    pid = pipeline(db, user)
    created = body(sweep(db, user, pid, {"fast": [5, 6]}, datasets, period="2y"))
    assert created["minbtl"]["trials_after"] == 2
    # Another capital is another trial config.
    other = body(sweep(db, user, pid, {"fast": [5]}, datasets, period="2y", initial_capital=50_000.0))
    assert other["minbtl"]["trials_after"] == 3


def test_the_cap_reads_only_the_research_data_before_the_holdout(db, market):
    user = make_user(db)
    datasets = snapshot(db, user)
    register(db, user)
    pid = pipeline(db, user)
    holdout = dt.date(2021, 6, 1)
    dataset_holdouts.set_start(db, "EQUITY_EOD", holdout, set_by="ops", reason="test")
    db.commit()
    created = body(sweep(db, user, pid, {"fast": [2]}, datasets))
    rows = [db.get(models.DatasetVersion, d) for d in datasets]
    expected = research_sweeps.minbtl_cap(research_sweeps.data_years("5y", rows, holdout))
    assert created["minbtl"]["cap"] == expected and 7 < expected < 45


def test_a_closed_pipeline_queues_nothing_more_and_keeps_what_it_paid(db, market):
    user = make_user(db)
    datasets = snapshot(db, user)
    register(db, user)
    pid = pipeline(db, user)
    created = body(sweep(db, user, pid, GRID, datasets))
    assert created["n_queued"] == 3
    main.close_research_pipeline(pid, user=user, db=db)
    run_all(db)
    advanced = main.advance_research_sweep(created["sweep_id"], user=user, db=db)
    assert advanced["queued"] == 0 and advanced["pipeline_status"] == "closed" and advanced["n_unqueued"] == 2
    assert counters(db, pid, user) == (5, 5)
    with pytest.raises(research_pipelines.PipelineClosed):
        with research_jobs.queue_transaction(db, user.id):
            research_jobs.insert_prepaid(db, user.id, "strategy_backtest", {}, pid, "sweep")
    # A closed pipeline takes no new sweep either.
    assert refused(409, sweep, db, user, pid, {"fast": [2]}, datasets)
    assert counters(db, pid, user) == (5, 5) and count(db, models.ResearchSweep) == 1


@pytest.mark.parametrize("budget", ["pipeline", "daily"])
def test_a_sweep_the_budget_cannot_pay_for_whole_is_refused_and_writes_nothing(db, market, monkeypatch, budget):
    user = make_user(db)
    datasets = snapshot(db, user)
    register(db, user)
    if budget == "pipeline":
        pid, status, says = pipeline(db, user, max_jobs=4), 409, "has 4 of its 4 jobs left; 5 are needed"
    else:
        pid, status, says = pipeline(db, user), 429, "4 more pipeline jobs today"
        monkeypatch.setattr(research_jobs, "RESEARCH_PIPELINE_DAILY_JOBS", 4)
    before = written(db)
    assert says in refused(status, sweep, db, user, pid, GRID, datasets)
    assert written(db) == before and counters(db, pid, user) == (0, 0)
    assert body(sweep(db, user, pid, {"fast": [5, 10, 15, 16]}, datasets))["n_candidates"] == 4
    assert counters(db, pid, user) == (4, 4)


def test_a_retried_sweep_is_the_first_one_and_a_used_key_is_refused_for_another(db, market):
    user = make_user(db)
    datasets = snapshot(db, user)
    register(db, user)
    pid = pipeline(db, user)
    routing = main.JobRouting(idempotency_key="sweep-key-1", request_hash="a" * 64)
    first = body(sweep(db, user, pid, GRID, datasets, routing=routing))
    again = sweep(db, user, pid, GRID, datasets, routing=routing)
    assert body(again)["sweep_id"] == first["sweep_id"]
    assert counters(db, pid, user) == (5, 5) and count(db, models.ResearchSweep) == 1
    assert db.execute(select(func.count()).select_from(models.AuditLog)
                      .where(models.AuditLog.action == "SWEEP_CREATED")).scalar_one() == 1
    other = pipeline(db, user)
    for pid_, hash_ in ((pid, "b" * 64), (other, "a" * 64)):
        detail = refused(409, sweep, db, user, pid_, GRID, datasets,
                         routing=main.JobRouting(idempotency_key="sweep-key-1", request_hash=hash_))
        assert detail == "Idempotency-Key was already used for a different research sweep"
    assert counters(db, pid, user) == (5, 5) and counters(db, other, user)[0] == 0


def test_a_sweep_takes_only_the_callers_own_strategy_pipeline_datasets_and_sweep(db, market):
    alice, bob = make_user(db, "alice@example.com"), make_user(db, "bob@example.com")
    datasets = snapshot(db, alice)
    register(db, alice)
    pid = pipeline(db, alice)
    sid = body(sweep(db, alice, pid, {"fast": [2, 30]}, datasets, period="2y"))["sweep_id"]
    before = written(db)
    assert refused(404, sweep, db, bob, pid, {"fast": [2]}, datasets) == "Research pipeline not found"
    bob_pid = pipeline(db, bob)
    assert refused(404, sweep, db, bob, bob_pid, {"fast": [2]}, datasets) == "Strategy version not found"
    register(db, bob)
    assert refused(404, sweep, db, bob, bob_pid, {"fast": [2]}, datasets).startswith("Dataset not found")
    assert refused(404, sweep, db, alice, "no-such-pipeline", {"fast": [2]}, datasets)
    for call in (main.get_research_sweep, main.advance_research_sweep):
        assert refused(404, call, sid, user=bob, db=db) == "Research sweep not found"
        assert refused(404, call, "no-such-sweep", user=alice, db=db) == "Research sweep not found"
    assert written(db)[:2] == before[:2]
    # Alice's refused point and unrun candidate are hers: Bob's family starts empty.
    assert research_sweeps.family_trials(db, bob.id, FAMILY)["trials"] == 0
    bobs = body(sweep(db, bob, bob_pid, {"fast": [2]}, snapshot(db, bob), period="2y"))
    assert bobs["minbtl"]["trials_after"] == 1
    assert research_sweeps.family_trials(db, alice.id, FAMILY)["sweep_rejections"] == 1


def test_a_sweep_refuses_a_request_it_cannot_honour_and_writes_nothing(db, market):
    user = make_user(db)
    datasets = snapshot(db, user)
    register(db, user)
    register(db, user, tag="wide", **{"universe.symbols": ["AAA", "DDD"]})
    pid = pipeline(db, user)
    before = written(db)
    assert "varies only fast, slow" in refused(422, sweep, db, user, pid, {"fraction_bps": [100]}, datasets)
    assert refused(422, sweep, db, user, pid, {"fast": [30, 40]}, datasets) == \
        "No point of the grid is a valid strategy spec"
    assert "hold no prices for DDD" in refused(422, sweep, db, user, pid, {"fast": [2]}, datasets, strategy="wide")
    assert "X-Research-Pipeline" in refused(422, sweep, db, user, pid, {"fast": [2]}, datasets,
                                            routing=main.JobRouting(pipeline_id="another"))
    assert refused(400, sweep, db, user, pid, {"fast": [2]}, datasets,
                   routing=main.JobRouting(idempotency_key="bad key!", request_hash="a" * 64))
    assert written(db) == before and counters(db, pid, user) == (0, 0)


def test_a_sweep_of_another_familys_datasets_is_refused(db, monkeypatch, tmp_path):
    days = pd.bdate_range("2023-01-02", "2024-03-29")
    rng = np.random.default_rng(2)
    fake_market(monkeypatch, tmp_path, pd.DataFrame(
        {("Close", t): 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days)))) for t in SYMBOLS + ["BTC-USD"]},
        index=days), crypto=("BTC-USD",))
    user = make_user(db)
    view = run_queued(db, main.snapshot_dataset(schemas.DatasetSnapshotRequest(
        symbols=SYMBOLS + ["BTC-USD"], start="2023-01-02", end="2024-03-30"), user=user, db=db))
    families = {d["identity"]["family"]: d["dataset_id"] for d in view["result"]["datasets"]}
    register(db, user)
    pid = pipeline(db, user)
    detail = refused(422, sweep, db, user, pid, {"fast": [2]}, [families["CRYPTO_SPOT_EOD"]], period="1y")
    assert "trades EQUITY_EOD data" in detail
    created = body(sweep(db, user, pid, {"fast": [2]}, [families["EQUITY_EOD"]], period="2y"))
    # About 1.24 years of data, whatever the period asks for.
    assert created["minbtl"]["cap"] == research_sweeps.minbtl_cap(1.24) == 4


# --------------------------------------------------------------------------
# The schema (migration 0019)
# --------------------------------------------------------------------------

def _refused(db, statement, **params):
    with pytest.raises(DBAPIError):
        db.execute(text(statement), params)
        db.flush()
    db.rollback()


@pytest.fixture
def swept(db, market):
    user = make_user(db)
    datasets = snapshot(db, user)
    register(db, user)
    pid = pipeline(db, user)
    created = body(sweep(db, user, pid, GRID, datasets))
    return user, created


@pytest.mark.parametrize("statement", [
    "UPDATE research_sweeps SET n_rejected = 0",
    "UPDATE research_sweeps SET minbtl_cap = minbtl_cap",
    "DELETE FROM research_sweeps",
    "TRUNCATE research_sweeps CASCADE",
    "UPDATE research_sweep_candidates SET job_id = 'other', queued_at = now() WHERE job_id IS NOT NULL",
    "UPDATE research_sweep_candidates SET params = '{}' WHERE job_id IS NULL",
    "UPDATE research_sweep_candidates SET rejection = NULL, strategy_version_id = NULL WHERE rejection IS NOT NULL",
    "UPDATE research_sweep_candidates SET job_id = 'j', queued_at = now(), params = '{}' WHERE job_id IS NULL",
    "DELETE FROM research_sweep_candidates",
    "TRUNCATE research_sweep_candidates",
])
def test_sweeps_and_their_candidates_never_change_but_to_be_queued_once(db, swept, statement):
    _refused(db, statement)
    assert count(db, models.ResearchSweep) == 1 and count(db, models.ResearchSweepCandidate) == 6
    assert db.execute(select(models.ResearchSweep.n_rejected)).scalar_one() == 1


def test_an_unqueued_candidate_may_be_queued_once(db, swept):
    _, created = swept
    job = db.execute(select(models.ResearchJob.id)).scalars().first()
    db.execute(text("UPDATE research_sweep_candidates SET job_id = :job, queued_at = now()"
                    " WHERE sweep_id = :sid AND ordinal = 3"), {"job": job, "sid": created["sweep_id"]})
    db.commit()
    _refused(db, "UPDATE research_sweep_candidates SET job_id = :job, queued_at = now()"
                 " WHERE sweep_id = :sid AND ordinal = 3", job=job, sid=created["sweep_id"])


_INSERT_SWEEP = ("INSERT INTO research_sweeps (id, user_id, pipeline_id, base_version_id, family_hash, grid,"
                 " period, initial_capital, dataset_ids, n_candidates, n_rejected, trials_after, minbtl_cap,"
                 " idempotency_key, request_hash, created_at) VALUES (:id, :user_id, :pipeline_id,"
                 " :base_version_id, :family_hash, :grid, :period, :initial_capital, :dataset_ids, :n_candidates,"
                 " :n_rejected, :trials_after, :minbtl_cap, :idempotency_key, :request_hash, :created_at)")


@pytest.mark.parametrize("changes", [
    {"n_candidates": 0, "n_rejected": 0},
    {"n_rejected": -1},
    {"n_rejected": 3},
    {"minbtl_cap": 0, "trials_after": 0},
    {"trials_after": 46},
    {"dataset_ids": "[]"},
    {"dataset_ids": '{"d": 1}'},
    {"idempotency_key": "k"},
    {"request_hash": "a" * 64},
])
def test_the_schema_refuses_a_sweep_that_cannot_be_one(db, swept, changes):
    user, created = swept
    good = {"id": "s-x", "user_id": user.id, "pipeline_id": created["pipeline_id"],
            "base_version_id": created["base_version_id"], "family_hash": FAMILY, "grid": '{"fast": [2]}',
            "period": "5y", "initial_capital": 1e5, "dataset_ids": '["d"]', "n_candidates": 2, "n_rejected": 1,
            "trials_after": 3, "minbtl_cap": 45, "idempotency_key": None, "request_hash": None,
            "created_at": dt.datetime.now(dt.timezone.utc)}
    _refused(db, _INSERT_SWEEP, **{**good, **changes})
    db.execute(text(_INSERT_SWEEP), good)  # the row the change broke is one
    db.rollback()


_INSERT_CANDIDATE = ("INSERT INTO research_sweep_candidates (sweep_id, ordinal, params, strategy_version_id,"
                     " config_hash, rejection, job_id, queued_at) VALUES (:sid, :ordinal, '{}', :version,"
                     " :config, :rejection, :job, :queued)")


@pytest.mark.parametrize("changes", [
    {"ordinal": -1},
    {"rejection": "both"},
    {"version": None},
    {"config": None},
    {"version": None, "config": None, "rejection": "refused", "job": "j", "queued": dt.datetime(2024, 1, 1)},
    {"job": "j"},
    {"queued": dt.datetime(2024, 1, 1)},
])
def test_the_schema_refuses_a_candidate_that_cannot_be_one(db, swept, changes):
    _, created = swept
    good = {"sid": created["sweep_id"], "ordinal": 99, "version": created["base_version_id"], "config": "c" * 64,
            "rejection": None, "job": None, "queued": None}
    _refused(db, _INSERT_CANDIDATE, **{**good, **changes})
    db.execute(text(_INSERT_CANDIDATE), good)  # the row the change broke is one
    db.rollback()
