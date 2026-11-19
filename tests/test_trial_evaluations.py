"""Strategy-version backtests and the trial evaluations they record.

POST /api/research/strategy-backtest queues the ``strategy_backtest`` job:
the worker loads the version's spec for the job's user, the job process runs
it with the event engine on the universe's prices, and once the worker has
accepted the result it records the run as a trial of the template's family
and what it measured as an evaluation of that trial
(services/research_trials.py, migration 0007).
"""
import datetime as dt
import hashlib
import json
import math
import struct

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException
from sqlalchemy import select, text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.orm import sessionmaker

from backend import main, models, schemas, worker
from backend.services import (backtest_service, dataset_holdouts, research_jobs, research_tasks,
                              research_trials, strategy_registry, strategy_spec)
from backend.services.backtest_service import StrategyType
from market_helpers import fake_market
from research_job_helpers import InlineJobProcess, run_queued
from test_strategy_spec import prices, spec

SYMBOLS = ["AAA", "BBB", "CCC"]


# --------------------------------------------------------------------------
# The return-series id and the metrics (no database)
# --------------------------------------------------------------------------

GOLDEN_RETURNS = [0.0, 0.01, -0.005, 1e-12, -1e-12, 0.123456789012345, -0.9999, 4e-13, -6e-13, 1234.5678]
GOLDEN_QUANTISED = [0, 10_000_000_000, -5_000_000_000, 1, -1, 123_456_789_012, -999_900_000_000, 0, -1,
                    1_234_567_800_000_000]
GOLDEN_RETURNS_ID = "qs-returns-v1:0c9c473bc56e8052dfd43f84dfe69e05d5c150b2a71d9be348a68a3914ca67c4"


def test_the_return_series_id_matches_its_golden_vector():
    """Pinned, and recomputed here from the definition (big-endian int64 of
    round(r * 1e12)), so another implementation can check itself against it."""
    assert [round(r * 1e12) for r in GOLDEN_RETURNS] == GOLDEN_QUANTISED
    independent = hashlib.sha256(b"".join(struct.pack(">q", q) for q in GOLDEN_QUANTISED)).hexdigest()
    assert GOLDEN_RETURNS_ID == f"qs-returns-v1:{independent}"
    assert research_trials.returns_id(GOLDEN_RETURNS) == GOLDEN_RETURNS_ID
    assert research_trials.returns_id(np.array(GOLDEN_RETURNS)) == GOLDEN_RETURNS_ID
    assert research_trials.RETURNS_ID.fullmatch(GOLDEN_RETURNS_ID)


def test_the_id_ignores_differences_below_the_quantum_and_sees_order_and_length():
    base = [0.01, -0.02, 0.003]
    assert research_trials.returns_id([r + 1e-14 for r in base]) == research_trials.returns_id(base)
    assert research_trials.returns_id([0.01, -0.02, 0.003 + 2e-12]) != research_trials.returns_id(base)
    assert research_trials.returns_id(base[::-1]) != research_trials.returns_id(base)
    assert research_trials.returns_id(base + [0.0]) != research_trials.returns_id(base)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, 1e7])
def test_the_id_refuses_what_it_cannot_represent(bad):
    with pytest.raises(ValueError):
        research_trials.returns_id([0.01, bad])


def test_the_metrics_are_the_engines_statistics_of_its_full_series():
    """Sharpe, n_obs and the id all come from the one unrounded series the
    engine computes its own statistics from (full_returns), not from the
    504 rounded returns it shows."""
    compiled = strategy_spec.compile_spec(strategy_spec.parse(spec(**{"signal.fast": 5, "signal.slow": 20})))
    data = prices(compiled.tickers, bars=700)
    result = compiled.run(data, 100_000.0, full_returns=True)
    returns = result["returns"]
    assert len(returns) == result["n_bars"] == 700 and len(result["daily_returns"]) == 504
    assert [round(float(r), 6) for r in returns[-504:]] == result["daily_returns"]
    metrics = research_trials.evaluation_metrics(returns)
    assert metrics["n_obs"] == 700
    assert round(metrics["sharpe"], 4) == result["sharpe_ratio"]
    assert metrics["returns_sha256"] == research_trials.returns_id(returns)
    # Without the flag the engine's output is as before.
    assert "returns" not in compiled.run(data, 100_000.0)


@pytest.mark.parametrize("bad", [[0.01], [], [0.01, math.nan]])
def test_the_metrics_need_two_finite_returns(bad):
    with pytest.raises(ValueError):
        research_trials.evaluation_metrics(bad)


def test_the_family_hash_matches_its_golden_value():
    """Strategy-version trials join the family the moving-average backtests
    of the same symbols count in, so its hash must never drift."""
    golden = "e05f4e0d2d1d2f645fbcb658640cc997c4e15793d6f89bad114d4581b7157b3f"
    assert research_trials.family_hash("ma_crossover", ["MSFT", "AAPL"]) == golden
    assert research_trials.family_hash("ma_crossover", [" aapl", "msft ", "AAPL"]) == golden
    # The engines name their templates with the same strings, so every source
    # lands in one family.
    assert StrategyType.MA_CROSSOVER == "ma_crossover"
    assert research_trials.family_hash(StrategyType.MA_CROSSOVER, ["AAPL", "MSFT"]) == golden
    for template in ("momentum", "mean_reversion"):
        assert getattr(StrategyType, template.upper()) == template


# --------------------------------------------------------------------------
# The job, end to end
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
    days = pd.bdate_range("2023-01-02", "2024-03-29")
    rng = np.random.default_rng(23)
    data = {}
    for i, t in enumerate(SYMBOLS):
        data[("Close", t)] = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, len(days))))
        data[("Volume", t)] = 1e6 + i
    return fake_market(monkeypatch, tmp_path, pd.DataFrame(data, index=days))


def make_user(db, email="researcher@example.com"):
    user = models.User(email=email, password_hash="x", role="user")
    db.add(user)
    db.commit()
    return user


def strategy(**changes):
    return spec(**{"signal.fast": 5, "signal.slow": 20, "universe.symbols": list(SYMBOLS), **changes})


def register(db, user, value=None, tag=None):
    return main.register_strategy(schemas.StrategyRegisterRequest(spec=value or strategy(), tag=tag),
                                  user=user, db=db)["version_id"]


def queue(db, user, ref, period="1y", **fields):
    return main.strategy_backtest(schemas.StrategyBacktestRequest(strategy=ref, period=period, **fields),
                                  user=user, db=db)


def backtest(db, user, ref, job_process=None, **fields):
    return run_queued(db, queue(db, user, ref, **fields), job_process)


def evaluations(db):
    db.expire_all()
    return db.execute(select(models.ResearchTrialEvaluation)
                      .order_by(models.ResearchTrialEvaluation.created_at)).scalars().all()


def trials(db, user, family):
    db.expire_all()
    return research_trials.count(db, user.id, family)


def test_a_backtest_runs_the_version_and_records_its_trial_and_evaluation(db, market):
    user = make_user(db)
    version_id = register(db, user, tag="1.0.0")
    view = backtest(db, user, "1.0.0")
    assert view["status"] == "succeeded", view
    result = view["result"]
    family = research_trials.family_hash("ma_crossover", SYMBOLS)
    assert result["strategy_version_id"] == version_id and result["family"] == "EQUITY_EOD"
    assert result["trial_family"] == family and result["trials_in_family"] == 1
    assert result["sessions"]["count"] >= research_tasks.STRATEGY_MIN_SESSIONS
    assert result["n_trades"] > 0
    evaluation = result["evaluation"]
    assert evaluation["n_obs"] == result["n_bars"] == result["sessions"]["count"]
    assert round(evaluation["sharpe"], 4) == result["sharpe_ratio"]
    # Measured on the unrounded series, not on the rounded returns shown.
    assert len(result["daily_returns"]) == evaluation["n_obs"]
    assert evaluation["returns_sha256"] != research_trials.returns_id(result["daily_returns"])
    [row] = evaluations(db)
    assert (row.job_id, row.user_id, row.family_hash, row.strategy_version_id) == \
        (view["job_id"], user.id, family, version_id)
    assert (row.sharpe, row.n_obs, row.returns_sha256) == \
        (evaluation["sharpe"], evaluation["n_obs"], evaluation["returns_sha256"])
    assert row.dataset_ids == result["market_data"]["datasets"] and row.dataset_ids
    trial = db.execute(select(models.ResearchTrial)).scalar_one()
    assert (trial.family_hash, trial.config_hash, trial.source) == (family, row.config_hash, "strategy_backtest")
    audit = db.execute(select(models.AuditLog).where(models.AuditLog.action == "STRATEGY_BACKTEST")).scalar_one()
    assert (audit.resource_type, audit.resource_id) == ("strategy_version", version_id)
    assert audit.metadata_json["returns_sha256"] == evaluation["returns_sha256"]
    assert audit.metadata_json["datasets"] == row.dataset_ids


def test_a_backtest_reports_its_risk_and_what_the_policys_risk_gate_says_of_it(db, market):
    """The run measures its drawdown and largest position (unrounded) and
    reports the validation policy's risk gate on them."""
    from backend.services import validation_policy
    user = make_user(db)
    capped = strategy(**{"risk_limits.max_drawdown_bps": 2_500, "risk_limits.max_position_bps": 2_500})
    for value, declared_ok in ((capped, True), (strategy(), False)):  # strategy() declares 50% positions
        view = backtest(db, user, register(db, user, value))
        assert view["status"] == "succeeded", view
        result = view["result"]
        risk, check = result["risk"], result["risk_check"]
        assert check == validation_policy.risk_gate(strategy_spec.parse(value).risk_limits, risk,
                                                    validation_policy.CURRENT_POLICY)
        assert check["policy"] == validation_policy.policy_id(validation_policy.CURRENT_POLICY)
        assert check["checks"]["declared_position_within_policy"]["passed"] is declared_ok
        assert check["passed"] is (declared_ok and check["checks"]["drawdown_within_declared"]["passed"])
        # 2% fixed-fractional positions.
        assert risk["equity_exhausted"] is False and 0.015 < risk["max_position_share"] < 0.05
        assert round(-risk["max_drawdown"], 4) == result["max_drawdown"]


def test_the_same_run_on_the_same_data_reproduces_its_evaluation_as_the_same_trial(db, market):
    """A rerun on the datasets the first run snapshotted is the same trial
    (the count stays 1) and a second evaluation with the same measurements;
    another period is another trial."""
    user = make_user(db)
    version_id = register(db, user)
    first = backtest(db, user, version_id)
    again = backtest(db, user, version_id, dataset_ids=first["result"]["market_data"]["datasets"])
    assert again["status"] == "succeeded", again
    assert again["result"]["trials_in_family"] == 1
    a, b = evaluations(db)
    assert (a.job_id, b.job_id) == (first["job_id"], again["job_id"])
    assert (a.config_hash, a.sharpe, a.n_obs, a.returns_sha256) == (b.config_hash, b.sharpe, b.n_obs, b.returns_sha256)
    other = backtest(db, user, version_id, period="6mo")
    assert other["status"] == "succeeded", other
    assert other["result"]["trials_in_family"] == 2
    assert evaluations(db)[2].returns_sha256 != a.returns_sha256


def test_a_version_counts_in_the_family_of_moving_average_backtests_of_its_symbols(db, market, monkeypatch):
    monkeypatch.setattr(backtest_service, "run_moving_average_backtest", lambda *a: {
        "initial_capital": 100_000.0, "final_capital": 101_000.0, "total_return": 0.01,
        "sharpe_ratio": 1.0, "max_drawdown": -0.02, "win_rate": 0.5, "total_trades": 4})
    user = make_user(db)
    dashboard = run_queued(db, main.run_backtest(schemas.BacktestRequest(asset="aaa"), user=user, db=db))
    assert dashboard["status"] == "succeeded", dashboard
    view = backtest(db, user, register(db, user, strategy(**{"universe.symbols": ["AAA"]})))
    assert view["status"] == "succeeded", view
    assert view["result"]["trial_family"] == dashboard["result"]["trial_family"]
    assert view["result"]["trials_in_family"] == 2


def test_only_a_version_the_caller_holds_can_be_queued(db, market):
    owner, other = make_user(db), make_user(db, "other@example.com")
    version_id = register(db, owner, tag="mine")
    for ref in (version_id, "mine", "qs-sv-sha256-v1:" + "0" * 64, "not a ref"):
        with pytest.raises(HTTPException) as caught:
            queue(db, other, ref)
        assert caught.value.status_code == 404
    with pytest.raises(HTTPException) as caught:
        queue(db, owner, version_id, dataset_ids=["qs-ds-sha256-v1:" + "0" * 64])
    assert caught.value.status_code == 404
    assert db.execute(select(models.ResearchJob)).first() is None


def test_the_tag_is_resolved_when_the_job_is_queued(db, market):
    user = make_user(db)
    version_id = register(db, user, tag="prod")
    response = queue(db, user, "prod")
    job = db.get(models.ResearchJob, json.loads(response.body)["job_id"])
    assert job.params_json["strategy_version_id"] == version_id
    assert "strategy" not in job.params_json


class MustNotRun(InlineJobProcess):
    def run(self, *args, **kwargs):
        raise AssertionError("the task ran")


def test_the_worker_runs_only_a_version_the_jobs_user_holds(db, market):
    """The job process gets the spec from the worker, never from params: a
    job naming a version its user does not hold fails before it runs."""
    owner, other = make_user(db), make_user(db, "other@example.com")
    version_id = register(db, owner)
    job = research_jobs.enqueue(db, other.id, "strategy_backtest", schemas.StrategyBacktestJob(
        strategy_version_id=version_id, period="1y", initial_capital=100_000.0).model_dump())
    runner = worker.Worker(job_process=MustNotRun(), session_factory=sessionmaker(bind=db.get_bind()),
                           lease_seconds=60, timeout_seconds=300, poll_seconds=0.1)
    assert runner.run_once() is True
    db.expire_all()
    view = research_jobs.view(db, db.get(models.ResearchJob, job.id))
    assert view["status"] == "failed" and view["error"] == {"status_code": 404,
                                                           "detail": "Strategy version not found"}
    assert db.execute(select(models.ResearchTrial)).first() is None


def test_a_stored_spec_todays_code_refuses_fails_the_job_before_it_runs(db, market, monkeypatch):
    user = make_user(db)
    version_id = register(db, user)

    def refuse(version):
        raise strategy_registry.StrategyRegistryError(f"{version.version_id}: the stored spec no longer validates")

    monkeypatch.setattr(strategy_registry, "spec_of", refuse)
    view = backtest(db, user, version_id, MustNotRun())
    assert view["status"] == "failed" and view["error"]["status_code"] == 422
    assert "no longer validates" in view["error"]["detail"]
    assert evaluations(db) == []


def test_the_real_job_process_receives_what_the_worker_prepared():
    """The worker hands the prepared inputs to a separate child process over
    a pipe, beside the params, and only to the job they were prepared for."""
    process = worker.JobProcess()
    try:
        status, result, _, _ = process.run("research_job_helpers:prepared_task", {"a": 1}, 120, lambda: None,
                                           prepared={"spec": {"dsl": "qs-strategy-v1"}})
        assert status == "ok" and result == {"params": {"a": 1}, "prepared": {"spec": {"dsl": "qs-strategy-v1"}}}
        status, result, _, _ = process.run("research_job_helpers:prepared_task", {}, 120, lambda: None)
        assert status == "ok" and result == {"params": {}, "prepared": None}
    finally:
        process.stop()


def test_the_task_refuses_a_spec_other_than_the_one_its_job_names():
    other = strategy_spec.parse(spec())
    params = schemas.StrategyBacktestJob(strategy_version_id="qs-sv-sha256-v1:" + "a" * 64, period="1y",
                                         initial_capital=100_000.0).model_dump()
    outcome = worker._run_target(research_jobs.KINDS["strategy_backtest"].target, params, None,
                                 {"spec": json.loads(strategy_spec.canonical(other))})
    assert outcome == ("error", 500, "The research job was given another strategy than the one it names")


class SetsAHoldoutMidJob(InlineJobProcess):
    def __init__(self, Session, start):
        self.Session, self.start = Session, start

    def run(self, *args, **kwargs):
        with self.Session() as other:
            dataset_holdouts.set_start(other, "EQUITY_EOD", self.start, set_by="ops", reason="mid-job")
        return super().run(*args, **kwargs)


class LosesItsLease(InlineJobProcess):
    """Another worker takes the job over while the task runs."""

    def __init__(self, Session):
        self.Session = Session

    def run(self, *args, **kwargs):
        with self.Session() as other:
            other.execute(text("UPDATE research_jobs SET worker_id = 'someone-else'"))
            other.commit()
        return super().run(*args, **kwargs)


def test_a_result_refused_for_a_holdout_set_mid_run_counts_the_trial_but_records_no_evaluation(
        db, Session, market):
    user = make_user(db)
    view = backtest(db, user, register(db, user), SetsAHoldoutMidJob(Session, dt.date(2024, 1, 2)))
    assert view["status"] == "failed" and view["error"]["status_code"] == 409
    assert trials(db, user, research_trials.family_hash("ma_crossover", SYMBOLS)) == 1
    assert evaluations(db) == []
    assert db.execute(select(models.AuditLog).where(models.AuditLog.action == "STRATEGY_BACKTEST")).first() is None


def test_a_result_whose_lease_was_lost_counts_the_trial_but_records_no_evaluation(db, Session, market):
    user = make_user(db)
    response = queue(db, user, register(db, user))
    job_id = json.loads(response.body)["job_id"]
    runner = worker.Worker(job_process=LosesItsLease(Session), session_factory=Session,
                           lease_seconds=60, timeout_seconds=300, poll_seconds=0.1)
    assert runner.run_once() is True
    db.expire_all()
    job = db.get(models.ResearchJob, job_id)
    assert (job.status, job.worker_id) == ("running", "someone-else")
    assert trials(db, user, research_trials.family_hash("ma_crossover", SYMBOLS)) == 1
    assert evaluations(db) == []


def test_prices_from_another_family_than_the_universes_are_refused(db, market):
    """``universe.family`` selects the data: equity prices for a spec that
    declares crypto never become its evaluation."""
    user = make_user(db)
    view = backtest(db, user, register(db, user, strategy(**{"universe.family": "CRYPTO_SPOT_EOD"})))
    assert view["status"] == "failed"
    assert view["error"] == {"status_code": 422, "detail": "The strategy trades CRYPTO_SPOT_EOD data, "
                                                           "but its prices came from EQUITY_EOD datasets"}
    assert evaluations(db) == []
    # The configuration was evaluated, so it still counts.
    assert trials(db, user, research_trials.family_hash("ma_crossover", SYMBOLS)) == 1


def test_a_universe_without_enough_prices_is_refused(db, market):
    user = make_user(db)
    view = backtest(db, user, register(db, user, strategy(**{"universe.symbols": ["AAA", "ZZZ"]})))
    assert view["status"] == "failed" and view["error"]["status_code"] == 422
    assert "ZZZ" in view["error"]["detail"]
    assert evaluations(db) == []


def test_the_request_validates_its_period_and_capital():
    for fields in ({"period": "2w"}, {"period": "0y"}, {"initial_capital": 10.0}, {"strategy": ""}):
        with pytest.raises(ValueError):
            schemas.StrategyBacktestRequest(**{"strategy": "x", **fields})
    with pytest.raises(ValueError):
        schemas.StrategyBacktestJob(strategy_version_id="1.0.0", period="1y", initial_capital=100_000.0)


# --------------------------------------------------------------------------
# What the migrated schema guarantees
# --------------------------------------------------------------------------

def _trial_and_version(db) -> tuple[str, dict]:
    user = make_user(db)
    version_id = strategy_registry.register(db, strategy(), user_id=user.id).version_id
    family = research_trials.family_hash("ma_crossover", SYMBOLS)
    config = {"strategy_version_id": version_id, "period": "1y", "initial_capital": 100_000.0}
    research_trials.record(db, user.id, family, [config], "strategy_backtest")
    return user.id, {"family": family, "config": config, "strategy_version_id": version_id}


def _insert(db, user_id, trial, job_id="j1", config_hash=None, **changes):
    values = {"sharpe": 1.0, "n_obs": 250, "returns_sha256": GOLDEN_RETURNS_ID, "dataset_ids": ["d1"], **changes}
    db.add(models.ResearchTrialEvaluation(
        job_id=job_id, user_id=user_id, family_hash=trial["family"],
        config_hash=config_hash or research_trials._digest(trial["config"]),
        strategy_version_id=trial["strategy_version_id"], created_at=dt.datetime.now(dt.timezone.utc),
        **values))
    db.flush()


@pytest.mark.parametrize("changes, constraint", [
    ({"sharpe": math.nan}, "ck_research_trial_evaluations_sharpe_finite"),
    ({"sharpe": math.inf}, "ck_research_trial_evaluations_sharpe_finite"),
    ({"sharpe": -math.inf}, "ck_research_trial_evaluations_sharpe_finite"),
    ({"n_obs": 1}, "ck_research_trial_evaluations_n_obs"),
    ({"returns_sha256": "sha256:" + "0" * 64}, "ck_research_trial_evaluations_returns_sha256"),
    ({"returns_sha256": GOLDEN_RETURNS_ID.upper()}, "ck_research_trial_evaluations_returns_sha256"),
    ({"dataset_ids": []}, "ck_research_trial_evaluations_dataset_ids"),
    ({"dataset_ids": {"d1": 1}}, "ck_research_trial_evaluations_dataset_ids"),
])
def test_the_schema_refuses_an_evaluation_that_cannot_be_one(db, changes, constraint):
    user_id, trial = _trial_and_version(db)
    with pytest.raises(DBAPIError, match=constraint):
        _insert(db, user_id, trial, **changes)
    db.rollback()


def test_an_evaluation_must_name_a_recorded_trial(db):
    user_id, trial = _trial_and_version(db)
    with pytest.raises(IntegrityError, match="fk_research_trial_evaluations_trial"):
        _insert(db, user_id, trial, config_hash="0" * 64)
    db.rollback()
    _insert(db, user_id, trial)  # the recorded one is accepted
    db.commit()


@pytest.mark.parametrize("statement", [
    "UPDATE research_trials SET source = 'x'", "DELETE FROM research_trials", "TRUNCATE research_trials CASCADE",
    "UPDATE research_trial_evaluations SET sharpe = 9", "DELETE FROM research_trial_evaluations",
    "TRUNCATE research_trial_evaluations CASCADE",
])
def test_trials_and_evaluations_are_immutable_even_for_the_owner(db, statement):
    user_id, trial = _trial_and_version(db)
    _insert(db, user_id, trial)
    db.commit()
    with pytest.raises(DBAPIError, match="its rows are immutable"):
        db.execute(text(statement))
    db.rollback()
    assert db.execute(text("SELECT count(*) FROM research_trials")).scalar() == 1
    assert db.execute(text("SELECT count(*) FROM research_trial_evaluations")).scalar() == 1
