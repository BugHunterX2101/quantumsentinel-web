"""The one-shot holdout evaluation (R1-5b-2): services/holdout_evaluation.py,
the final_evaluation job the research evaluator runs, migration 0011's
tables and the holdout counter in a family's trials.

The job tests run end to end on the synthetic market of
test_trial_evaluations, with the EQUITY_EOD holdout set before the first
backtest. The real validation policy cannot pass on that market, so they
force the verdict to pass (``force_pass``); the grade is then computed for
real. The grade's returns are checked against a run of the strategy on the
market's own closes, outside the task's code.
"""
import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import sessionmaker

from backend import main, models, schemas, worker
from backend.services import (dataset_holdouts, holdout_evaluation, market_data, research_jobs, research_tasks,
                              research_trials, strategy_spec, trial_matrix, validation_policy)
from backend.services.research_tasks import TaskError
from research_job_helpers import InlineJobProcess
from test_strategy_validation import validate
from test_trial_evaluations import SYMBOLS, Session, db, make_user, market, strategy  # noqa: F401
from test_trial_matrix import FAMILY, evaluated

START = dt.date(2023, 12, 1)
CAPITAL = 100_000.0
SUMMARY_KEYS = {"holdout_evaluation", "validation", "strategy_version_id", "dataset_family", "holdout_start",
                "policy", "passed"}


def force_pass(monkeypatch):
    """Make every verdict pass: the synthetic market cannot pass the real
    gates, and only a passing verdict is graded."""
    real = validation_policy.evaluate
    monkeypatch.setattr(validation_policy, "evaluate", lambda *a, **k: {**real(*a, **k), "passed": True})


def candidate(db, monkeypatch, user=None, start=START, period="1y", passing=True):
    """A validated candidate under the holdout ``start`` (set first)."""
    user = user or make_user(db)
    if start is not None:
        dataset_holdouts.set_start(db, "EQUITY_EOD", start, set_by="ops", reason="test")
    if passing:
        force_pass(monkeypatch)
    version_id, view = evaluated(db, user, period=period)
    done = validate(db, user, view["job_id"], period=period)
    assert done["status"] == "succeeded", done
    assert done["result"]["validation"]["passed"] is passing
    return user, version_id, view, done["job_id"]


def evaluator(db, job_process=None, role=research_jobs.EVALUATOR_ROLE):
    return worker.Worker(job_process=job_process or InlineJobProcess(),
                         session_factory=sessionmaker(bind=db.get_bind()),
                         lease_seconds=60, timeout_seconds=300, poll_seconds=0.1, role=role)


def run(db, job_id, job_process=None):
    assert evaluator(db, job_process).run_once() is True
    db.expire_all()
    return research_jobs.view(db, db.get(models.ResearchJob, job_id))


def grade_it(db, validation, period="1y", job_process=None):
    job = holdout_evaluation.queue(db, validation, period, CAPITAL)
    return run(db, job.id, job_process)


def rows(db, model):
    db.expire_all()
    return db.execute(select(model)).scalars().all()


def synthetic_closes():
    """The closes of test_trial_evaluations' ``market``, rebuilt."""
    days = pd.bdate_range("2023-01-02", "2024-03-29")
    rng = np.random.default_rng(23)
    return pd.DataFrame({t: 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, len(days)))) for t in SYMBOLS},
                        index=days)


# --------------------------------------------------------------------------
# The policy and the grade (no database)
# --------------------------------------------------------------------------

def test_the_holdout_policy_has_its_own_pinned_id():
    policy = holdout_evaluation.HOLDOUT_POLICY_V1
    assert (policy.name, policy.min_sessions, policy.min_sharpe_annual_milli) == ("qs-holdout-policy-v1", 63, 500)
    assert holdout_evaluation.policy_id(policy) == \
        "qs-hp-sha256-v1:be0ac42c859eb273fc8bc0963c99a6708fc2c314dac42f6f38ba135cc9c60997"
    assert holdout_evaluation.policy_id(policy) != validation_policy.policy_id(validation_policy.CURRENT_POLICY)


def _series(n, mean, sd=0.01, seed=3):
    rng = np.random.default_rng(seed)
    dates = [str(d.date()) for d in pd.bdate_range("2024-01-02", periods=n)]
    returns = rng.normal(0, sd, n)
    returns = returns - returns.mean() + mean
    return dates, research_trials.quantise(list(returns))


def test_the_grade_passes_at_the_floor_and_fails_below_it():
    start = dt.date(2024, 1, 2)
    dates, high = _series(100, 0.0006)
    passed = holdout_evaluation.grade(dates, high, start, "EQUITY_EOD")
    assert passed["passed"] is True and passed["reason"] is None and passed["sharpe_annual"] >= 0.5
    dates, low = _series(100, 0.0001)
    failed = holdout_evaluation.grade(dates, low, start, "EQUITY_EOD")
    assert failed["passed"] is False and failed["sharpe_annual"] < 0.5
    assert failed["reason"] == "the annualised Sharpe ratio is below the policy's floor"
    # The annualisation is the family's calendar: crypto's 365 sessions.
    crypto = holdout_evaluation.grade(dates, high, start, "CRYPTO_SPOT_EOD")
    assert crypto["sharpe_annual"] == pytest.approx(passed["sharpe_annual"] * (365 / 252) ** 0.5)


@pytest.mark.parametrize("dates,quanta,family,reason", [
    (_series(62, 0.01)[0], _series(62, 0.01)[1], "EQUITY_EOD", "62 holdout sessions; the policy needs at least 63"),
    ([], [], "EQUITY_EOD", "0 holdout sessions; the policy needs at least 63"),
    (_series(80, 0.01)[0], [5] * 80, "EQUITY_EOD", "no Sharpe ratio: a constant series has no Sharpe ratio"),
    (_series(80, 0.01)[0], _series(80, 0.01)[1], "MOON_EOD",
     "the policy gives no sessions a year for the 'MOON_EOD' dataset family"),
], ids=["too-few", "none", "constant", "unknown-family"])
def test_a_grade_that_is_not_defined_fails_closed(dates, quanta, family, reason):
    found = holdout_evaluation.grade(dates, quanta, dt.date(2024, 1, 2), family)
    assert (found["passed"], found["reason"]) == (False, reason)


# --------------------------------------------------------------------------
# The job, end to end
# --------------------------------------------------------------------------

@pytest.mark.parametrize("period,start", [("1y", START), ("6mo", dt.date(2023, 9, 1))],
                         ids=["holdout-shorter-than-period", "holdout-longer-than-period"])
def test_the_evaluator_grades_a_passing_candidate_once_on_its_holdout(db, market, monkeypatch, period, start):
    user, version_id, view, validation = candidate(db, monkeypatch, start=start, period=period)
    before = trial_matrix.family_state(db, user.id, FAMILY)

    done = grade_it(db, validation, period=period)
    assert done["status"] == "succeeded", done
    job = db.get(models.ResearchJob, done["job_id"])
    assert job.max_attempts == 1 and job.attempts == 1

    [attempt] = rows(db, models.HoldoutAttempt)
    assert (attempt.job_id, attempt.user_id, attempt.validation_job_id, attempt.strategy_version_id,
            attempt.family_hash, attempt.dataset_family, attempt.holdout_start) == \
        (done["job_id"], user.id, validation, version_id, FAMILY, "EQUITY_EOD", start)
    [row] = rows(db, models.HoldoutEvaluation)
    grade = row.grade
    assert row.job_id == done["job_id"] and row.passed is grade["passed"]
    assert row.policy_id == holdout_evaluation.policy_id(holdout_evaluation.CURRENT_POLICY) == grade["policy"]

    # The grade's returns are the strategy's on the uncut closes, from the
    # evaluation's first session, counted from the holdout start only.
    closes = synthetic_closes()
    cut = closes[closes.index < pd.Timestamp(start)]
    window_start = market_data.period_start((cut.index.max() + pd.Timedelta(days=1)).date(), period)
    first = cut.index[cut.index >= pd.Timestamp(window_start)][0]
    assert grade["window_first_session"] == first.strftime("%Y-%m-%d")
    window = closes[closes.index >= first]
    compiled = strategy_spec.compile_spec(strategy_spec.parse(strategy()))
    returns = compiled.run({s: window[s].to_numpy(dtype=float) for s in compiled.tickers}, CAPITAL,
                           full_returns=True, risk_metrics=True)["returns"]
    held = [(d.strftime("%Y-%m-%d"), q) for d, q in zip(window.index, research_trials.quantise(returns))
            if d >= pd.Timestamp(start)]
    assert grade["returns_sha256"] == research_trials.quantised_returns_id([q for _, q in held])
    expected = holdout_evaluation.grade([d for d, _ in held], [q for _, q in held], start, "EQUITY_EOD")
    assert {key: grade[key] for key in expected} == expected
    assert grade["sessions"] == len(closes[closes.index >= pd.Timestamp(start)]) >= 63

    # The manifest is signed and names the grade by hash, never its figures.
    manifest = json.loads(bytes(row.canonical))
    assert holdout_evaluation.signature_valid(db, row)
    assert (manifest["holdout_job_id"], manifest["validation_job_id"], manifest["passed"],
            manifest["holdout_start"], manifest["grade"]["sha256"]) == \
        (done["job_id"], validation, row.passed, start.isoformat(), holdout_evaluation.grade_sha256(grade))
    assert "sharpe" not in json.dumps(manifest)

    # The job's result and audit event carry ids and pass/fail only.
    assert set(done["result"]) == SUMMARY_KEYS and done["result"]["passed"] is row.passed
    [audit] = db.execute(select(models.AuditLog).where(models.AuditLog.action == "HOLDOUT_EVALUATION")).scalars()
    audit_text = json.dumps({c.name: getattr(audit, c.name) for c in models.AuditLog.__table__.columns},
                            default=str)
    for figure in ("sharpe", "sessions", "returns", "first_session"):
        assert figure not in audit_text and figure not in json.dumps(done["result"])

    # The attempt counts as a trial of the family; completeness does not change.
    after = trial_matrix.family_state(db, user.id, FAMILY)
    assert after["trials"] == before["trials"] + 1 and after["holdout_queries"] == 1
    assert after["research_trials"] == before["research_trials"]
    assert after["every_trial_evaluated"] == before["every_trial_evaluated"]

    # The owner sees whether it passed, and nothing more.
    [mine] = main.list_holdout_evaluations(limit=20, user=user, db=db)
    assert (mine["job_id"], mine["graded"], mine["passed"]) == (done["job_id"], True, row.passed)
    assert not {"grade", "sharpe_annual", "sessions"} & set(mine)


def test_an_attempt_is_spent_once_per_holdout(db, market, monkeypatch):
    user, version_id, _, validation = candidate(db, monkeypatch)
    assert grade_it(db, validation)["status"] == "succeeded"
    with pytest.raises(holdout_evaluation.HoldoutEvaluationError, match="already been graded"):
        holdout_evaluation.queue(db, validation, "1y", CAPITAL)
    # A job queued around the check is refused by the evaluator, before it
    # reads any data: one attempt, one grade.
    job = research_jobs.enqueue(db, user.id, "final_evaluation",
                                {"validation": validation, "period": "1y", "initial_capital": CAPITAL})
    done = run(db, job.id)
    assert (done["status"], done["error"]["status_code"]) == ("failed", 409), done
    assert "already been graded" in done["error"]["detail"]
    assert len(rows(db, models.HoldoutAttempt)) == len(rows(db, models.HoldoutEvaluation)) == 1
    # Moving the holdout later releases the old one, but this verdict was
    # reached under the old start: it must be validated again first.
    dataset_holdouts.set_start(db, "EQUITY_EOD", dt.date(2024, 2, 1), set_by="ops", reason="roll forward")
    with pytest.raises(holdout_evaluation.HoldoutEvaluationError, match="another holdout start"):
        holdout_evaluation.queue(db, validation, "1y", CAPITAL)


def test_only_a_passing_verdict_with_the_evaluations_period_and_capital_is_queued(db, market, monkeypatch):
    user, _, view, failing = candidate(db, monkeypatch, passing=False)
    with pytest.raises(holdout_evaluation.HoldoutEvaluationError, match="did not pass"):
        holdout_evaluation.queue(db, failing, "1y", CAPITAL)
    with pytest.raises(holdout_evaluation.HoldoutEvaluationError, match="Validation not found"):
        holdout_evaluation.queue(db, "no-such-job", "1y", CAPITAL)
    force_pass(monkeypatch)
    passing = validate(db, user, view["job_id"])["job_id"]
    with pytest.raises(holdout_evaluation.HoldoutEvaluationError, match="period and initial capital"):
        holdout_evaluation.queue(db, passing, "2y", CAPITAL)
    with pytest.raises(holdout_evaluation.HoldoutEvaluationError, match="period and initial capital"):
        holdout_evaluation.queue(db, passing, "1y", 50_000.0)
    assert rows(db, models.HoldoutAttempt) == []


def test_a_verdict_reached_with_no_holdout_is_never_graded(db, market, monkeypatch):
    """Its evaluation saw every session, the later ones included."""
    _, _, _, validation = candidate(db, monkeypatch, start=None)
    with pytest.raises(holdout_evaluation.HoldoutEvaluationError, match="no holdout"):
        holdout_evaluation.queue(db, validation, "1y", CAPITAL)
    dataset_holdouts.set_start(db, "EQUITY_EOD", START, set_by="ops", reason="set after the verdict")
    with pytest.raises(holdout_evaluation.HoldoutEvaluationError, match="another holdout start"):
        holdout_evaluation.queue(db, validation, "1y", CAPITAL)


def test_no_ordinary_worker_claims_a_holdout_evaluation_and_its_view_counts_only_evaluators(
        db, market, monkeypatch):
    _, _, _, validation = candidate(db, monkeypatch)
    job = holdout_evaluation.queue(db, validation, "1y", CAPITAL)
    for role in ("api", "research-worker"):
        assert evaluator(db, role=role).run_once() is False
    db.expire_all()
    queued = db.get(models.ResearchJob, job.id)
    assert queued.status == "queued"
    assert rows(db, models.HoldoutAttempt) == []
    # Those two workers heartbeated, but neither can run it.
    assert research_jobs.view(db, queued)["workers_online"] == 0
    research_jobs.heartbeat_worker(db, "evaluator-1", "host", 1, None, research_jobs.EVALUATOR_ROLE)
    db.expire_all()
    assert research_jobs.view(db, db.get(models.ResearchJob, job.id))["workers_online"] == 1


def test_a_stat_test_counts_the_familys_holdout_attempts_as_trials(db, market, monkeypatch):
    user, _, _, validation = candidate(db, monkeypatch)
    before = research_trials.count(db, user.id, FAMILY)
    assert grade_it(db, validation)["status"] == "succeeded"
    response = main.statistical_tests(schemas.StatTestRequest(returns=[0.01, -0.02, 0.003, 0.004, -0.001, 0.002],
                                                              trial_family=FAMILY), user=user, db=db)
    queued = db.get(models.ResearchJob, json.loads(response.body)["job_id"])
    assert queued.params_json["server_counted"] == before + 1 == research_trials.count(db, user.id, FAMILY) + 1


class _Altered(InlineJobProcess):
    """Hands the job process an evaluation hash of another run."""

    def run(self, target, params, timeout, on_tick, tick_seconds=1.0, *, holdouts=None, prepared=None):
        prepared = {**prepared, "evaluation": {**prepared["evaluation"], "returns_sha256": "qs-r-sha256-v1:00"}}
        return super().run(target, params, timeout, on_tick, tick_seconds, holdouts=holdouts, prepared=prepared)


def test_an_evaluation_that_does_not_reproduce_under_the_cut_is_refused_and_the_attempt_spent(
        db, market, monkeypatch):
    _, _, _, validation = candidate(db, monkeypatch)
    done = grade_it(db, validation, job_process=_Altered())
    assert (done["status"], done["error"]["status_code"]) == ("failed", 409), done
    assert "does not reproduce" in done["error"]["detail"]
    assert len(rows(db, models.HoldoutAttempt)) == 1 and rows(db, models.HoldoutEvaluation) == []


def test_a_failure_of_the_holdout_run_says_nothing_of_the_holdout_and_spends_the_attempt(db, market, monkeypatch):
    _, _, _, validation = candidate(db, monkeypatch)
    from backend.services import historical_data
    real = historical_data.aligned_panel

    def leaky(raw, tickers, min_rows=1):
        if min_rows == 1:  # the uncut run (the cut run asks for STRATEGY_MIN_SESSIONS)
            raise ValueError("holdout Sharpe 1.2345 over 84 sessions")
        return real(raw, tickers, min_rows)

    monkeypatch.setattr(historical_data, "aligned_panel", leaky)
    done = grade_it(db, validation)
    assert done["status"] == "failed"
    assert done["error"] == {"status_code": 422, "detail": holdout_evaluation.HOLDOUT_RUN_FAILED}
    assert len(rows(db, models.HoldoutAttempt)) == 1 and rows(db, models.HoldoutEvaluation) == []


class _Shutdown(InlineJobProcess):
    def run(self, *args, **kwargs):
        return ("shutdown",)


def test_the_evaluator_never_hands_a_started_job_back(db, market, monkeypatch):
    _, _, _, validation = candidate(db, monkeypatch)
    done = grade_it(db, validation, job_process=_Shutdown())
    assert done["status"] == "failed"
    assert done["error"] == {"status_code": 500, "detail": holdout_evaluation.SHUTDOWN_DETAIL}
    assert len(rows(db, models.HoldoutAttempt)) == 1


def test_a_lapsed_lease_fails_the_job_rather_than_requeueing_it(db, market, monkeypatch):
    _, _, _, validation = candidate(db, monkeypatch)
    job = holdout_evaluation.queue(db, validation, "1y", CAPITAL)
    db.execute(text("UPDATE research_jobs SET status = 'running', attempts = 1, worker_id = 'gone',"
                    " lease_expires_at = now() - interval '1 minute' WHERE id = :id"), {"id": job.id})
    db.commit()
    assert research_jobs.reap_expired(db) == 1
    db.expire_all()
    reaped = db.get(models.ResearchJob, job.id)
    assert (reaped.status, reaped.error_detail) == ("failed", research_jobs.WORKER_LOST_DETAIL)


def test_the_holdout_run_never_takes_a_snapshot():
    with pytest.raises(market_data.MarketDataError):
        research_tasks._NoSnapshots().snapshot(None)
    spec = strategy()
    prepared = {"strategy_version_id": strategy_spec.version_id(strategy_spec.parse(spec)), "spec": spec,
                "evaluation": {"job_id": "e", "returns_sha256": "x", "dataset_ids": []}, "period": "1y",
                "initial_capital": CAPITAL, "dataset_family": "EQUITY_EOD", "holdout_start": "2024-01-02"}
    with pytest.raises(TaskError) as refused:
        research_tasks.final_evaluation({"validation": "v", "period": "1y", "initial_capital": CAPITAL}, prepared)
    assert refused.value.status_code == 500


def test_the_holdout_tables_are_append_only(db, market, monkeypatch):
    _, _, _, validation = candidate(db, monkeypatch)
    assert grade_it(db, validation)["status"] == "succeeded"
    for statement in ("UPDATE holdout_attempts SET holdout_start = holdout_start", "DELETE FROM holdout_attempts",
                      "TRUNCATE holdout_attempts CASCADE", "UPDATE holdout_evaluations SET passed = NOT passed",
                      "DELETE FROM holdout_evaluations", "TRUNCATE holdout_evaluations"):
        with pytest.raises(DBAPIError, match="refused"):
            db.execute(text(statement))
        db.rollback()
    # A grade whose passed differs from its grade's is refused by the CHECKs.
    [row] = rows(db, models.HoldoutEvaluation)
    with pytest.raises(DBAPIError, match="ck_holdout_evaluations"):
        db.execute(text("INSERT INTO holdout_evaluations SELECT job_id, policy_id, policy_canonical,"
                        " NOT passed, grade, manifest_id || 'x', canonical, signature, signing_key_id, created_at"
                        " FROM holdout_evaluations WHERE job_id = :id"), {"id": row.job_id})
    db.rollback()
