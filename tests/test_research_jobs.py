"""Research job queue: research runs in a worker process, never in the API.

Covers the API contract (202 + job to poll), per-user limits and ownership,
atomic claiming, lease fencing, the reaper, and the real child job process
(timeouts, crashes, cancellation, shutdown, recycling).
"""
import enum
import json
import os
import signal
import subprocess
import sys
import threading
import time

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from backend import main, models, schemas, worker
from backend.services import (backtest_service, research_jobs, research_tasks, research_trials,
                              security_service)


# ── fixtures and helpers ──────────────────────────────────────────────────────

@pytest.fixture
def Session(make_engine):
    return sessionmaker(bind=make_engine())


@pytest.fixture
def db(Session):
    session = Session()
    yield session
    session.close()


def make_user(db, email="researcher@example.com", role="user"):
    user = models.User(email=email, password_hash="x", role=role)
    db.add(user)
    db.commit()
    return user


def body(response):
    return json.loads(response.body)


class InlineJobProcess:
    """Runs tasks in this process (so monkeypatches apply); no time limit."""

    def run(self, target, params, timeout, on_tick, tick_seconds=1.0):
        return worker._run_target(target, params)

    def stop(self):
        pass


def make_worker(Session, job_process=None, **kwargs):
    kwargs.setdefault("lease_seconds", 60)
    kwargs.setdefault("timeout_seconds", 60)
    kwargs.setdefault("poll_seconds", 0.1)
    return worker.Worker(job_process=job_process or InlineJobProcess(), session_factory=Session, **kwargs)


def state(Session, job_id):
    s = Session()
    try:
        job = s.get(models.ResearchJob, job_id)
        return {"status": job.status, "attempts": job.attempts, "worker_id": job.worker_id,
                "error_status": job.error_status, "error_detail": job.error_detail,
                "result": job.result_json, "cancel_requested": job.cancel_requested}
    finally:
        s.close()


def wait_for(predicate, timeout=90.0, interval=0.1):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def register_kind(monkeypatch, name, function):
    monkeypatch.setitem(research_jobs.KINDS, name, research_jobs.JobKind(
        dict, f"research_job_helpers:{function}", "TEST_JOB"))


def yf_frame(series: dict) -> pd.DataFrame:
    cols = {}
    for ticker, s in series.items():
        for field in ("Open", "High", "Low", "Close"):
            cols[(field, ticker)] = s
        cols[("Volume", ticker)] = s * 0 + 2e6
    frame = pd.DataFrame(cols)
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)
    return frame


def random_walk(seed, n):
    return 100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0.0003, 0.015, n)))


# ── 1. API contract ─────────────────────────────────────────────────────────────

ENDPOINTS = [
    (main.run_backtest, lambda: schemas.BacktestRequest(asset="AAPL"), "ma_backtest"),
    (main.advanced_backtest, schemas.AdvancedBacktestRequest, "advanced_backtest"),
    (main.walk_forward_validation, schemas.WalkForwardRequest, "walk_forward"),
    (main.event_backtest_endpoint, schemas.EventBacktestRequest, "event_backtest"),
    (main.alpha_research_endpoint, schemas.AlphaResearchRequest, "alpha"),
    (main.factor_model_endpoint, schemas.FactorModelRequest, "factor_model"),
    (main.correlation_endpoint, schemas.CorrelationRequest, "correlation"),
    (main.portfolio_optimize_endpoint, schemas.PortfolioOptRequest, "optimize"),
    (main.regime_detection_endpoint, schemas.RegimeDetectionRequest, "regime"),
    (main.neutral_strategy_endpoint, schemas.NeutralStrategyRequest, "neutral_strategy"),
    (main.pairs_trading_endpoint, schemas.PairsTradingRequest, "pairs_trading"),
    (main.latency_benchmark_endpoint, schemas.LatencyBenchmarkRequest, "latency_benchmark"),
    (main.research_report_endpoint, schemas.ReportRequest, "report"),
]


class TestApiContract:
    @pytest.mark.parametrize("endpoint,request_factory,kind", ENDPOINTS,
                             ids=[kind for _, _, kind in ENDPOINTS])
    def test_research_endpoint_queues_a_job_and_computes_nothing(self, db, monkeypatch,
                                                                 endpoint, request_factory, kind):
        target = research_jobs.KINDS[kind].target.split(":")[1]
        monkeypatch.setattr(research_tasks, target,
                            lambda _params: pytest.fail("the API process must not compute research"))
        user = make_user(db)
        req = request_factory()
        response = endpoint(req, user=user, db=db)
        payload = body(response)
        assert response.status_code == 202
        assert response.headers["location"] == f"/api/research/jobs/{payload['job_id']}"
        assert payload["kind"] == kind and payload["status"] == "queued"
        assert payload["queue_position"] == 1 and payload["workers_online"] == 0
        job = db.get(models.ResearchJob, payload["job_id"])
        assert job.user_id == user.id
        assert job.params_json == json.loads(json.dumps(req.model_dump()))

    def test_every_kind_resolves_to_its_task(self):
        for kind in research_jobs.KINDS.values():
            assert callable(worker._resolve(kind.target))

    def test_invalid_pair_is_rejected_before_queueing(self, db):
        user = make_user(db)
        with pytest.raises(HTTPException) as exc:
            main.pairs_trading_endpoint(schemas.PairsTradingRequest(asset_y="GS", asset_x="GS"),
                                        user=user, db=db)
        assert exc.value.status_code == 422
        assert db.execute(select(models.ResearchJob)).first() is None

    def test_active_jobs_are_limited_per_user(self, db, monkeypatch):
        monkeypatch.setattr(research_jobs, "RESEARCH_MAX_ACTIVE_JOBS_PER_USER", 3)
        alice, bob = make_user(db, "alice@example.com"), make_user(db, "bob@example.com")
        req = schemas.CorrelationRequest()
        ids = [body(main.correlation_endpoint(req, user=alice, db=db))["job_id"] for _ in range(3)]
        with pytest.raises(HTTPException) as exc:
            main.correlation_endpoint(req, user=alice, db=db)
        assert exc.value.status_code == 429 and "limit 3" in exc.value.detail
        assert main.correlation_endpoint(req, user=bob, db=db).status_code == 202
        assert main.cancel_research_job(ids[0], user=alice, db=db)["status"] == "cancelled"
        assert main.correlation_endpoint(req, user=alice, db=db).status_code == 202

    def test_limit_holds_under_concurrent_submissions(self, Session, monkeypatch):
        monkeypatch.setattr(research_jobs, "RESEARCH_MAX_ACTIVE_JOBS_PER_USER", 3)
        setup = Session()
        user_id = make_user(setup).id
        setup.close()
        outcomes = []

        def submit():
            s = Session()
            try:
                research_jobs.enqueue(s, user_id, "correlation", schemas.CorrelationRequest().model_dump())
                outcomes.append("queued")
            except research_jobs.QueueFull:
                outcomes.append("full")
            finally:
                s.close()

        threads = [threading.Thread(target=submit) for _ in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert outcomes.count("queued") == 3 and outcomes.count("full") == 9

    def test_jobs_are_private_to_their_owner(self, db):
        alice, bob = make_user(db, "alice@example.com"), make_user(db, "bob@example.com")
        job_id = body(main.correlation_endpoint(schemas.CorrelationRequest(), user=alice, db=db))["job_id"]
        for call in (main.get_research_job, main.cancel_research_job):
            with pytest.raises(HTTPException) as exc:
                call(job_id, user=bob, db=db)
            assert exc.value.status_code == 404
        assert main.list_research_jobs(limit=20, user=bob, db=db) == []
        listed = main.list_research_jobs(limit=20, user=alice, db=db)
        assert [j["job_id"] for j in listed] == [job_id] and "result" not in listed[0]
        assert db.get(models.ResearchJob, job_id).status == "queued"

    def test_queue_status_is_operator_only(self, db):
        user, operator = make_user(db, "u@example.com"), make_user(db, "ops@example.com", role="admin")
        main.correlation_endpoint(schemas.CorrelationRequest(), user=user, db=db)
        with pytest.raises(HTTPException) as exc:
            main.research_queue_status(user=user, db=db)
        assert exc.value.status_code == 403
        stats = main.research_queue_status(user=operator, db=db)
        assert stats["queued"] == 1 and stats["running"] == 0 and stats["workers_online"] == 0
        assert stats["oldest_queued_seconds"] >= 0


# ── 2. Worker execution (in-process job runner) ────────────────────────────────

class TestWorkerExecution:
    def test_dashboard_backtest_job_end_to_end(self, Session, monkeypatch):
        returns = list(np.random.default_rng(7).normal(0.001, 0.01, 60))
        monkeypatch.setattr(backtest_service, "run_moving_average_backtest", lambda *a: {
            "initial_capital": 100_000.0, "final_capital": 101_500.0, "total_return": 0.015,
            "sharpe_ratio": np.float64(1.25), "max_drawdown": -0.04, "win_rate": 0.5,
            "total_trades": np.int64(6), "daily_returns_net": np.array(returns),
        })
        db = Session()
        user = make_user(db)
        job_id = body(main.run_backtest(schemas.BacktestRequest(asset="aapl"), user=user, db=db))["job_id"]

        assert make_worker(Session).run_once() is True

        db.expire_all()
        view = main.get_research_job(job_id, user=user, db=db)
        assert view["status"] == "succeeded" and view["attempts"] == 1
        result = view["result"]
        row = db.execute(select(models.Backtest)).scalars().one()
        assert result["id"] == row.id and row.user_id == user.id
        assert result["sharpe_ratio"] == 1.25 and result["total_trades"] == 6
        assert result["trial_family"] == research_trials.family_hash("ma_crossover", ["AAPL"])
        assert result["trials_in_family"] == 1
        assert main.list_backtests(user=user, db=db)[0]["id"] == row.id
        # The stored row feeds the statistical tests exactly as before.
        stat = main.statistical_tests(schemas.StatTestRequest(backtest_id=row.id), user, db)
        assert stat["trial_count"]["server_counted"] == 1
        audit = db.execute(select(models.AuditLog).where(
            models.AuditLog.action == "BACKTEST_COMPLETED")).scalars().one()
        assert audit.resource_id == row.id and audit.metadata_json["job_id"] == job_id
        assert security_service.audit_chain_status(db)["valid"]
        db.close()

    def test_advanced_backtest_runs_the_real_engine_as_a_job(self, Session, monkeypatch):
        import yfinance
        days = pd.bdate_range("2023-01-02", periods=500)
        frame = yf_frame({t: pd.Series(random_walk(i, len(days)), days)
                          for i, t in enumerate(["AAPL", "MSFT", "SPY"])})
        monkeypatch.setattr(yfinance, "download", lambda *a, **k: frame)
        db = Session()
        user = make_user(db)
        req = schemas.AdvancedBacktestRequest(assets=["AAPL", "MSFT"])
        job_id = body(main.advanced_backtest(req, user=user, db=db))["job_id"]

        make_worker(Session).run_once()

        db.expire_all()
        view = main.get_research_job(job_id, user=user, db=db)
        assert view["status"] == "succeeded", view
        result = view["result"]
        assert result["assets_traded"] == ["AAPL", "MSFT"] and result["execution_delay_bars"] == 1
        assert result["trials_in_family"] == 1
        json.dumps(result, allow_nan=False)  # what the API will serialise
        assert db.execute(select(models.AuditLog).where(
            models.AuditLog.action == "ADVANCED_BACKTEST")).scalars().one().metadata_json["job_id"] == job_id
        db.close()

    def test_walk_forward_job_records_every_grid_config_as_a_trial(self, Session, monkeypatch):
        import yfinance
        days = pd.bdate_range("2019-01-01", periods=1300)
        frame = yf_frame({"AAPL": pd.Series(random_walk(3, len(days)), days)})
        monkeypatch.setattr(yfinance, "download", lambda *a, **k: frame)
        db = Session()
        user = make_user(db)
        req = schemas.WalkForwardRequest(assets=["AAPL"], total_years=5, train_years=2, test_years=1,
                                         optimize_parameters=True)
        job_id = body(main.walk_forward_validation(req, user=user, db=db))["job_id"]

        make_worker(Session).run_once()

        db.expire_all()
        result = main.get_research_job(job_id, user=user, db=db)["result"]
        assert result["n_trials"] == len(result["parameter_grid"]) == 25
        assert result["trials_in_family"] == 25
        db.close()

    def test_task_error_fails_the_job_with_its_status_and_message(self, Session, monkeypatch):
        import yfinance
        monkeypatch.setattr(yfinance, "download", lambda *a, **k: pd.DataFrame())
        db = Session()
        user = make_user(db)
        job_id = body(main.correlation_endpoint(schemas.CorrelationRequest(), user=user, db=db))["job_id"]

        make_worker(Session).run_once()

        db.expire_all()
        view = main.get_research_job(job_id, user=user, db=db)
        assert view["status"] == "failed" and "result" not in view
        assert view["error"] == {"status_code": 422, "detail": "Failed to download market data"}
        db.close()

    def test_worker_heartbeat_is_visible_to_waiting_jobs(self, Session):
        db = Session()
        user = make_user(db)
        w = make_worker(Session)
        w.run_once()  # nothing queued; registers the worker
        job_id = body(main.correlation_endpoint(schemas.CorrelationRequest(), user=user, db=db))["job_id"]
        assert main.get_research_job(job_id, user=user, db=db)["workers_online"] == 1
        research_jobs.remove_worker(db, w.id)
        assert main.get_research_job(job_id, user=user, db=db)["workers_online"] == 0
        db.close()

    def test_jobs_run_oldest_first(self, Session, monkeypatch):
        register_kind(monkeypatch, "test_echo", "echo_task")
        db = Session()
        user = make_user(db)
        ids = [research_jobs.enqueue(db, user.id, "test_echo", {"n": n}).id for n in range(3)]
        w = make_worker(Session)
        order = []
        for _ in range(3):
            s = Session()
            job = research_jobs.claim_next(s, w.id, 60)
            order.append(job.id)
            s.close()
        assert order == ids
        db.close()


# ── 3. Claiming, leases and the reaper ──────────────────────────────────────────

class TestLeases:
    def test_each_job_is_claimed_exactly_once_by_concurrent_workers(self, Session):
        setup = Session()
        users = [make_user(setup, f"u{i}@example.com") for i in range(10)]
        expected = set()
        for user in users:
            for _ in range(3):
                expected.add(research_jobs.enqueue(setup, user.id, "correlation", {}).id)
        setup.close()
        claimed, lock = [], threading.Lock()

        def claimer(n):
            s = Session()
            try:
                while True:
                    job = research_jobs.claim_next(s, f"worker-{n}", 60)
                    if job is None:
                        return
                    with lock:
                        claimed.append(job.id)
            finally:
                s.close()

        threads = [threading.Thread(target=claimer, args=(n,)) for n in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sorted(claimed) == sorted(expected)

    def test_a_replaced_worker_cannot_overwrite_the_new_attempt(self, Session):
        db = Session()
        user = make_user(db)
        job_id = research_jobs.enqueue(db, user.id, "correlation", {}).id
        first = research_jobs.claim_next(db, "worker-a", 60)
        # worker-a stops renewing; its lease lapses and the reaper requeues the job.
        db.query(models.ResearchJob).filter_by(id=job_id).update(
            {"lease_expires_at": research_jobs.utcnow() - pd.Timedelta(seconds=1)})
        db.commit()
        assert research_jobs.reap_expired(db) == 1
        assert state(Session, job_id)["status"] == "queued"
        second = research_jobs.claim_next(db, "worker-b", 60)
        assert second.attempts == 2

        assert research_jobs.renew_lease(db, first, "worker-a", 1, 60) == "lost"
        assert research_jobs.complete(db, first, "worker-a", 1, {"stale": True}, {}, None) is False
        assert research_jobs.fail(db, first, "worker-a", 1, 500, "stale") is False
        assert state(Session, job_id)["status"] == "running"

        assert research_jobs.complete(db, second, "worker-b", 2, {"fresh": True}, {}, None) is True
        final = state(Session, job_id)
        assert final["status"] == "succeeded" and final["result"] == {"fresh": True}
        db.close()

    def test_the_lease_is_fenced_by_worker_and_attempt_separately(self, Session):
        db = Session()
        user = make_user(db)
        job_id = research_jobs.enqueue(db, user.id, "correlation", {}).id
        first = research_jobs.claim_next(db, "worker-a", 60)
        # Another worker naming the current attempt does not hold the lease.
        assert research_jobs.renew_lease(db, first, "worker-b", 1, 60) == "lost"
        assert research_jobs.complete(db, first, "worker-b", 1, {"intruder": True}, {}, None) is False
        assert research_jobs.fail(db, first, "worker-b", 1, 500, "intruder") is False
        assert state(Session, job_id)["status"] == "running"

        # The same worker reclaiming the job after its lease lapsed: its old
        # attempt is stale even though the worker id matches.
        db.query(models.ResearchJob).filter_by(id=job_id).update(
            {"lease_expires_at": research_jobs.utcnow() - pd.Timedelta(seconds=1)})
        db.commit()
        assert research_jobs.reap_expired(db) == 1
        second = research_jobs.claim_next(db, "worker-a", 60)
        assert second.attempts == 2
        assert research_jobs.renew_lease(db, first, "worker-a", 1, 60) == "lost"
        assert research_jobs.complete(db, first, "worker-a", 1, {"stale": True}, {}, None) is False
        assert research_jobs.fail(db, first, "worker-a", 1, 500, "stale") is False
        assert state(Session, job_id)["status"] == "running"

        assert research_jobs.complete(db, second, "worker-a", 2, {"fresh": True}, {}, None) is True
        final = state(Session, job_id)
        assert final["status"] == "succeeded" and final["result"] == {"fresh": True}
        db.close()

    def test_reaper_fails_a_job_after_max_attempts_and_honours_cancel(self, Session):
        db = Session()
        user = make_user(db)
        exhausted = research_jobs.enqueue(db, user.id, "correlation", {}).id
        cancelled = research_jobs.enqueue(db, user.id, "correlation", {}).id
        past = research_jobs.utcnow() - pd.Timedelta(seconds=1)
        db.query(models.ResearchJob).filter_by(id=exhausted).update(
            {"status": "running", "worker_id": "w", "attempts": 2, "max_attempts": 2, "lease_expires_at": past})
        db.query(models.ResearchJob).filter_by(id=cancelled).update(
            {"status": "running", "worker_id": "w", "attempts": 1, "cancel_requested": True,
             "lease_expires_at": past})
        db.commit()

        assert research_jobs.reap_expired(db) == 2
        assert state(Session, exhausted)["status"] == "failed"
        assert state(Session, exhausted)["error_detail"] == research_jobs.WORKER_LOST_DETAIL
        assert state(Session, cancelled)["status"] == "cancelled"
        db.close()

    def test_a_live_lease_is_never_reaped(self, Session):
        db = Session()
        user = make_user(db)
        job_id = research_jobs.enqueue(db, user.id, "correlation", {}).id
        research_jobs.claim_next(db, "worker-a", 60)
        assert research_jobs.reap_expired(db) == 0
        assert state(Session, job_id)["status"] == "running"
        db.close()

    def test_cancelling_a_queued_job_is_immediate(self, Session, monkeypatch):
        register_kind(monkeypatch, "test_echo", "echo_task")
        db = Session()
        user = make_user(db)
        job_id = research_jobs.enqueue(db, user.id, "test_echo", {}).id
        assert main.cancel_research_job(job_id, user=user, db=db)["status"] == "cancelled"
        assert make_worker(Session).run_once() is False
        assert state(Session, job_id)["status"] == "cancelled"
        db.close()

    def test_finished_jobs_are_purged_after_retention(self, Session):
        db = Session()
        user = make_user(db)
        old = research_jobs.enqueue(db, user.id, "correlation", {}).id
        recent = research_jobs.enqueue(db, user.id, "correlation", {}).id
        active = research_jobs.enqueue(db, user.id, "correlation", {}).id
        db.query(models.ResearchJob).filter_by(id=old).update(
            {"status": "succeeded", "finished_at": research_jobs.utcnow() - pd.Timedelta(days=8)})
        db.query(models.ResearchJob).filter_by(id=recent).update(
            {"status": "failed", "finished_at": research_jobs.utcnow() - pd.Timedelta(days=1)})
        db.commit()
        assert research_jobs.purge_finished(db, retention_days=7) == 1
        remaining = {j.id for j in db.execute(select(models.ResearchJob)).scalars()}
        assert remaining == {recent, active}
        db.close()


# ── 4. The real child job process ─────────────────────────────────────────────

class TestJobProcess:
    def test_child_is_reused_then_recycled(self, Session, monkeypatch):
        register_kind(monkeypatch, "test_echo", "echo_task")
        db = Session()
        user = make_user(db, role="admin")
        w = make_worker(Session, job_process=worker.JobProcess(recycle_after=2))
        pids = []
        try:
            for n in range(3):
                job_id = research_jobs.enqueue(db, user.id, "test_echo", {"n": n}).id
                assert w.run_once() is True
                result = state(Session, job_id)["result"]
                assert result["echo"] == {"n": n}
                pids.append(result["pid"])
        finally:
            w.job_process.stop()
        assert pids[0] == pids[1] != pids[2]
        db.close()

    def test_timeout_kills_the_job_and_the_next_job_still_runs(self, Session, monkeypatch):
        register_kind(monkeypatch, "test_sleep", "sleep_task")
        register_kind(monkeypatch, "test_echo", "echo_task")
        db = Session()
        user = make_user(db)
        w = make_worker(Session, job_process=worker.JobProcess(), timeout_seconds=3)
        try:
            slow = research_jobs.enqueue(db, user.id, "test_sleep", {"seconds": 120}).id
            started = time.monotonic()
            w.run_once()
            assert time.monotonic() - started < 60
            failed = state(Session, slow)
            assert failed["status"] == "failed" and failed["error_status"] == 504
            assert "3-second time limit" in failed["error_detail"]
            fast = research_jobs.enqueue(db, user.id, "test_echo", {"after": "timeout"}).id
            w.run_once()
            assert state(Session, fast)["status"] == "succeeded"
        finally:
            w.job_process.stop()
        db.close()

    def test_a_crashing_job_fails_alone(self, Session, monkeypatch):
        register_kind(monkeypatch, "test_crash", "crash_task")
        register_kind(monkeypatch, "test_echo", "echo_task")
        db = Session()
        user = make_user(db)
        w = make_worker(Session, job_process=worker.JobProcess())
        try:
            crash = research_jobs.enqueue(db, user.id, "test_crash", {}).id
            w.run_once()
            crashed = state(Session, crash)
            assert crashed["status"] == "failed" and crashed["error_status"] == 500
            assert crashed["error_detail"] == "The research job process stopped unexpectedly."
            after = research_jobs.enqueue(db, user.id, "test_echo", {}).id
            w.run_once()
            assert state(Session, after)["status"] == "succeeded"
        finally:
            w.job_process.stop()
        db.close()

    def test_cancelling_a_running_job_stops_its_process(self, Session, monkeypatch):
        register_kind(monkeypatch, "test_sleep", "sleep_task")
        db = Session()
        user = make_user(db)
        w = make_worker(Session, job_process=worker.JobProcess(), lease_seconds=3)
        job_id = research_jobs.enqueue(db, user.id, "test_sleep", {"seconds": 120}).id
        runner = threading.Thread(target=w.run_once)
        try:
            runner.start()
            assert wait_for(lambda: state(Session, job_id)["status"] == "running")
            assert main.cancel_research_job(job_id, user=user, db=db)["cancel_requested"] is True
            runner.join(60)
            assert not runner.is_alive()
            assert state(Session, job_id)["status"] == "cancelled"
        finally:
            w.stop_event.set()
            runner.join(60)
            w.job_process.stop()
        db.close()

    def test_cancel_is_noticed_between_lease_renewals(self, Session, monkeypatch):
        # With the production lease (60 s, renewed every 20 s) a cancel must
        # still stop the job within seconds, not at the next renewal.
        register_kind(monkeypatch, "test_sleep", "sleep_task")
        db = Session()
        user = make_user(db)
        w = make_worker(Session, job_process=worker.JobProcess(), lease_seconds=60)
        job_id = research_jobs.enqueue(db, user.id, "test_sleep", {"seconds": 120}).id
        runner = threading.Thread(target=w.run_once)
        try:
            runner.start()
            assert wait_for(lambda: state(Session, job_id)["status"] == "running")
            requested = time.monotonic()
            main.cancel_research_job(job_id, user=user, db=db)
            assert wait_for(lambda: state(Session, job_id)["status"] == "cancelled", timeout=30)
            assert time.monotonic() - requested < 5
            runner.join(30)
            assert not runner.is_alive()
        finally:
            w.stop_event.set()
            runner.join(60)
            w.job_process.stop()
        db.close()

    def test_shutdown_hands_the_running_job_back_to_the_queue(self, Session, monkeypatch):
        register_kind(monkeypatch, "test_sleep", "sleep_task")
        db = Session()
        user = make_user(db)
        w = make_worker(Session, job_process=worker.JobProcess(), lease_seconds=3)
        job_id = research_jobs.enqueue(db, user.id, "test_sleep", {"seconds": 120}).id
        runner = threading.Thread(target=w.run_once)
        try:
            runner.start()
            assert wait_for(lambda: state(Session, job_id)["status"] == "running")
            w.stop_event.set()
            runner.join(60)
            assert not runner.is_alive()
            released = state(Session, job_id)
            assert released["status"] == "queued"
            assert released["attempts"] == 0 and released["worker_id"] is None
        finally:
            w.stop_event.set()
            runner.join(60)
            w.job_process.stop()
        db.close()


# ── 5. Result encoding ───────────────────────────────────────────────────────────

class Colour(enum.Enum):
    RED = "red"


def test_results_are_encoded_like_the_api_did_with_nan_as_null():
    encoded = research_jobs.json_safe({
        "f": np.float64(1.5), "i": np.int64(3), "b": np.bool_(True), "arr": np.array([1.0, np.nan]),
        "nan": float("nan"), "inf": np.float64(np.inf), "enum": Colour.RED,
        "ts": pd.Timestamp("2026-01-02"), "nested": [{"x": np.float32(0.5)}],
    })
    assert encoded == {"f": 1.5, "i": 3, "b": True, "arr": [1.0, None], "nan": None, "inf": None,
                       "enum": "red", "ts": "2026-01-02T00:00:00", "nested": [{"x": 0.5}]}
    json.dumps(encoded, allow_nan=False)


# ── 6. Embedded worker lifecycle ─────────────────────────────────────────────────

def test_embedded_worker_stops_when_the_api_process_dies(tmp_path, make_engine):
    """Killed outright (no lifespan shutdown: a hard kill, or uvicorn --reload
    on Windows), the API process must not leave its worker running as an orphan."""
    engine = make_engine(migrated=True)
    db_url = engine.url.render_as_string(hide_password=False)
    Session = sessionmaker(bind=engine)
    pid_file = tmp_path / "worker.pid"
    # Stands in for the API process: starts the worker exactly as main does.
    api = subprocess.Popen(
        [sys.executable, "-c",
         "import pathlib, time\nfrom backend import main\n"
         "proc = main._start_embedded_worker()\n"
         f"pathlib.Path({str(pid_file)!r}).write_text(str(proc.pid))\n"
         "time.sleep(600)\n"],
        cwd=str(main.BASE_DIR), env={**os.environ, "DATABASE_URL": db_url})
    worker_pid = None

    def registered_workers():
        with Session() as s:
            return s.execute(select(models.ResearchWorker)).scalars().all()

    try:
        assert wait_for(pid_file.exists, timeout=120)
        worker_pid = int(pid_file.read_text())
        assert wait_for(lambda: len(registered_workers()) == 1, timeout=120)
        api.kill()
        api.wait(30)
        # The worker noticed, shut down cleanly and deregistered itself.
        assert wait_for(lambda: registered_workers() == [], timeout=30)
        worker_pid = None
    finally:
        if api.poll() is None:
            api.kill()
        if worker_pid is not None:
            os.kill(worker_pid, signal.SIGTERM)  # do not leak it from a failing run
        engine.dispose()
