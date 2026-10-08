"""Experiment run/validate/replay and the statistical tests are research jobs.

The API checks the request, queues it and answers 202 with the job to poll;
a worker computes it and records the outcome (signed results, validation
gates, audit event) in the same transaction that completes the job.
"""
import json
import math

import numpy as np
import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from backend import main, models, schemas, worker
from backend.services import experiment_registry as er
from backend.services import research_jobs, research_trials, security_service, stat_tests
from research_job_helpers import result_of, run_queued


@pytest.fixture
def db(make_engine):
    session = sessionmaker(bind=make_engine())()
    yield session
    session.close()


def make_user(db, email="researcher@example.com", role="user"):
    user = models.User(email=email, password_hash="x", role=role)
    db.add(user)
    db.commit()
    return user


@pytest.fixture
def owner(db):
    return make_user(db)


BARS = [{"timestamp": 1_700_000_000 + i * 86400, "open": 100 + 0.5 * i, "high": 102 + 0.5 * i,
         "low": 99 + 0.5 * i, "close": 100.5 + 0.5 * i, "volume": 200_000} for i in range(6)]
PARAMS = {"obi_threshold": 0.2, "events_per_bar": 120, "warmup_events": 30, "latency_preset": "cloud"}
SEED = 11


def create(db, user, strategy_id="obi_momentum", dataset=BARS, parameters=PARAMS):
    return main.experiment_create({"strategy_id": strategy_id, "strategy_version": "1.0",
                                   "dataset_id": "synthetic-bars", "dataset": dataset,
                                   "parameters": parameters, "random_seed": SEED}, user, db)["experiment_id"]


def queued(response, kind):
    payload = json.loads(response.body)
    assert response.status_code == 202
    assert response.headers["location"] == f"/api/research/jobs/{payload['job_id']}"
    assert payload["kind"] == kind and payload["status"] == "queued"
    return payload


def jobs(db):
    return db.execute(select(models.ResearchJob)).scalars().all()


def audits(db, action):
    return db.execute(select(models.AuditLog).where(models.AuditLog.action == action)).scalars().all()


def never_in_the_api(*_a, **_k):
    pytest.fail("the API process must not compute research")


def quality_gates_pass(monkeypatch):
    """Hold the market-outcome gates passing; integrity gates stay real."""
    real = er.validate_for_deployment

    def passing(exp):
        report = real(exp)
        for gate in report["gates"].values():
            gate["passed"] = True
        return report
    monkeypatch.setattr(er, "validate_for_deployment", passing)


# ── 1. The API queues and computes nothing ─────────────────────────────────────

class TestApiQueues:
    def test_run_is_queued_not_computed(self, db, owner, monkeypatch):
        exp_id = create(db, owner)
        monkeypatch.setattr(er, "run_strategy", never_in_the_api)
        payload = queued(main.experiment_run(exp_id, owner, db), "experiment_run")
        job = db.get(models.ResearchJob, payload["job_id"])
        assert job.user_id == owner.id and job.params_json["experiment_id"] == exp_id
        assert db.get(models.ResearchExperiment, exp_id).status == "CREATED"

    def test_validate_and_replay_are_queued_not_computed(self, db, owner, monkeypatch):
        exp_id = create(db, owner)
        result_of(db, main.experiment_run(exp_id, owner, db))
        monkeypatch.setattr(er, "run_strategy", never_in_the_api)
        monkeypatch.setattr(er, "validate_for_deployment", never_in_the_api)
        queued(main.experiment_validate(exp_id, owner, db), "experiment_validate")
        queued(main.experiment_replay(exp_id, {}, owner, db), "experiment_replay")
        row = db.get(models.ResearchExperiment, exp_id)
        assert row.status == "COMPLETED" and not row.validation_gates_json

    def test_stat_test_is_queued_not_computed(self, db, owner, monkeypatch):
        monkeypatch.setattr(stat_tests, "run_full_stat_tests", never_in_the_api)
        returns = list(np.random.default_rng(1).normal(0.001, 0.01, 300))
        payload = queued(main.statistical_tests(schemas.StatTestRequest(returns=returns), owner, db), "stat_test")
        params = db.get(models.ResearchJob, payload["job_id"]).params_json
        assert params["returns"] == returns and params["n_trials"] == 1

    def test_every_new_kind_resolves_to_its_task_and_finalizer(self):
        for name in ("experiment_run", "experiment_validate", "experiment_replay", "stat_test"):
            kind = research_jobs.KINDS[name]
            assert callable(worker._resolve(kind.target))
            if kind.finalize:
                assert callable(worker._resolve(kind.finalize))


# ── 2. Experiment jobs on a worker ─────────────────────────────────────────────

class TestRunJob:
    def test_records_the_same_signed_result_the_synchronous_run_did(self, db, owner):
        exp_id = create(db, owner)
        response = main.experiment_run(exp_id, owner, db)
        job_id = json.loads(response.body)["job_id"]
        result = result_of(db, response)

        expected = er.run_strategy("obi_momentum", BARS, PARAMS, SEED)
        assert result["results"] == expected
        assert result["result_hash"] == er.hash_results(expected)
        assert result["status"] == "COMPLETED" and result["has_signature"] is True
        assert result["experiment_id"] == exp_id
        manifest = main.experiment_manifest(exp_id, owner, db)
        assert manifest["signature_valid"] is True and manifest["consistent_with_record"] is True
        assert manifest["result_hash"] == result["result_hash"]

        [audit] = audits(db, "EXPERIMENT_RUN")
        assert (audit.resource_type, audit.resource_id) == ("experiment", exp_id)
        assert audit.metadata_json == {"result_hash": result["result_hash"], "job_id": job_id}
        assert security_service.audit_chain_status(db)["valid"]

    def test_runs_in_the_real_job_process(self, db, owner):
        exp_id = create(db, owner)
        process = worker.JobProcess()
        try:
            result = result_of(db, main.experiment_run(exp_id, owner, db), job_process=process)
        finally:
            process.stop()
        assert result["result_hash"] == er.hash_results(er.run_strategy("obi_momentum", BARS, PARAMS, SEED))

    def test_resubmitting_while_queued_returns_the_same_job(self, db, owner):
        exp_id = create(db, owner)
        first = queued(main.experiment_run(exp_id, owner, db), "experiment_run")
        second = queued(main.experiment_run(exp_id, owner, db), "experiment_run")
        assert first["job_id"] == second["job_id"] and len(jobs(db)) == 1

    def test_a_completed_experiment_cannot_run_again(self, db, owner):
        exp_id = create(db, owner)
        result_of(db, main.experiment_run(exp_id, owner, db))
        with pytest.raises(HTTPException) as exc:
            main.experiment_run(exp_id, owner, db)
        assert exc.value.status_code == 422 and "already has results" in exc.value.detail

    @pytest.mark.parametrize("strategy_id,dataset,parameters", [
        ("external-model", BARS, {}),                                 # not executable
        ("obi_momentum", BARS, {**PARAMS, "leverage": 10}),           # unknown parameter
        ("obi_momentum", BARS, {**PARAMS, "obi_threshold": 1.5}),     # out of range
        ("obi_momentum", BARS * 100, PARAMS),                         # 600 bars > 500
        ("obi_momentum", [{"close": 1}], PARAMS),                     # malformed bars
    ], ids=["not-executable", "unknown-param", "bad-threshold", "too-many-bars", "malformed-bars"])
    def test_bad_inputs_are_rejected_before_queueing(self, db, owner, strategy_id, dataset, parameters):
        exp_id = create(db, owner, strategy_id=strategy_id, dataset=dataset, parameters=parameters)
        with pytest.raises(HTTPException) as exc:
            main.experiment_run(exp_id, owner, db)
        assert exc.value.status_code == 422
        assert jobs(db) == []

    def test_executed_inputs_must_be_the_recorded_ones(self, db, owner):
        # The worker refuses to record a result computed from inputs other
        # than the experiment's recorded ones, and the job fails rather than
        # hanging as "running".
        exp_id = create(db, owner)
        response = main.experiment_run(exp_id, owner, db)
        job = db.get(models.ResearchJob, json.loads(response.body)["job_id"])
        job.params_json = {**job.params_json, "random_seed": SEED + 1}
        db.commit()
        view = run_queued(db, response)
        assert view["status"] == "failed" and view["error"]["status_code"] == 409
        row = db.get(models.ResearchExperiment, exp_id)
        assert row.status == "CREATED" and not row.result_hash and not row.manifest_signature
        assert audits(db, "EXPERIMENT_RUN") == []

    def test_results_recorded_while_the_run_waited_are_kept(self, db, owner):
        exp_id = create(db, owner)
        response = main.experiment_run(exp_id, owner, db)
        recorded = er.PersistentExperimentRegistry(db, owner.id).complete(exp_id, {"recorded": "elsewhere"})
        view = run_queued(db, response)
        assert view["status"] == "failed" and view["error"]["status_code"] == 422
        row = db.get(models.ResearchExperiment, exp_id)
        assert row.result_hash == recorded.result_hash and row.results_json == {"recorded": "elsewhere"}
        assert row.manifest_signature == recorded.manifest_signature
        assert audits(db, "EXPERIMENT_RUN") == []

    def test_unknown_or_foreign_experiments_are_404(self, db, owner):
        exp_id = create(db, owner)
        stranger = make_user(db, "stranger@example.com")
        for call in (lambda u: main.experiment_run(exp_id, u, db),
                     lambda u: main.experiment_validate(exp_id, u, db),
                     lambda u: main.experiment_replay(exp_id, {}, u, db),
                     lambda u: main.experiment_run("QS-NOPE", u, db)):
            with pytest.raises(HTTPException) as exc:
                call(stranger)
            assert exc.value.status_code == 404
        assert jobs(db) == []


class TestValidateJob:
    def test_evaluates_every_gate_and_records_the_report(self, db, owner):
        exp_id = create(db, owner)
        result_of(db, main.experiment_run(exp_id, owner, db))
        response = main.experiment_validate(exp_id, owner, db)
        job_id = json.loads(response.body)["job_id"]
        report = result_of(db, response)
        for gate in ("signature_valid", "manifest_consistent", "inputs_unchanged", "result_reproducible"):
            assert report["gates"][gate]["passed"] is True, gate
        assert {"net_pnl", "fill_ratio", "implementation_shortfall", "risk_limits"} <= set(report["gates"])
        row = db.get(models.ResearchExperiment, exp_id)
        assert row.validation_gates_json == report
        assert row.status == ("VALIDATED" if report["all_gates_passed"] else "COMPLETED")
        [audit] = audits(db, "EXPERIMENT_VALIDATED")
        assert audit.resource_id == exp_id
        assert audit.metadata_json == {"all_gates_passed": report["all_gates_passed"], "job_id": job_id}

    def test_passing_gates_validate_the_experiment(self, db, owner, monkeypatch):
        exp_id = create(db, owner)
        result_of(db, main.experiment_run(exp_id, owner, db))
        quality_gates_pass(monkeypatch)
        assert result_of(db, main.experiment_validate(exp_id, owner, db))["all_gates_passed"] is True
        assert db.get(models.ResearchExperiment, exp_id).status == "VALIDATED"

    def test_only_completed_experiments_can_be_validated(self, db, owner):
        exp_id = create(db, owner)
        with pytest.raises(HTTPException) as exc:
            main.experiment_validate(exp_id, owner, db)
        assert exc.value.status_code == 409
        assert jobs(db) == []

    def test_a_non_executable_experiment_is_validated_without_execution(self, db, owner, monkeypatch):
        exp_id = create(db, owner, strategy_id="external-model", parameters={})
        er.PersistentExperimentRegistry(db, owner.id).complete(exp_id, {"oos_sharpe": 1.0})
        monkeypatch.setattr(er, "run_strategy", never_in_the_api)
        report = result_of(db, main.experiment_validate(exp_id, owner, db))
        reproducible = report["gates"]["result_reproducible"]
        assert reproducible["passed"] is False and "not executable" in reproducible["description"]
        assert report["gates"]["signature_valid"]["passed"] is True

    def test_tampered_inputs_fail_validation(self, db, owner):
        exp_id = create(db, owner)
        result_of(db, main.experiment_run(exp_id, owner, db))
        row = db.get(models.ResearchExperiment, exp_id)
        row.dataset_json = BARS[:-1]
        db.commit()
        report = result_of(db, main.experiment_validate(exp_id, owner, db))
        assert report["gates"]["inputs_unchanged"]["passed"] is False
        assert report["gates"]["result_reproducible"]["passed"] is False
        assert report["all_gates_passed"] is False

    def test_an_approval_made_while_validation_waits_is_not_overwritten(self, db, owner, monkeypatch):
        operator = make_user(db, "risk@example.com", role="risk_admin")
        exp_id = create(db, owner)
        result_of(db, main.experiment_run(exp_id, owner, db))
        quality_gates_pass(monkeypatch)
        first = result_of(db, main.experiment_validate(exp_id, owner, db))
        pending = main.experiment_validate(exp_id, owner, db)          # queued, not yet run
        assert main.experiment_approve(exp_id, operator, db)["status"] == "APPROVED"
        view = run_queued(db, pending)
        assert view["status"] == "failed" and view["error"]["status_code"] == 409
        row = db.get(models.ResearchExperiment, exp_id)
        assert row.status == "APPROVED" and row.validation_gates_json == first

    def test_resubmitting_while_queued_returns_the_same_job(self, db, owner):
        exp_id = create(db, owner)
        result_of(db, main.experiment_run(exp_id, owner, db))
        first = queued(main.experiment_validate(exp_id, owner, db), "experiment_validate")
        second = queued(main.experiment_validate(exp_id, owner, db), "experiment_validate")
        assert first["job_id"] == second["job_id"]


class TestReplayJob:
    def test_re_executes_and_reproduces_the_recorded_result(self, db, owner):
        exp_id = create(db, owner)
        ran = result_of(db, main.experiment_run(exp_id, owner, db))
        replay = result_of(db, main.experiment_replay(exp_id, {}, owner, db))
        assert replay["matches_experiment"] is True and replay["replay_status"] == "re-executed"
        assert replay["result_reproduced"] is True and replay["engine_version_matches"] is True
        assert replay["result_hash"] == replay["recorded_result_hash"] == ran["result_hash"]
        [audit] = audits(db, "EXPERIMENT_REPLAYED")
        assert audit.resource_id == exp_id and audit.metadata_json["result_reproduced"] is True

    def test_a_matching_result_from_other_inputs_is_not_a_reproduction(self, db, owner):
        # An extra bar field the strategy ignores leaves the result unchanged
        # but makes the executed inputs differ from the recorded ones.
        exp_id = create(db, owner)
        ran = result_of(db, main.experiment_run(exp_id, owner, db))
        other = [{**bar, "note": "not recorded"} for bar in BARS]
        assert er.hash_results(er.run_strategy("obi_momentum", other, PARAMS, SEED)) == ran["result_hash"]
        response = main.experiment_replay(exp_id, {}, owner, db)
        job = db.get(models.ResearchJob, json.loads(response.body)["job_id"])
        job.params_json = {**job.params_json, "dataset": other}
        db.commit()
        replay = result_of(db, response)
        assert replay["result_hash"] == ran["result_hash"] and replay["result_reproduced"] is False

    def test_changed_inputs_are_verified_but_not_executed(self, db, owner, monkeypatch):
        exp_id = create(db, owner)
        result_of(db, main.experiment_run(exp_id, owner, db))
        monkeypatch.setattr(er, "run_strategy", never_in_the_api)
        replay = result_of(db, main.experiment_replay(exp_id, {"random_seed": SEED + 1}, owner, db))
        assert replay["matches_experiment"] is False and replay["random_seed"] == SEED + 1
        assert replay["replay_status"] == "deterministic_input_verification"
        assert replay["result_reproduced"] is None

    def test_an_experiment_without_results_is_only_verified(self, db, owner):
        exp_id = create(db, owner)
        replay = result_of(db, main.experiment_replay(exp_id, {}, owner, db))
        assert replay["matches_experiment"] is True
        assert replay["replay_status"] == "deterministic_input_verification"


# ── 3. Statistical tests ───────────────────────────────────────────────────────

RETURNS = list(np.random.default_rng(1).normal(0.001, 0.01, 300))


class TestStatTestJob:
    def test_defaults_reproduce_the_synchronous_result(self, db, owner):
        result = result_of(db, main.statistical_tests(schemas.StatTestRequest(returns=RETURNS), owner, db))
        expected = research_jobs.json_safe(stat_tests.run_full_stat_tests(np.array(RETURNS), n_strategies_tested=1))
        assert result.pop("trial_count") == {"used": 1, "declared": 1, "server_counted": 0, "family": None}
        assert result == expected

    def test_bootstrap_and_permutation_counts_are_used(self, db, owner):
        req = schemas.StatTestRequest(returns=RETURNS, n_bootstrap=500, n_permutations=300)
        result = result_of(db, main.statistical_tests(req, owner, db))
        assert result["bootstrap_sharpe"]["n_bootstrap"] == 500
        assert result["permutation_test"]["n_permutations"] == 300
        expected = stat_tests.permutation_test(np.array(RETURNS), n_permutations=300)
        assert result["permutation_test"]["p_value"] == expected["p_value"]

    def test_returns_are_capped_and_must_be_finite(self):
        cap = schemas.STAT_TEST_MAX_RETURNS
        assert cap == 10_000
        assert len(schemas.StatTestRequest(returns=[0.001] * cap).returns) == cap
        with pytest.raises(ValidationError):
            schemas.StatTestRequest(returns=[0.001] * (cap + 1))
        for bad in (math.nan, math.inf, -math.inf):
            with pytest.raises(ValidationError):
                schemas.StatTestRequest(returns=[0.01, bad, 0.02, 0.0, 0.01])

    @pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
    def test_a_non_finite_return_is_a_422_not_a_500(self, owner, token):
        # The validation error echoes the rejected input; JSON cannot carry
        # NaN/Infinity, which used to turn the 422 into a 500.
        from fastapi.testclient import TestClient
        main.app.dependency_overrides[main.get_current_user] = lambda: owner
        try:
            response = TestClient(main.app).post(
                "/api/research/stat-test", content=f'{{"returns": [0.01, {token}, 0.02, 0.0, 0.01]}}',
                headers={"Content-Type": "application/json"})
        finally:
            main.app.dependency_overrides.pop(main.get_current_user, None)
        assert response.status_code == 422, response.text
        [error] = response.json()["detail"]
        assert error["type"] == "finite_number" and error["loc"] == ["body", "returns", 1]
        assert error["input"] == {"NaN": "nan", "Infinity": "inf", "-Infinity": "-inf"}[token]

    def test_stored_backtest_returns_get_the_same_limits(self, db, owner):
        def backtest(daily):
            row = models.Backtest(user_id=owner.id, initial_capital=1.0, final_capital=1.0, sharpe_ratio=0.0,
                                  max_drawdown=0.0, win_rate=0.0, total_trades=0,
                                  result_json={"daily_returns_net": daily})
            db.add(row)
            db.commit()
            return row.id
        for daily, detail in (([0.001] * (schemas.STAT_TEST_MAX_RETURNS + 1), "at most"),
                              ([0.01, None, 0.02, 0.0, 0.01, 0.03], "finite"),
                              ([0.01, 0.02, 0.0, 0.01], "at least 5")):
            with pytest.raises(HTTPException) as exc:
                main.statistical_tests(schemas.StatTestRequest(backtest_id=backtest(daily)), owner, db)
            assert exc.value.status_code == 422 and detail in exc.value.detail.lower()
        assert jobs(db) == []

    def test_server_counted_trials_and_the_audit_event(self, db, owner):
        fam = research_trials.family_hash("ma_crossover", ["AAPL"])
        research_trials.record(db, owner.id, fam, [{"fast": f} for f in range(25)], "walk_forward")
        response = main.statistical_tests(
            schemas.StatTestRequest(returns=RETURNS, n_strategies_tested=1, trial_family=fam), owner, db)
        job_id = json.loads(response.body)["job_id"]
        result = result_of(db, response)
        assert result["trial_count"] == {"used": 25, "declared": 1, "server_counted": 25, "family": fam}
        assert result["deflated_sharpe"]["n_trials"] == 25
        [audit] = audits(db, "STAT_TEST")
        assert audit.metadata_json == {"n_obs": 300, "n_strategies": 25, "declared_strategies": 1,
                                       "server_counted": 25, "job_id": job_id}


# ── 4. Browsers pick up the frontend that matches the API ──────────────────────

class TestVersionedAssets:
    """Assets are cached for a day; a deploy that changes the API contract
    (the stat-test answering 202) must not leave browsers running the old
    app.js against it."""

    ASSETS = ("styles.css", "vendor/three-global.js", "bg3d.js", "app.js")

    @staticmethod
    def version(path):
        import hashlib
        return hashlib.sha256(path.read_bytes()).hexdigest()[:12]

    def test_index_names_every_asset_by_its_content(self):
        from fastapi.testclient import TestClient
        client = TestClient(main.app)
        page = client.get("/")
        assert page.status_code == 200 and page.headers["cache-control"] == "no-cache"
        for name in self.ASSETS:
            assert f'"/assets/{name}?v={self.version(main.FRONTEND_DIR / name)}"' in page.text, name
            assert f'"/assets/{name}"' not in page.text, name
        asset = client.get(f"/assets/app.js?v={self.version(main.FRONTEND_DIR / 'app.js')}")
        assert asset.status_code == 200 and asset.content == (main.FRONTEND_DIR / "app.js").read_bytes()
        assert client.get("/research/lab").text == page.text            # SPA routes too

    def test_an_edited_asset_gets_a_new_url(self, tmp_path, monkeypatch):
        import os
        import shutil
        from fastapi.testclient import TestClient
        shutil.copytree(main.FRONTEND_DIR, tmp_path / "frontend")
        monkeypatch.setattr(main, "FRONTEND_DIR", tmp_path / "frontend")
        client = TestClient(main.app)
        before = self.version(tmp_path / "frontend" / "app.js")
        assert f"app.js?v={before}" in client.get("/").text
        app_js = tmp_path / "frontend" / "app.js"
        app_js.write_text(app_js.read_text(encoding="utf-8") + "\n// changed\n", encoding="utf-8")
        stat = app_js.stat()
        os.utime(app_js, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        after = self.version(app_js)
        assert after != before
        page = client.get("/").text
        assert f"app.js?v={after}" in page and f"app.js?v={before}" not in page
