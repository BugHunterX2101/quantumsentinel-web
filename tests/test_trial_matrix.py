"""The trial matrix of a strategy evaluation's family (R3-2).

POST /api/research/validation-matrix queues the ``validation_matrix`` job:
the worker reads the candidate's evaluation and the family's trials, the job
process re-runs every strategy version evaluated in the family on the
candidate's datasets, and the worker accepts the matrix only if the family
did not change meanwhile (services/trial_matrix.py).
"""
import datetime as dt
import hashlib
import json
import re

import pytest
from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from backend import main, models, schemas, worker
from backend.services import (backtest_service, dataset_holdouts, research_jobs, research_trials,
                              strategy_registry, strategy_spec, trial_matrix)
from research_job_helpers import InlineJobProcess, run_queued
from test_trial_evaluations import (SYMBOLS, Session, backtest, db, evaluations, make_user, market,  # noqa: F401
                                    register, strategy)

A, B, C, D = ("qs-sv-sha256-v1:" + c * 64 for c in "abcd")
FAMILY = research_trials.family_hash("ma_crossover", SYMBOLS)


# --------------------------------------------------------------------------
# The matrix (no database)
# --------------------------------------------------------------------------

GOLDEN_DATES = ["2024-01-02", "2024-01-03", "2024-01-04"]
GOLDEN_RUNS = {C: [0.0, -0.02, 0.003], A: [0.0, 0.01, -0.005], B: [0.0, 0.01, -0.005]}
GOLDEN_MATRIX_ID = "qs-tm-sha256-v1:19713d4cc7f00b835ecf4fa0cb6c45d58317fe73c178b6de407e74154c307452"


def test_the_matrix_id_matches_its_golden_vector():
    """Pinned, and recomputed here from the definition (SHA-256 of the
    canonical JSON of scheme, dates and columns), so another implementation
    can check itself against it."""
    matrix = trial_matrix.build(GOLDEN_DATES, GOLDEN_RUNS)
    same = research_trials.returns_id([0.0, 0.01, -0.005])
    other = research_trials.returns_id([0.0, -0.02, 0.003])
    expected_columns = sorted([
        {"returns_sha256": same, "strategy_version_ids": [A, B], "returns": [0, 10_000_000_000, -5_000_000_000]},
        {"returns_sha256": other, "strategy_version_ids": [C], "returns": [0, -20_000_000_000, 3_000_000_000]},
    ], key=lambda column: column["returns_sha256"])
    body = {"scheme": "qs-tm-sha256-v1", "dates": GOLDEN_DATES, "columns": expected_columns}
    independent = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert matrix == {"id": f"qs-tm-sha256-v1:{independent}", **body}
    assert matrix["id"] == GOLDEN_MATRIX_ID
    assert trial_matrix.MATRIX_ID.fullmatch(matrix["id"])
    assert trial_matrix.matrix_id(matrix) == matrix["id"]


def test_versions_with_the_same_series_share_one_column_and_columns_follow_their_ids():
    matrix = trial_matrix.build(GOLDEN_DATES, GOLDEN_RUNS)
    assert sorted(c["strategy_version_ids"] for c in matrix["columns"]) == [[A, B], [C]]
    assert [c["returns_sha256"] for c in matrix["columns"]] == sorted(c["returns_sha256"] for c in matrix["columns"])
    assert trial_matrix.column_of(matrix, B) == trial_matrix.column_of(matrix, A) != trial_matrix.column_of(matrix, C)
    with pytest.raises(trial_matrix.MatrixError):
        trial_matrix.column_of(matrix, "qs-sv-sha256-v1:" + "d" * 64)
    # The order the runs come in changes nothing.
    assert trial_matrix.build(GOLDEN_DATES, dict(reversed(list(GOLDEN_RUNS.items())))) == matrix


def test_a_columns_values_hash_to_its_returns_sha256_exactly():
    """The stored integers are the series' quanta, so its id is recomputed
    from them without going back through floats."""
    returns = [0.0, 1234.5678, -0.9999, 1e-12]
    matrix = trial_matrix.build(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"], {A: returns})
    [column] = matrix["columns"]
    assert column["returns"] == research_trials.quantise(returns)
    assert column["returns_sha256"] == research_trials.returns_id(returns) == \
        research_trials.quantised_returns_id(column["returns"])


@pytest.mark.parametrize("dates, runs", [
    (GOLDEN_DATES, {}),
    (GOLDEN_DATES, {A: [0.0, 0.01]}),
    (GOLDEN_DATES, {A: [0.0, 0.01, float("nan")]}),
    (GOLDEN_DATES, {A: [0.0, 0.01, 1e4]}),  # beyond what canonical JSON holds as quanta
    (["2024-01-03", "2024-01-02", "2024-01-04"], {A: [0.0, 0.01, 0.02]}),
    (["2024-01-02", "2024-01-02", "2024-01-04"], {A: [0.0, 0.01, 0.02]}),
    (["2024-1-2", "2024-01-03", "2024-01-04"], {A: [0.0, 0.01, 0.02]}),
    ([], {A: []}),
])
def test_runs_that_do_not_form_a_matrix_are_refused(dates, runs):
    with pytest.raises(ValueError):
        trial_matrix.build(dates, runs)


def _tampered(change):
    matrix = json.loads(json.dumps(trial_matrix.build(GOLDEN_DATES, GOLDEN_RUNS)))
    pair = next(c for c in matrix["columns"] if len(c["strategy_version_ids"]) == 2)
    single = next(c for c in matrix["columns"] if len(c["strategy_version_ids"]) == 1)
    change(matrix, pair, single)
    return matrix


@pytest.mark.parametrize("change", [
    lambda m, pair, single: m.update(scheme="qs-tm-sha256-v0"),
    lambda m, pair, single: pair["returns"].__setitem__(1, pair["returns"][1] + 1),
    lambda m, pair, single: pair["returns"].append(0),
    lambda m, pair, single: pair["returns"].__setitem__(0, 0.0),
    lambda m, pair, single: pair["returns"].__setitem__(0, True),
    lambda m, pair, single: m["columns"].reverse(),
    lambda m, pair, single: m["columns"].append(pair),
    lambda m, pair, single: pair["strategy_version_ids"].reverse(),
    lambda m, pair, single: single["strategy_version_ids"].clear(),
    lambda m, pair, single: single["strategy_version_ids"].append(single["strategy_version_ids"][0]),
    lambda m, pair, single: single["strategy_version_ids"].insert(0, pair["strategy_version_ids"][0]),
    lambda m, pair, single: pair.update(extra=1),
    lambda m, pair, single: m["dates"].reverse(),
    lambda m, pair, single: m.update(columns=[]),
    lambda m, pair, single: m.update(dates=[]),
])
def test_a_matrix_that_is_not_one_has_no_id(change):
    with pytest.raises(trial_matrix.MatrixError):
        trial_matrix.matrix_id(_tampered(change))


def _rehashed(column):
    column["returns_sha256"] = research_trials.quantised_returns_id(column["returns"])


def _resorted(matrix):
    matrix["columns"].sort(key=lambda column: column["returns_sha256"])


@pytest.mark.parametrize("change, message", [
    # Each change breaks only the rule it names: the hashes, the order and
    # the other rules still hold, so no other check can refuse it instead.
    (lambda m, pair, single: m["dates"].__setitem__(2, "2024-01-04T00:00"), "YYYY-MM-DD"),
    (lambda m, pair, single: (pair["returns"].append(0), _rehashed(pair), _resorted(m)), "4 returns for 3 sessions"),
    (lambda m, pair, single: single["strategy_version_ids"].append(single["strategy_version_ids"][0]),
     "distinct strategy versions"),
    (lambda m, pair, single: m["columns"].insert(m["columns"].index(pair) + 1,
                                                 {**pair, "strategy_version_ids": [D]}), "distinct and ordered"),
])
def test_each_rule_of_a_matrix_is_checked_on_its_own(change, message):
    with pytest.raises(trial_matrix.MatrixError, match=message):
        trial_matrix.matrix_id(_tampered(change))


def test_a_run_of_the_wrong_length_is_refused_by_name():
    with pytest.raises(trial_matrix.MatrixError, match=re.escape(f"{A}: 2 returns for 3 sessions")):
        trial_matrix.build(GOLDEN_DATES, {A: [0.0, 0.01]})


def test_a_change_to_any_value_changes_the_id():
    base = trial_matrix.build(GOLDEN_DATES, GOLDEN_RUNS)["id"]
    assert trial_matrix.build(["2024-01-02", "2024-01-03", "2024-01-05"], GOLDEN_RUNS)["id"] != base
    assert trial_matrix.build(GOLDEN_DATES, {**GOLDEN_RUNS, C: [0.0, -0.02, 0.003 + 2e-12]})["id"] != base
    assert trial_matrix.build(GOLDEN_DATES, {**GOLDEN_RUNS, B: [0.0, -0.02, 0.003]})["id"] != base


# --------------------------------------------------------------------------
# The job, end to end
# --------------------------------------------------------------------------

def request(evaluation, period="1y", initial_capital=100_000.0):
    return schemas.ValidationMatrixRequest(evaluation=evaluation, period=period, initial_capital=initial_capital)


def queue(db, user, evaluation, **fields):
    return main.validation_matrix(request(evaluation, **fields), user=user, db=db)


def build(db, user, evaluation, job_process=None, **fields):
    return run_queued(db, queue(db, user, evaluation, **fields), job_process)


def evaluated(db, user, value=None, **fields):
    """Register a version, backtest it, and return (version id, job view)."""
    version_id = register(db, user, value)
    view = backtest(db, user, version_id, **fields)
    assert view["status"] == "succeeded", view
    return version_id, view


def trial_count(db, user, family=FAMILY):
    db.expire_all()
    return research_trials.count(db, user.id, family)


def matrix_audits(db):
    return db.execute(select(models.AuditLog).where(models.AuditLog.action == "VALIDATION_MATRIX")).scalars().all()


def test_the_matrix_re_runs_every_version_evaluated_in_the_family_on_the_candidates_data(db, market):
    user = make_user(db)
    first, view = evaluated(db, user)
    datasets = view["result"]["market_data"]["datasets"]
    second, _ = evaluated(db, user, strategy(**{"signal.fast": 10, "signal.slow": 40}), dataset_ids=datasets)
    # Differs only in a field the engine does not read: the same series.
    twin, _ = evaluated(db, user, strategy(**{"risk_limits.max_drawdown_bps": 1_000}), dataset_ids=datasets)
    before, rows_before = trial_count(db, user), len(evaluations(db))

    built = build(db, user, view["job_id"])
    assert built["status"] == "succeeded", built
    result = built["result"]
    matrix = result["matrix"]
    assert trial_matrix.matrix_id(matrix) == matrix["id"]
    by_version = {v: c for c in matrix["columns"] for v in c["strategy_version_ids"]}
    assert sorted(by_version) == sorted([first, second, twin]) and len(matrix["columns"]) == 2
    assert by_version[first] is by_version[twin]
    # Each re-run reproduces what that version's own evaluation measured on
    # these datasets, over this period, with this capital.
    measured = {row.strategy_version_id: row.returns_sha256 for row in evaluations(db)}
    assert {v: c["returns_sha256"] for v, c in by_version.items()} == measured
    assert len(matrix["dates"]) == result["sessions"]["count"] == view["result"]["evaluation"]["n_obs"]
    assert result["candidate"] == {"evaluation": view["job_id"], "strategy_version_id": first,
                                   "returns_sha256": view["result"]["evaluation"]["returns_sha256"],
                                   "column": trial_matrix.column_of(matrix, first)}
    assert result["family"] == FAMILY
    assert result["ledger"] == {"trials": 3, "research_trials": 3, "holdout_queries": 0, "sweep_rejections": 0,
                                "evaluated_trials": 3, "every_trial_evaluated": True,
                                "strategy_version_ids": sorted([first, second, twin])}
    assert sorted(result["market_data"]["datasets"]) == sorted(datasets)
    # Building the matrix records no trial and no evaluation.
    assert "trial_family" not in result and trial_count(db, user) == before == 3
    assert len(evaluations(db)) == rows_before
    [audit] = matrix_audits(db)
    assert (audit.resource_type, audit.resource_id) == ("strategy_version", first)
    assert audit.metadata_json["matrix_id"] == matrix["id"]
    assert audit.metadata_json["every_trial_evaluated"] is True


def test_the_job_runs_on_exactly_the_datasets_the_evaluation_read(db, market):
    user = make_user(db)
    _, view = evaluated(db, user)
    response = queue(db, user, view["job_id"])
    job = db.get(models.ResearchJob, json.loads(response.body)["job_id"])
    assert job.kind == "validation_matrix"
    assert job.params_json["dataset_ids"] == sorted(view["result"]["market_data"]["datasets"])
    assert job.params_json["evaluation"] == view["job_id"]


def test_trials_without_an_evaluation_leave_the_matrix_incomplete(db, market, monkeypatch):
    """A moving-average backtest of the same symbols and a run whose result
    was refused both count as trials of the family; neither has a version to
    re-run, so the matrix covers fewer trials than the ledger counts."""
    monkeypatch.setattr(backtest_service, "run_moving_average_backtest", lambda *a: {
        "initial_capital": 100_000.0, "final_capital": 101_000.0, "total_return": 0.01,
        "sharpe_ratio": 1.0, "max_drawdown": -0.02, "win_rate": 0.5, "total_trades": 4})
    user = make_user(db)
    one = [SYMBOLS[0]]
    family = research_trials.family_hash("ma_crossover", one)
    candidate, view = evaluated(db, user, strategy(**{"universe.symbols": one}))
    complete = build(db, user, view["job_id"])
    assert complete["status"] == "succeeded", complete
    assert complete["result"]["ledger"]["every_trial_evaluated"] is True

    dashboard = run_queued(db, main.run_backtest(schemas.BacktestRequest(asset=SYMBOLS[0]), user=user, db=db))
    assert dashboard["status"] == "succeeded" and dashboard["result"]["trial_family"] == family
    refused = backtest(db, user, register(db, user, strategy(**{"universe.symbols": one,
                                                                 "universe.family": "CRYPTO_SPOT_EOD"})))
    assert refused["status"] == "failed" and trial_count(db, user, family) == 3

    partial = build(db, user, view["job_id"])
    assert partial["status"] == "succeeded", partial
    assert partial["result"]["ledger"] == {"trials": 3, "research_trials": 3, "holdout_queries": 0,
                                           "sweep_rejections": 0, "evaluated_trials": 1, "every_trial_evaluated": False,
                                           "strategy_version_ids": [candidate]}
    assert trial_count(db, user, family) == 3


def test_the_candidate_must_reproduce_its_evaluation(db, market, monkeypatch):
    """A change to the engine since the evaluation refuses the matrix: the
    candidate is backtested again first."""
    user = make_user(db)
    _, view = evaluated(db, user)
    run = strategy_spec.CompiledStrategy.run

    def changed_engine(self, *args, **kwargs):
        result = run(self, *args, **kwargs)
        result["returns"] = result["returns"] * 1.001
        return result

    monkeypatch.setattr(strategy_spec.CompiledStrategy, "run", changed_engine)
    built = build(db, user, view["job_id"])
    assert built["status"] == "failed" and built["error"]["status_code"] == 409
    assert "backtest it again" in built["error"]["detail"]
    assert matrix_audits(db) == []


def test_a_holdout_set_after_the_evaluation_refuses_the_matrix(db, market):
    """The holdout cuts the window before it is anchored, so the candidate's
    run no longer reproduces and must be backtested again on what remains."""
    user = make_user(db)
    _, view = evaluated(db, user)
    dataset_holdouts.set_start(db, "EQUITY_EOD", dt.date(2024, 2, 1), set_by="ops", reason="test")
    db.commit()
    built = build(db, user, view["job_id"])
    assert built["status"] == "failed" and built["error"]["status_code"] == 409


def test_only_the_callers_evaluation_with_the_period_and_capital_it_ran_with_can_be_queued(db, market):
    owner, other = make_user(db), make_user(db, "other@example.com")
    _, view = evaluated(db, owner)
    for user, evaluation, fields, status in (
            (other, view["job_id"], {}, 404),
            (owner, "no-such-job", {}, 404),
            (owner, view["job_id"], {"period": "6mo"}, 422),
            (owner, view["job_id"], {"initial_capital": 100_001.0}, 422)):
        with pytest.raises(HTTPException) as caught:
            queue(db, user, evaluation, **fields)
        assert caught.value.status_code == status
    assert db.execute(select(func.count()).select_from(models.ResearchJob)
                      .where(models.ResearchJob.kind == "validation_matrix")).scalar() == 0


class MustNotRun(InlineJobProcess):
    def run(self, *args, **kwargs):
        raise AssertionError("the task ran")


def run_job(db, user, job_process, **params):
    """Queue a validation matrix job around the API and run it."""
    job = research_jobs.enqueue(db, user.id, "validation_matrix", schemas.ValidationMatrixJob(**params).model_dump())
    runner = worker.Worker(job_process=job_process, session_factory=sessionmaker(bind=db.get_bind()),
                           lease_seconds=60, timeout_seconds=300, poll_seconds=0.1)
    assert runner.run_once() is True
    db.expire_all()
    return research_jobs.view(db, db.get(models.ResearchJob, job.id))


def test_the_worker_checks_the_job_against_the_evaluation_before_it_runs(db, market):
    """A job queued around the API (another user's evaluation, another
    period, other datasets) fails before the job process runs."""
    owner, other = make_user(db), make_user(db, "other@example.com")
    _, view = evaluated(db, owner)
    datasets = sorted(view["result"]["market_data"]["datasets"])
    good = dict(evaluation=view["job_id"], period="1y", initial_capital=100_000.0, dataset_ids=datasets)
    assert run_job(db, other, MustNotRun(), **good)["error"] == {"status_code": 404, "detail": "Evaluation not found"}
    assert run_job(db, owner, MustNotRun(), **{**good, "period": "6mo"})["error"]["status_code"] == 422
    wrong = run_job(db, owner, MustNotRun(), **{**good, "dataset_ids": ["qs-ds-sha256-v1:" + "0" * 64]})
    assert wrong["error"] == {"status_code": 422, "detail": "The datasets are not the ones the evaluation read"}
    assert run_job(db, owner, InlineJobProcess(), **good)["status"] == "succeeded"


def test_a_version_that_can_no_longer_be_read_refuses_the_matrix(db, market, monkeypatch):
    """Leaving it out would understate the number of trials."""
    user = make_user(db)
    _, view = evaluated(db, user)
    broken, _ = evaluated(db, user, strategy(**{"signal.fast": 10, "signal.slow": 40}),
                          dataset_ids=view["result"]["market_data"]["datasets"])
    spec_of = strategy_registry.spec_of

    def refuse(version):
        if version.version_id == broken:
            raise strategy_registry.StrategyRegistryError(f"{broken}: the stored spec no longer validates")
        return spec_of(version)

    monkeypatch.setattr(strategy_registry, "spec_of", refuse)
    built = build(db, user, view["job_id"], MustNotRun())
    assert built["status"] == "failed" and built["error"]["status_code"] == 422
    assert broken in built["error"]["detail"]


def test_a_version_the_engine_refuses_refuses_the_matrix(db, market, monkeypatch):
    user = make_user(db)
    _, view = evaluated(db, user)
    broken, _ = evaluated(db, user, strategy(**{"signal.fast": 10, "signal.slow": 40}),
                          dataset_ids=view["result"]["market_data"]["datasets"])
    run = strategy_spec.CompiledStrategy.run

    def refuses_one(self, *args, **kwargs):
        if self.version_id == broken:
            raise strategy_spec.StrategySpecError("the engine refuses it")
        return run(self, *args, **kwargs)

    monkeypatch.setattr(strategy_spec.CompiledStrategy, "run", refuses_one)
    built = build(db, user, view["job_id"])
    assert built["status"] == "failed"
    assert built["error"] == {"status_code": 422, "detail": f"{broken} cannot be re-run: the engine refuses it"}


class RecordsATrialMidJob(InlineJobProcess):
    def __init__(self, db, user):
        self.Session, self.user = sessionmaker(bind=db.get_bind()), user

    def run(self, *args, **kwargs):
        with self.Session() as other:
            research_trials.record(other, self.user.id, FAMILY, [{"another": "configuration"}], "strategy_backtest")
        return super().run(*args, **kwargs)


def test_a_family_that_changes_while_the_matrix_is_built_refuses_it(db, market):
    user = make_user(db)
    _, view = evaluated(db, user)
    built = build(db, user, view["job_id"], RecordsATrialMidJob(db, user))
    assert built["status"] == "failed" and built["error"]["status_code"] == 409
    assert trial_count(db, user) == 2


def test_the_worker_refuses_a_matrix_that_leaves_out_a_version(db, market, monkeypatch):
    user = make_user(db)
    _, view = evaluated(db, user)
    left_out, _ = evaluated(db, user, strategy(**{"signal.fast": 10, "signal.slow": 40}),
                            dataset_ids=view["result"]["market_data"]["datasets"])
    real_build = trial_matrix.build
    monkeypatch.setattr(trial_matrix, "build",
                        lambda dates, runs: real_build(dates, {v: r for v, r in runs.items() if v != left_out}))
    built = build(db, user, view["job_id"])
    assert built["status"] == "failed"
    assert built["error"] == {"status_code": 500, "detail": "The research job reported an invalid matrix"}


class MisreportsItsResult(InlineJobProcess):
    """A job process whose result is changed after the task ran, as a
    misbehaving or compromised job process could report it."""

    def __init__(self, change):
        self.change = change

    def run(self, *args, **kwargs):
        outcome = super().run(*args, **kwargs)
        assert outcome[0] == "ok", outcome
        self.change(outcome[1])
        return outcome


def test_the_worker_refuses_prices_from_other_datasets_than_the_evaluations(db, market):
    """The datasets the job says it read must be the evaluation's, not just
    datasets the user holds (which is all record_market_data asks)."""
    user = make_user(db)
    _, view = evaluated(db, user)
    built = build(db, user, view["job_id"], MisreportsItsResult(lambda r: r["market_data"].update(datasets=[])))
    assert built["status"] == "failed"
    assert built["error"] == {"status_code": 422,
                              "detail": "The prices came from other datasets than the ones the evaluation read"}
    assert matrix_audits(db) == []


@pytest.mark.parametrize("change", [
    lambda r: r["matrix"].update(id=trial_matrix.MATRIX_SCHEME + ":" + "0" * 64),
    lambda r: r["matrix"]["columns"][0]["returns"].__setitem__(-1, r["matrix"]["columns"][0]["returns"][-1] + 1),
])
def test_the_worker_verifies_the_matrix_id(db, market, change):
    user = make_user(db)
    _, view = evaluated(db, user)
    built = build(db, user, view["job_id"], MisreportsItsResult(change))
    assert built["status"] == "failed"
    assert built["error"] == {"status_code": 500, "detail": "The research job reported an invalid matrix"}
    assert matrix_audits(db) == []


def test_the_task_refuses_specs_other_than_the_ones_it_names():
    value = json.loads(strategy_spec.canonical(strategy_spec.parse(strategy())))
    version_id = strategy_spec.version_id(strategy_spec.parse(value))
    params = schemas.ValidationMatrixJob(evaluation="j", period="1y", initial_capital=100_000.0,
                                         dataset_ids=["qs-ds-sha256-v1:" + "0" * 64]).model_dump()
    target = research_jobs.KINDS["validation_matrix"].target
    candidate = {"evaluation": "j", "strategy_version_id": version_id, "returns_sha256": "x"}
    another = "The research job was given another strategy than the ones it names"
    for prepared, detail in (
            # The spec is filed under an id that is not its own (the candidate's, or another).
            ({"candidate": {**candidate, "strategy_version_id": A}, "family": FAMILY, "ledger": {},
              "specs": {A: value}}, another),
            ({"candidate": candidate, "family": FAMILY, "ledger": {}, "specs": {A: value}}, another),
            ({"candidate": candidate, "family": "another family", "ledger": {}, "specs": {version_id: value}},
             another),
            ({"candidate": {**candidate, "strategy_version_id": A}, "family": FAMILY, "ledger": {},
              "specs": {version_id: value}}, "The research job was not given the candidate's strategy")):
        assert worker._run_target(target, params, None, prepared) == ("error", 500, detail)


def test_the_request_needs_the_evaluation_period_and_capital():
    for fields in ({"period": "1y", "initial_capital": 100_000.0},
                   {"evaluation": "j", "initial_capital": 100_000.0},
                   {"evaluation": "j", "period": "1y"},
                   {"evaluation": "j", "period": "1 year", "initial_capital": 100_000.0},
                   {"evaluation": "j", "period": "1y", "initial_capital": 10.0}):
        with pytest.raises(ValueError):
            schemas.ValidationMatrixRequest(**fields)
    with pytest.raises(ValueError):
        schemas.ValidationMatrixJob(evaluation="j", period="1y", initial_capital=100_000.0)
