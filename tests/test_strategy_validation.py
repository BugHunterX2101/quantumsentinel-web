"""Strategy validations (R3-4): the validation policy's statistical gates
(services/validation_policy.py) and the ``strategy_validation`` job that
records their verdict (services/strategy_validation.py, migration 0008).

The gates are tested on constructed matrices, where a candidate that stands
out from noise passes every gate and each gate can be made to fail, or to be
inapplicable (which fails it), on its own. The job is tested end to end on
the synthetic market of test_trial_evaluations, whose small families can
only fail (fewer distinct series than PBO needs).
"""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException
from sqlalchemy import func, select, text
from sqlalchemy.exc import DBAPIError

from backend import main, models, schemas
from backend.services import (research_trials, strategy_registry, strategy_spec, strategy_validation,
                              trial_matrix, validation_policy, validation_stats)
from backend.services.validation_policy import POLICY_V1, evaluate, policy_id
from research_job_helpers import run_queued
from test_trial_evaluations import Session, db, make_user, market, strategy  # noqa: F401
from test_trial_matrix import FAMILY, MisreportsItsResult, evaluated, trial_count
from test_validation_policy import POLICY_V1_ID

V = ["qs-sv-sha256-v1:" + f"{i:064x}" for i in range(1, 41)]
CANDIDATE = V[0]
LIMITS = strategy_spec.RiskLimits(max_drawdown_bps=2_500, max_position_bps=2_500)
RISK = {"max_drawdown": 0.10, "max_position_share": 0.20, "equity_exhausted": False}


def family(sessions=504, strategies=10, drift=0.003, seed=7, twins=()):
    """A matrix of a candidate (V[0]) with daily returns of mean ``drift``
    and ``strategies - 1`` zero-mean variants, all with a 1% daily standard
    deviation; each of ``twins`` shares the column of the version it names."""
    rng = np.random.default_rng(seed)
    dates = [str(day.date()) for day in pd.bdate_range("2022-01-03", periods=sessions)]
    runs = {CANDIDATE: list(rng.normal(drift, 0.01, sessions))}
    for version in V[1:strategies]:
        runs[version] = list(rng.normal(0.0, 0.01, sessions))
    for twin, of in twins:
        runs[twin] = runs[of]
    return trial_matrix.build(dates, runs)


def runs_of(matrix):
    return {v: [r / 1e12 for r in c["returns"]] for c in matrix["columns"] for v in c["strategy_version_ids"]}


def ledger(trials, complete=True):
    return {"trials": trials, "every_trial_evaluated": complete}


def verdict(matrix, trials=10, family_name="EQUITY_EOD", risk=RISK, candidate=CANDIDATE, **ledger_fields):
    return evaluate(matrix, candidate, {**ledger(trials), **ledger_fields}, family_name, LIMITS, risk)


def failed_closed(gate):
    return gate["passed"] is False and gate["reason"].startswith("not evaluated, so the gate fails")


# --------------------------------------------------------------------------
# The gates (no database)
# --------------------------------------------------------------------------

def test_a_candidate_that_stands_out_from_its_family_passes_every_gate():
    """The pass path, which the synthetic market of the job tests cannot
    reach: ten distinct series over 504 sessions, the candidate's t about 4."""
    found = verdict(family())
    assert found["passed"] is True, found
    assert set(found["gates"]) == {"risk_limits", *validation_policy.STATISTICAL_GATES}
    assert all(gate["passed"] for gate in found["gates"].values())
    assert (found["policy"], found["stage"], found["candidate"]) == (POLICY_V1_ID, "pre-holdout", CANDIDATE)
    gates = found["gates"]
    assert gates["deflated_sharpe"]["value"] >= 0.95 and gates["deflated_sharpe"]["n_trials_source"] == "implied"
    assert gates["overfitting"]["n_splits"] == 16 and gates["overfitting"]["value"] <= 0.05
    assert gates["t_statistic"]["value"] >= 3 and gates["haircut_sharpe"]["value_annual"] >= 0.5
    assert gates["backtest_length"]["sessions"] == 504 > gates["backtest_length"]["minimum"]
    json.dumps(found, allow_nan=False)  # storable as JSON


def test_each_statistical_gate_reports_what_its_statistic_computes():
    matrix = family()
    columns = [c["returns"] for c in matrix["columns"]]
    index = trial_matrix.column_of(matrix, CANDIDATE)
    gates = verdict(matrix)["gates"]
    dsr = validation_stats.deflated_sharpe(columns, index, 10, True)
    assert gates["deflated_sharpe"]["value"] == dsr["dsr"]
    assert gates["deflated_sharpe"]["n_trials"] == dsr["n_trials"]
    assert gates["overfitting"]["value"] == validation_stats.pbo_cscv(columns, 16)["pbo"]
    sharpe = validation_stats.sharpe(columns[index])
    assert gates["t_statistic"]["value"] == sharpe * 504 ** 0.5
    assert gates["backtest_length"]["minimum"] == validation_stats.min_backtest_length(10, sharpe)
    others = [validation_stats.p_value(validation_stats.sharpe(c), 504) for i, c in enumerate(columns) if i != index]
    haircut = validation_stats.haircut(sharpe, 504, 10, "bhy", others)
    assert gates["haircut_sharpe"]["haircut_sharpe"] == haircut["haircut_sharpe"]
    assert gates["haircut_sharpe"]["value_annual"] == haircut["haircut_sharpe"] * 252 ** 0.5


def test_a_skill_less_candidate_fails_every_statistical_gate():
    found = verdict(family(drift=0.0005))
    gates = found["gates"]
    assert found["passed"] is False and gates["risk_limits"]["passed"] is True
    assert gates["deflated_sharpe"]["passed"] is False and gates["deflated_sharpe"]["value"] < 0.95
    assert gates["overfitting"]["passed"] is False and gates["overfitting"]["value"] > 0.05
    assert gates["t_statistic"]["passed"] is False and gates["t_statistic"]["value"] < 3
    # Its Sharpe ratio is not positive: no haircut or minimum length exists.
    assert failed_closed(gates["haircut_sharpe"]) and "positive" in gates["haircut_sharpe"]["reason"]
    assert failed_closed(gates["backtest_length"]) and "positive" in gates["backtest_length"]["reason"]


def test_a_breached_risk_limit_alone_fails_the_verdict():
    found = verdict(family(), risk={**RISK, "max_drawdown": 0.30})
    assert found["passed"] is False
    assert [name for name, gate in found["gates"].items() if not gate["passed"]] == ["risk_limits"]


def test_missing_risk_measurements_fail_the_risk_gate():
    found = verdict(family(), risk=None)
    assert found["passed"] is False and found["gates"]["risk_limits"]["passed"] is False


def test_a_short_backtest_fails_the_t_statistic_and_the_haircut():
    gates = verdict(family(sessions=251))["gates"]
    assert gates["t_statistic"]["passed"] is False and gates["t_statistic"]["value"] < 3
    assert gates["haircut_sharpe"]["passed"] is False and gates["haircut_sharpe"]["value_annual"] < 0.5


def test_more_trials_than_the_backtest_can_bear_fail_the_minimum_length():
    gates = verdict(family(), trials=10 ** 6)["gates"]
    assert gates["backtest_length"]["passed"] is False
    assert gates["backtest_length"]["minimum"] > 504 and gates["backtest_length"]["n_trials"] == 10 ** 6


def test_an_incomplete_ledger_deflates_by_the_raw_count():
    complete = verdict(family())["gates"]["deflated_sharpe"]
    partial = verdict(family(), trials=40, every_trial_evaluated=False)["gates"]["deflated_sharpe"]
    assert partial["n_trials_source"] == "ledger" and partial["n_trials"] == 40
    assert partial["value"] < complete["value"]


@pytest.mark.parametrize("overfit, passed", [(1, True), (2, False)])
def test_the_pbo_ceiling_is_compared_exactly(monkeypatch, overfit, passed):
    """1/20 is exactly 5%: at the ceiling passes, above it fails."""
    monkeypatch.setattr(validation_stats, "pbo_cscv", lambda columns, splits: {
        "n_splits": splits, "rows": 504, "rows_used": 496, "rows_dropped": 8, "strategies": len(columns),
        "combinations": 20, "overfit": overfit, "pbo": overfit / 20})
    assert verdict(family())["gates"]["overfitting"]["passed"] is passed


@pytest.mark.parametrize("dsr, passed", [(0.95, True), (0.9499999, False)])
def test_the_dsr_floor_is_inclusive(monkeypatch, dsr, passed):
    real = validation_stats.deflated_sharpe
    monkeypatch.setattr(validation_stats, "deflated_sharpe", lambda *args: {**real(*args), "dsr": dsr})
    assert verdict(family())["gates"]["deflated_sharpe"]["passed"] is passed


# A year of 256 sessions makes sqrt exact (16), so the floors below can be
# met exactly.
EXACT = validation_policy.ValidationPolicy(**{**POLICY_V1.__dict__, "name": "exact",
                                              "sessions_per_year": (("EQUITY_EOD", 256),)})


@pytest.mark.parametrize("haircut_sharpe, passed", [(0.5 / 16, True), (0.5 / 16 - 1e-9, False)])
def test_the_haircut_floor_is_inclusive(monkeypatch, haircut_sharpe, passed):
    real = validation_stats.haircut
    monkeypatch.setattr(validation_stats, "haircut",
                        lambda *args: {**real(*args), "haircut_sharpe": haircut_sharpe})
    gate = evaluate(family(), CANDIDATE, ledger(10), "EQUITY_EOD", LIMITS, RISK, EXACT)["gates"]["haircut_sharpe"]
    assert gate["passed"] is passed and (gate["value_annual"] == 0.5) is passed


@pytest.mark.parametrize("sharpe, passed", [(3 / 16, True), (3 / 16 - 1e-9, False)])
def test_the_t_statistic_floor_is_inclusive(monkeypatch, sharpe, passed):
    """Over 256 sessions a per-period Sharpe ratio of 3/16 is t = 3 exactly."""
    monkeypatch.setattr(validation_stats, "sharpe", lambda column: sharpe)
    gate = verdict(family(sessions=256))["gates"]["t_statistic"]
    assert gate["passed"] is passed and (gate["value"] == 3.0) is passed


@pytest.mark.parametrize("minimum, passed", [(504.0, True), (504.5, False)])
def test_the_minimum_backtest_length_is_inclusive(monkeypatch, minimum, passed):
    monkeypatch.setattr(validation_stats, "min_backtest_length", lambda n, sr: minimum)
    assert verdict(family())["gates"]["backtest_length"]["passed"] is passed


@pytest.mark.parametrize("sessions, splits", [(504, 16), (336, 16), (335, 12), (300, 12), (252, 12),
                                              (251, 8), (168, 8)])
def test_pbo_uses_the_most_splits_whose_blocks_are_long_enough(sessions, splits):
    """Blocks of at least 21 sessions: 16 splits from 336 sessions, 12 from
    252, 8 from 168."""
    gate = verdict(family(sessions=sessions))["gates"]["overfitting"]
    assert gate["n_splits"] == splits and gate["rows_used"] // splits >= 21


def test_pbo_is_inapplicable_to_too_short_a_backtest():
    gate = verdict(family(sessions=167))["gates"]["overfitting"]
    assert failed_closed(gate) and "167 sessions" in gate["reason"]


@pytest.mark.parametrize("strategies", [2, 9])
def test_pbo_is_inapplicable_to_fewer_than_ten_distinct_series(strategies):
    """Twins are one series: nine series and a twin are still nine."""
    gate = verdict(family(strategies=strategies, twins=[(V[39], V[1])]), trials=strategies + 1)["gates"]["overfitting"]
    assert failed_closed(gate) and f"{strategies} distinct return series, fewer than 10" in gate["reason"]


def test_ten_distinct_series_are_enough_for_pbo():
    assert "reason" not in verdict(family(strategies=10))["gates"]["overfitting"]


def test_the_haircut_is_annualised_by_the_familys_sessions_a_year():
    equity = verdict(family())["gates"]["haircut_sharpe"]
    crypto = verdict(family(), family_name="CRYPTO_SPOT_EOD")["gates"]["haircut_sharpe"]
    assert (equity["sessions_per_year"], crypto["sessions_per_year"]) == (252, 365)
    assert crypto["haircut_sharpe"] == equity["haircut_sharpe"]
    assert crypto["value_annual"] == equity["haircut_sharpe"] * 365 ** 0.5


def test_a_family_the_policy_does_not_list_fails_the_haircut():
    gate = verdict(family(), family_name="MOON_EOD")["gates"]["haircut_sharpe"]
    assert failed_closed(gate) and "'MOON_EOD'" in gate["reason"]


def test_twins_enter_the_haircut_as_unseen_trials():
    """A twin of the candidate (and of another variant) counts as a trial
    with p = 1. Giving it its column's p-value instead would lower the
    candidate's adjusted p-value, so the stored figure is the larger."""
    matrix = family(twins=[(V[38], CANDIDATE), (V[39], V[1])])
    gate = verdict(matrix, trials=12)["gates"]["haircut_sharpe"]
    columns = [c["returns"] for c in matrix["columns"]]
    index = trial_matrix.column_of(matrix, CANDIDATE)
    sharpe = validation_stats.sharpe(columns[index])
    p = {i: validation_stats.p_value(validation_stats.sharpe(c), 504) for i, c in enumerate(columns)}
    others = [p[i] for i in p if i != index]
    as_unseen = validation_stats.haircut(sharpe, 504, 12, "bhy", others)
    twin_p = [p[index], p[trial_matrix.column_of(matrix, V[1])]]
    as_copies = validation_stats.haircut(sharpe, 504, 12, "bhy", others + twin_p)
    assert gate["unseen_trials"] == 2 and gate["adjusted_p_value"] == as_unseen["adjusted_p_value"]
    assert as_copies["adjusted_p_value"] < gate["adjusted_p_value"]


def test_a_constant_variant_enters_the_haircut_with_p_one():
    """A version that never trades has a constant series and no Sharpe
    ratio. In the haircut it is unseen (p = 1); the DSR, which needs every
    series' Sharpe ratio, fails closed."""
    matrix = family()
    flat = trial_matrix.build(matrix["dates"], {**runs_of(matrix), V[20]: [0.0] * 504})
    gates = verdict(flat, trials=11)["gates"]
    assert "reason" not in gates["haircut_sharpe"]
    columns = [c["returns"] for c in flat["columns"]]
    index = trial_matrix.column_of(flat, CANDIDATE)
    flat_index = trial_matrix.column_of(flat, V[20])
    others = [validation_stats.p_value(validation_stats.sharpe(c), 504)
              for i, c in enumerate(columns) if i not in (index, flat_index)]
    expected = validation_stats.haircut(validation_stats.sharpe(columns[index]), 504, 11, "bhy", others + [1.0])
    assert gates["haircut_sharpe"]["adjusted_p_value"] == expected["adjusted_p_value"]
    assert failed_closed(gates["deflated_sharpe"]) and "constant" in gates["deflated_sharpe"]["reason"]


def test_a_constant_candidate_fails_every_statistical_gate_closed():
    matrix = family()
    flat = trial_matrix.build(matrix["dates"], {**runs_of(matrix), CANDIDATE: [0.0] * 504})
    gates = verdict(flat)["gates"]
    for name in ("deflated_sharpe", "haircut_sharpe", "t_statistic", "backtest_length"):
        assert failed_closed(gates[name]), name


@pytest.mark.parametrize("fields", [{"trials": None}, {"trials": 9}, {"trials": 10.0},
                                    {"every_trial_evaluated": None}])
def test_a_ledger_that_cannot_be_one_fails_the_gates_that_count_trials(fields):
    gates = evaluate(family(), CANDIDATE, {**ledger(10), **fields}, "EQUITY_EOD", LIMITS, RISK)["gates"]
    assert failed_closed(gates["deflated_sharpe"])
    if fields.get("trials") in (None, 10.0) and "trials" in fields:
        assert failed_closed(gates["haircut_sharpe"]) and failed_closed(gates["backtest_length"])


def test_a_candidate_missing_from_the_matrix_fails_every_statistical_gate():
    found = verdict(family(), candidate=V[30])
    assert found["passed"] is False and found["candidate"] == V[30]
    for name in validation_policy.STATISTICAL_GATES:
        assert failed_closed(found["gates"][name]) and "not in the matrix" in found["gates"][name]["reason"]


def test_the_verdict_uses_the_policy_it_is_given():
    loose = validation_policy.ValidationPolicy(**{**POLICY_V1.__dict__, "name": "loose", "pbo_min_strategies": 2,
                                                  "min_t_statistic_milli": 0})
    matrix = family(strategies=6)
    strict = verdict(matrix)
    relaxed = evaluate(matrix, CANDIDATE, ledger(10), "EQUITY_EOD", LIMITS, RISK, loose)
    assert failed_closed(strict["gates"]["overfitting"]) and "reason" not in relaxed["gates"]["overfitting"]
    assert relaxed["policy"] == policy_id(loose) != strict["policy"]


# --------------------------------------------------------------------------
# The job, end to end
# --------------------------------------------------------------------------

def request(evaluation, period="1y", initial_capital=100_000.0):
    return schemas.StrategyValidationRequest(evaluation=evaluation, period=period, initial_capital=initial_capital)


def validate(db, user, evaluation, job_process=None, **fields):
    return run_queued(db, main.validate_strategy(request(evaluation, **fields), user=user, db=db), job_process)


def rows(db, model):
    db.expire_all()
    return db.execute(select(model)).scalars().all()


def a_family(db, user):
    """A candidate, another version, and a twin of the candidate (it differs
    only in its declared drawdown limit)."""
    candidate, view = evaluated(db, user)
    datasets = view["result"]["market_data"]["datasets"]
    evaluated(db, user, strategy(**{"signal.fast": 10, "signal.slow": 40}), dataset_ids=datasets)
    evaluated(db, user, strategy(**{"risk_limits.max_drawdown_bps": 1_000}), dataset_ids=datasets)
    return candidate, view


def test_a_validation_judges_the_candidate_and_records_the_verdict_and_the_matrix(db, market):
    user = make_user(db)
    candidate, view = a_family(db, user)
    before = trial_count(db, user)

    done = validate(db, user, view["job_id"])
    assert done["status"] == "succeeded", done
    result = done["result"]
    found = result["validation"]
    assert (found["policy"], found["stage"], found["candidate"]) == (POLICY_V1_ID, "pre-holdout", candidate)
    # Two distinct series: PBO cannot be computed, so the verdict fails.
    assert found["passed"] is False
    assert "2 distinct return series, fewer than 10" in found["gates"]["overfitting"]["reason"]
    assert "matrix" not in result and "risk" not in result
    assert result["ledger"]["trials"] == 3 and trial_count(db, user) == before == 3

    [row] = rows(db, models.StrategyValidation)
    assert (row.job_id, row.user_id, row.evaluation_job_id, row.strategy_version_id) == \
        (done["job_id"], user.id, view["job_id"], candidate)
    assert (row.family_hash, row.matrix_id, row.policy_id, row.passed) == \
        (FAMILY, result["matrix_id"], POLICY_V1_ID, False)
    assert row.policy_canonical == validation_policy.policy_canonical(POLICY_V1)
    assert row.verdict == found

    [stored] = rows(db, models.TrialMatrix)
    assert stored.matrix_id == result["matrix_id"]
    assert stored.matrix_id == f"qs-tm-sha256-v1:{hashlib.sha256(stored.canonical).hexdigest()}"
    matrix = strategy_validation.matrix_of(db, row)
    assert trial_matrix.matrix_id(matrix) == matrix["id"] == row.matrix_id
    assert len(matrix["columns"]) == 2 and len(matrix["dates"]) == result["sessions"]["count"]

    # The worker's verdict is the policy's on the stored matrix, the ledger
    # and the candidate's spec.
    spec = strategy_registry.spec_of(strategy_registry.get_for_user(db, user.id, candidate))
    again = evaluate(matrix, candidate, result["ledger"], spec.universe.family, spec.risk_limits, None)
    assert {k: g for k, g in again["gates"].items() if k != "risk_limits"} == \
        {k: g for k, g in found["gates"].items() if k != "risk_limits"}
    assert found["gates"]["risk_limits"]["checks"]["drawdown_within_declared"]["value_bps"] is not None

    [audit] = db.execute(select(models.AuditLog).where(models.AuditLog.action == "STRATEGY_VALIDATION")).scalars()
    assert (audit.resource_type, audit.resource_id) == ("strategy_version", candidate)
    assert audit.metadata_json["policy"] == POLICY_V1_ID and audit.metadata_json["passed"] is False
    assert "overfitting" in audit.metadata_json["failed_gates"]
    assert audit.metadata_json["matrix_id"] == row.matrix_id


@pytest.mark.parametrize("first", [True, False])
def test_the_risk_gate_judges_the_candidates_own_re_run(db, market, first):
    """Every version re-runs; the risk measurements are the candidate's,
    whichever comes first in the job's order (by version id)."""
    user = make_user(db)
    a, view_a = evaluated(db, user, strategy(**{"sizing.fraction_bps": 1_000}))
    b, view_b = evaluated(db, user, strategy(**{"sizing.fraction_bps": 2_400}),
                          dataset_ids=view_a["result"]["market_data"]["datasets"])
    views = {a: view_a, b: view_b}
    candidate = sorted(views)[0 if first else 1]
    view = views[candidate]
    assert view_a["result"]["risk"] != view_b["result"]["risk"]
    done = validate(db, user, view["job_id"])
    assert done["status"] == "succeeded", done
    checks = done["result"]["validation"]["gates"]["risk_limits"]["checks"]
    assert checks["drawdown_within_declared"]["value_bps"] == view["result"]["risk"]["max_drawdown"] * 10_000
    assert checks["position_within_declared"]["value_bps"] == view["result"]["risk"]["max_position_share"] * 10_000


def test_the_verdicts_can_be_read_by_their_owner_only(db, market):
    owner, other = make_user(db), make_user(db, "other@example.com")
    _, view = evaluated(db, owner)
    first = validate(db, owner, view["job_id"])
    second = validate(db, owner, view["job_id"])
    assert first["status"] == second["status"] == "succeeded"
    listed = main.list_strategy_validations(limit=20, user=owner, db=db)
    assert {v["job_id"] for v in listed} == {first["job_id"], second["job_id"]}
    assert listed[0]["policy"]["id"] == POLICY_V1_ID and listed[0]["policy"]["min_dsr_bps"] == 9_500
    one = main.get_strategy_validation(first["job_id"], user=owner, db=db)
    assert one["verdict"] == first["result"]["validation"] and one["evaluation"] == view["job_id"]
    assert main.get_strategy_validation_matrix(first["job_id"], user=owner, db=db)["id"] == one["matrix_id"]
    # The same matrix is stored once.
    assert first["result"]["matrix_id"] == second["result"]["matrix_id"]
    assert len(rows(db, models.TrialMatrix)) == 1 and len(rows(db, models.StrategyValidation)) == 2
    assert main.list_strategy_validations(limit=20, user=other, db=db) == []
    for read in (main.get_strategy_validation, main.get_strategy_validation_matrix):
        for user, job_id in ((other, first["job_id"]), (owner, "no-such-job")):
            with pytest.raises(HTTPException) as caught:
                read(job_id, user=user, db=db)
            assert caught.value.status_code == 404


def test_only_the_callers_evaluation_with_the_period_and_capital_it_ran_with_can_be_validated(db, market):
    owner, other = make_user(db), make_user(db, "other@example.com")
    _, view = evaluated(db, owner)
    for user, evaluation, fields, status in (
            (other, view["job_id"], {}, 404),
            (owner, "no-such-job", {}, 404),
            (owner, view["job_id"], {"period": "6mo"}, 422),
            (owner, view["job_id"], {"initial_capital": 100_001.0}, 422)):
        with pytest.raises(HTTPException) as caught:
            main.validate_strategy(request(evaluation, **fields), user=user, db=db)
        assert caught.value.status_code == status
    assert db.execute(select(func.count()).select_from(models.ResearchJob)
                      .where(models.ResearchJob.kind == "strategy_validation")).scalar() == 0


def _passing(result):
    result["validation"] = {"policy": POLICY_V1_ID, "passed": True, "gates": {}}
    result["passed"] = True


def test_the_worker_keeps_its_own_verdict_not_the_job_processs(db, market):
    user = make_user(db)
    _, view = evaluated(db, user)
    done = validate(db, user, view["job_id"], MisreportsItsResult(_passing))
    assert done["status"] == "succeeded", done
    assert done["result"]["validation"]["passed"] is False
    [row] = rows(db, models.StrategyValidation)
    assert row.passed is False and row.verdict == done["result"]["validation"]
    assert set(row.verdict["gates"]) == {"risk_limits", *validation_policy.STATISTICAL_GATES}


def test_risk_measurements_the_job_process_leaves_out_fail_the_risk_gate(db, market):
    user = make_user(db)
    _, view = evaluated(db, user)
    done = validate(db, user, view["job_id"], MisreportsItsResult(lambda r: r.pop("risk")))
    assert done["status"] == "succeeded", done
    checks = done["result"]["validation"]["gates"]["risk_limits"]["checks"]
    assert checks["drawdown_within_declared"]["passed"] is False
    assert "not measured" in checks["drawdown_within_declared"]["description"]


def _other_candidate_series(result):
    """A well-formed matrix in which the candidate's column is not its
    evaluation's series (rehashed, reordered and with a recomputed id), so
    only the worker's comparison with the evaluation can refuse it."""
    matrix, version = result["matrix"], result["candidate"]["strategy_version_id"]
    column = matrix["columns"][trial_matrix.column_of(matrix, version)]
    column["returns"][-1] += 1
    column["returns_sha256"] = research_trials.quantised_returns_id(column["returns"])
    matrix["columns"].sort(key=lambda c: c["returns_sha256"])
    matrix["id"] = trial_matrix.matrix_id(matrix)


def test_the_worker_refuses_a_matrix_whose_candidate_is_not_its_evaluation(db, market):
    user = make_user(db)
    _, view = evaluated(db, user)
    done = validate(db, user, view["job_id"], MisreportsItsResult(_other_candidate_series))
    assert done["status"] == "failed"
    assert done["error"] == {"status_code": 500,
                             "detail": "The research job reported a matrix in which the candidate is not its evaluation"}
    assert rows(db, models.StrategyValidation) == [] and rows(db, models.TrialMatrix) == []


def test_the_worker_refuses_an_invalid_matrix_and_records_nothing(db, market):
    user = make_user(db)
    _, view = evaluated(db, user)
    done = validate(db, user, view["job_id"],
                    MisreportsItsResult(lambda r: r["matrix"].update(id=trial_matrix.MATRIX_SCHEME + ":" + "0" * 64)))
    assert done["status"] == "failed"
    assert done["error"] == {"status_code": 500, "detail": "The research job reported an invalid matrix"}
    assert rows(db, models.StrategyValidation) == [] and rows(db, models.TrialMatrix) == []


def test_a_candidate_spec_the_registry_can_no_longer_read_fails_the_job(db, market, monkeypatch):
    user = make_user(db)
    candidate, view = evaluated(db, user)
    calls = []
    spec_of = strategy_registry.spec_of

    def refuse_in_finalize(version):
        calls.append(version.version_id)
        if len(calls) > 1:  # matrix_inputs reads it first
            raise strategy_registry.StrategyRegistryError(f"{candidate}: the stored spec no longer validates")
        return spec_of(version)

    monkeypatch.setattr(strategy_registry, "spec_of", refuse_in_finalize)
    done = validate(db, user, view["job_id"])
    assert done["status"] == "failed" and done["error"]["status_code"] == 422
    assert candidate in done["error"]["detail"]
    assert rows(db, models.StrategyValidation) == []


def test_a_candidate_version_no_longer_held_fails_the_job(db, market, monkeypatch):
    user = make_user(db)
    candidate, view = evaluated(db, user)
    calls = []
    get_for_user = strategy_registry.get_for_user

    def gone_in_finalize(session, user_id, version_id):
        calls.append(version_id)
        if len(calls) > 1:  # matrix_inputs reads it first
            return None
        return get_for_user(session, user_id, version_id)

    monkeypatch.setattr(strategy_registry, "get_for_user", gone_in_finalize)
    done = validate(db, user, view["job_id"])
    assert len(calls) == 2
    assert done["status"] == "failed"
    assert done["error"] == {"status_code": 422, "detail": f"The candidate's strategy version {candidate} is not held"}
    assert rows(db, models.StrategyValidation) == [] and rows(db, models.TrialMatrix) == []


# --------------------------------------------------------------------------
# The tables (migration 0008)
# --------------------------------------------------------------------------

@pytest.fixture
def stored(db, market):
    user = make_user(db)
    _, view = evaluated(db, user)
    done = validate(db, user, view["job_id"])
    assert done["status"] == "succeeded", done
    [row] = rows(db, models.StrategyValidation)
    return row


def _copy(db, row, **changes):
    values = {c.name: getattr(row, c.name) for c in models.StrategyValidation.__table__.columns}
    db.add(models.StrategyValidation(**{**values, "job_id": "another-job", **changes}))
    db.flush()


@pytest.mark.parametrize("change, constraint", [
    ("policy_id", "ck_strategy_validations_policy_id_is_hash"),
    ("policy_canonical", "ck_strategy_validations_policy_id_is_hash"),
    ("passed", "ck_strategy_validations_verdict_matches"),
    ("verdict_policy", "ck_strategy_validations_verdict_matches"),
    ("verdict_candidate", "ck_strategy_validations_verdict_matches"),
    ("verdict_list", "ck_strategy_validations_verdict_matches"),
])
def test_the_schema_refuses_a_verdict_that_does_not_match_its_row(db, stored, change, constraint):
    changes = {"policy_id": {"policy_id": "qs-vp-sha256-v1:" + "0" * 64},
               "policy_canonical": {"policy_canonical": b"{}"},
               "passed": {"passed": True},
               "verdict_policy": {"verdict": {**stored.verdict, "policy": "qs-vp-sha256-v1:" + "0" * 64}},
               "verdict_candidate": {"verdict": {**stored.verdict, "candidate": "qs-sv-sha256-v1:" + "0" * 64}},
               "verdict_list": {"verdict": [stored.verdict]}}[change]
    values = {c.name: getattr(stored, c.name) for c in models.StrategyValidation.__table__.columns}
    with pytest.raises(DBAPIError, match=constraint):
        _copy(db, stored, **changes)
    db.rollback()
    db.add(models.StrategyValidation(**{**values, "job_id": "another-job"}))  # an unchanged copy is accepted
    db.flush()
    db.rollback()


def test_the_schema_refuses_a_matrix_stored_under_another_id(db, stored):
    with pytest.raises(DBAPIError, match="ck_trial_matrices_id_is_hash"):
        db.add(models.TrialMatrix(matrix_id="qs-tm-sha256-v1:" + "0" * 64, canonical=b"{}"))
        db.flush()
    db.rollback()


@pytest.mark.parametrize("statement", [
    "UPDATE strategy_validations SET passed = true", "DELETE FROM strategy_validations",
    "TRUNCATE strategy_validations", "UPDATE trial_matrices SET canonical = canonical",
    "DELETE FROM trial_matrices", "TRUNCATE trial_matrices CASCADE",
])
def test_verdicts_and_matrices_are_immutable_even_for_the_owner(db, stored, statement):
    with pytest.raises(DBAPIError, match="its rows are immutable"):
        db.execute(text(statement))
    db.rollback()
    assert db.execute(text("SELECT count(*) FROM strategy_validations")).scalar() == 1
    assert db.execute(text("SELECT count(*) FROM trial_matrices")).scalar() == 1
