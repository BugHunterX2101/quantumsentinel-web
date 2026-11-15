"""The validation policy (services/validation_policy.py) and the risk
measurements the event engine reports for its risk gate."""
import hashlib
import math

import numpy as np
import pytest

from backend.services import strategy_spec, validation_policy
from backend.services.event_simulator import run_event_backtest
from backend.services.strategy_signals import ma_crossover_series
from backend.services.validation_policy import POLICY_V1, ValidationPolicy, policy_id, risk_gate
from test_strategy_spec import prices, spec

# --------------------------------------------------------------------------
# The policy and its id
# --------------------------------------------------------------------------

POLICY_V1_CANONICAL = (b'{"max_declared_drawdown_bps":2500,"max_declared_position_bps":2500,'
                       b'"name":"qs-validation-policy-v1"}')
POLICY_V1_ID = "qs-vp-sha256-v1:d2320ec069c167cef0629c93e82014e1eec25aa8be10658bb1a44890b8b1a0b7"


def test_policy_v1_has_its_golden_id():
    """Pinned, and recomputed here from the canonical JSON written out by
    hand, so a verdict naming the id can be checked by another
    implementation."""
    assert POLICY_V1_ID == f"qs-vp-sha256-v1:{hashlib.sha256(POLICY_V1_CANONICAL).hexdigest()}"
    assert policy_id(POLICY_V1) == POLICY_V1_ID
    assert validation_policy.POLICY_ID.fullmatch(POLICY_V1_ID)
    assert (POLICY_V1.max_declared_drawdown_bps, POLICY_V1.max_declared_position_bps) == (2_500, 2_500)


@pytest.mark.parametrize("field", ["name", "max_declared_drawdown_bps", "max_declared_position_bps"])
def test_every_field_is_part_of_the_id(field):
    changed = {"name": "other", "max_declared_drawdown_bps": 2_499, "max_declared_position_bps": 2_499}
    other = ValidationPolicy(**{**POLICY_V1.__dict__, field: changed[field]})
    assert policy_id(other) != POLICY_V1_ID


@pytest.mark.parametrize("changes", [
    {"name": ""}, {"name": 1},
    {"max_declared_drawdown_bps": 0}, {"max_declared_drawdown_bps": 10_001},
    {"max_declared_drawdown_bps": 2_500.0}, {"max_declared_drawdown_bps": True},
    {"max_declared_position_bps": -1}, {"max_declared_position_bps": "2500"},
])
def test_a_policy_refuses_thresholds_that_are_not_basis_points(changes):
    with pytest.raises(ValueError):
        ValidationPolicy(**{**POLICY_V1.__dict__, **changes})


def test_a_policy_is_frozen():
    with pytest.raises(AttributeError):
        POLICY_V1.max_declared_drawdown_bps = 10_000  # type: ignore[misc]


# --------------------------------------------------------------------------
# The engine's risk measurements
# --------------------------------------------------------------------------

def test_without_the_flag_the_engine_reports_exactly_what_it_did_before():
    compiled = strategy_spec.compile_spec(strategy_spec.parse(spec(**{"signal.fast": 5, "signal.slow": 20})))
    data = prices(compiled.tickers, bars=400)
    plain = compiled.run(data, 100_000.0, full_returns=True)
    measured = compiled.run(data, 100_000.0, full_returns=True, risk_metrics=True)
    assert "risk" not in plain
    risk = measured.pop("risk")
    assert set(risk) == {"max_drawdown", "max_position_share", "equity_exhausted"}
    np.testing.assert_array_equal(measured.pop("returns"), plain.pop("returns"))
    assert measured == plain


def _reference(closes, fast, slow, fraction, capital):
    """Equity, the largest position's share of it and the gross exposure's
    share of it at each bar, worked out independently of the engine for a
    long-only, zero-cost crossover on each ticker: a signal at bar s fills at
    bar s + 1's close; a long entry is sized at ``fraction`` of the equity
    marked at bar s - 1 (the last mark when bar s's signals are sized) at
    bar s's price, and a cross down closes it."""
    tickers = sorted(closes)
    signals = {t: ma_crossover_series(closes[t], fast, slow) for t in tickers}
    cash, held, staged = capital, dict.fromkeys(tickers, 0.0), {}
    last_mark = capital
    equity, largest, gross = [], [], []
    for bar in range(len(closes[tickers[0]])):
        price = {t: closes[t][bar] for t in tickers}
        for t, qty in staged.items():
            cash -= qty * price[t]
            held[t] += qty
        staged = {}
        for t in tickers:
            if signals[t][bar] > 0 and held[t] == 0:
                staged[t] = fraction * last_mark / price[t]
            elif signals[t][bar] < 0 and held[t] > 0:
                staged[t] = -held[t]
        values = [abs(held[t]) * price[t] for t in tickers]
        last_mark = cash + sum(held[t] * price[t] for t in tickers)
        equity.append(last_mark)
        largest.append(max(values) / last_mark)
        gross.append(sum(values) / last_mark)
    return np.array(equity), np.array(largest), np.array(gross)


CASES = {
    "one ticker": {"X": [100.0] * 8 + [110, 120, 130, 125, 90, 80, 85, 95, 100, 70]},
    "one ticker, re-entered": {
        "X": [100.0] * 8 + [104, 112, 121, 133, 128, 140, 101, 97, 99, 108, 117, 126, 90, 70, 75]},
    "one ticker, a new high after the deepest fall": {
        "X": [100.0] * 8 + [110, 120, 130, 100, 85, 95, 110, 130, 150, 170, 190, 210, 230, 250]},
    "two tickers held together": {
        "X": [100.0] * 8 + [110, 120, 130, 125, 90, 80, 85, 95, 100, 70],
        "Y": [100.0] * 8 + [105, 115, 140, 150, 145, 130, 100, 90, 95, 99]},
}


@pytest.mark.parametrize("label", CASES)
def test_the_measurements_match_an_independent_reference(label):
    closes = {t: np.array(c, dtype=float) for t, c in CASES[label].items()}
    fraction = 0.5 if len(closes) == 1 else 0.3
    result = run_event_backtest(sorted(closes), closes, "ma_crossover", {"fast": 2, "slow": 5, "fraction": fraction},
                                initial_capital=100_000.0, cost_model_name="zero_cost", risk_metrics=True)
    equity, largest, gross = _reference(closes, 2, 5, fraction, 100_000.0)
    assert result["n_trades"] >= 2 * len(closes)  # every ticker entered and left
    expected_drawdown = float(np.max(1 - equity / np.maximum.accumulate(equity)))
    assert expected_drawdown > 0
    if "new high" in label:
        # A later, higher peak: measuring from it would overstate the fall.
        assert equity.argmax() > np.argmax(1 - equity / np.maximum.accumulate(equity))
        assert float(np.max(1 - equity / equity.max())) > expected_drawdown + 0.05
    assert result["risk"]["max_drawdown"] == pytest.approx(expected_drawdown, rel=1e-12)
    assert result["risk"]["max_position_share"] == pytest.approx(float(largest.max()), rel=1e-12)
    assert result["risk"]["equity_exhausted"] is False
    if len(closes) > 1:  # the largest single position, not the gross exposure
        assert largest.max() < gross.max() - 0.05
    # Unrounded, unlike the engine's displayed max_drawdown.
    assert result["max_drawdown"] == round(-expected_drawdown, 4)


def test_the_first_case_by_hand():
    """Entry signalled at bar 8 (110), filled at bar 9 (120) with
    0.5 * 100,000 / 110 shares; at 130 the position is 59,090.9 of
    104,545.5 equity."""
    close = np.array([100.0] * 8 + [110, 120, 130, 125, 90, 80, 85, 95, 100, 70])
    result = run_event_backtest(["X"], {"X": close}, "ma_crossover", {"fast": 2, "slow": 5, "fraction": 0.5},
                                initial_capital=100_000.0, cost_model_name="zero_cost", risk_metrics=True)
    qty = 0.5 * 100_000 / 110
    cash = 100_000 - qty * 120
    assert result["risk"]["max_position_share"] == pytest.approx(qty * 130 / (cash + qty * 130), rel=1e-12)


def test_a_short_that_loses_everything_is_reported_as_exhausting_equity():
    """A full-size short into a steady climb: equity crosses zero once the
    price doubles past the entry. Shares of non-positive equity are not
    measured; the drawdown keeps counting past 100%."""
    close = 100.0 + 5.0 * np.arange(120)
    result = run_event_backtest(["X"], {"X": close}, "mean_reversion",
                                {"window": 20, "n_std": 1.0, "fraction": 1.0, "leverage_limit": 5.0},
                                initial_capital=100_000.0, cost_model_name="zero_cost", allow_short=True,
                                risk_metrics=True)
    risk = result["risk"]
    assert risk["equity_exhausted"] is True
    assert risk["max_drawdown"] > 1.0
    assert math.isfinite(risk["max_position_share"]) and risk["max_position_share"] > 1.0


# --------------------------------------------------------------------------
# The risk gate
# --------------------------------------------------------------------------

LIMITS = strategy_spec.RiskLimits(max_drawdown_bps=2_000, max_position_bps=2_500)
CLEAN = {"max_drawdown": 0.1, "max_position_share": 0.2, "equity_exhausted": False}


def test_a_backtest_within_its_declared_limits_passes():
    verdict = risk_gate(LIMITS, CLEAN)
    assert verdict["passed"] is True and verdict["gate"] == "risk_limits"
    assert verdict["policy"] == POLICY_V1_ID
    assert all(check["passed"] for check in verdict["checks"].values())
    assert verdict["checks"]["drawdown_within_declared"]["value_bps"] == pytest.approx(1_000)


def test_a_measurement_equal_to_its_limit_passes_and_just_above_fails():
    at = {**CLEAN, "max_drawdown": 0.2, "max_position_share": 0.25}
    assert risk_gate(LIMITS, at)["passed"] is True
    for key in ("max_drawdown", "max_position_share"):
        over = {**at, key: at[key] + 1e-9}
        assert risk_gate(LIMITS, over)["passed"] is False


@pytest.mark.parametrize("measurement, failing", [
    ({"max_drawdown": 0.21}, "drawdown_within_declared"),
    ({"max_position_share": 0.26}, "position_within_declared"),
    ({"equity_exhausted": True}, "equity_never_exhausted"),
])
def test_each_breach_alone_fails_the_gate(measurement, failing):
    verdict = risk_gate(LIMITS, {**CLEAN, **measurement})
    assert verdict["passed"] is False
    assert [name for name, check in verdict["checks"].items() if not check["passed"]] == [failing]


def test_a_declaration_at_the_policys_cap_passes():
    at_cap = strategy_spec.RiskLimits(max_drawdown_bps=2_500, max_position_bps=2_500)
    assert risk_gate(at_cap, CLEAN)["passed"] is True


@pytest.mark.parametrize("limits, failing", [
    (strategy_spec.RiskLimits(max_drawdown_bps=2_501, max_position_bps=2_500), "declared_drawdown_within_policy"),
    (strategy_spec.RiskLimits(max_drawdown_bps=2_000, max_position_bps=10_000), "declared_position_within_policy"),
])
def test_a_declaration_beyond_the_policys_cap_fails_however_small_the_risk_taken(limits, failing):
    verdict = risk_gate(limits, {"max_drawdown": 0.0, "max_position_share": 0.0, "equity_exhausted": False})
    assert verdict["passed"] is False
    assert [name for name, check in verdict["checks"].items() if not check["passed"]] == [failing]


_UNMEASURED = {"drawdown_within_declared", "position_within_declared", "equity_never_exhausted"}


@pytest.mark.parametrize("risk, failing", [
    (None, _UNMEASURED),
    ({}, _UNMEASURED),
    ([0.1, 0.2, False], _UNMEASURED),
    ({**CLEAN, "max_drawdown": math.nan}, {"drawdown_within_declared"}),
    ({**CLEAN, "max_drawdown": math.inf}, {"drawdown_within_declared"}),
    ({**CLEAN, "max_drawdown": -0.1}, {"drawdown_within_declared"}),
    ({**CLEAN, "max_drawdown": "0.1"}, {"drawdown_within_declared"}),
    ({**CLEAN, "max_drawdown": None}, {"drawdown_within_declared"}),
    ({**CLEAN, "max_position_share": math.nan}, {"position_within_declared"}),
    ({**CLEAN, "max_position_share": False}, {"position_within_declared"}),
    ({k: v for k, v in CLEAN.items() if k != "max_position_share"}, {"position_within_declared"}),
    ({**CLEAN, "equity_exhausted": 0}, {"equity_never_exhausted"}),
    ({**CLEAN, "equity_exhausted": "false"}, {"equity_never_exhausted"}),
    ({k: v for k, v in CLEAN.items() if k != "equity_exhausted"}, {"equity_never_exhausted"}),
])
def test_a_measurement_that_is_missing_or_not_a_share_fails_closed(risk, failing):
    verdict = risk_gate(LIMITS, risk)
    assert verdict["passed"] is False
    assert {name for name, check in verdict["checks"].items() if not check["passed"]} == failing
    for name in failing:
        assert "not measured" in verdict["checks"][name]["description"]


def test_the_gate_uses_the_policy_it_is_given():
    loose = ValidationPolicy(name="loose", max_declared_drawdown_bps=10_000, max_declared_position_bps=10_000)
    limits = strategy_spec.RiskLimits(max_drawdown_bps=5_000, max_position_bps=5_000)
    assert risk_gate(limits, CLEAN)["passed"] is False
    verdict = risk_gate(limits, CLEAN, loose)
    assert verdict["passed"] is True and verdict["policy"] == policy_id(loose)
