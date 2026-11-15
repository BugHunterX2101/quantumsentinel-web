"""QuantumSentinel — the validation policy (R3).

A strategy version earns a place in paper trading only by passing every gate
of a validation policy. This module holds the policy and its gates.

* A policy is data: frozen, integer thresholds (fractions in basis points,
  1/10,000). Its id is ``qs-vp-sha256-v1:<hex>``, the SHA-256 of its canonical
  JSON (``dataset_format.canonical_json``), so a verdict that names the
  policy id stays tied to the exact thresholds it was reached under, and a
  later policy never rewrites an earlier verdict.
* Gates are pure functions of their inputs: no database, no clock, no
  randomness. The same inputs always give the same verdict.
* Every gate fails closed. A gate that cannot be evaluated (a measurement
  missing, not finite, or of the wrong type) fails; it never passes by
  default.

The risk gate (``risk_gate``) checks a backtest against the limits its spec
declares (``strategy_spec.RiskLimits``), which nothing enforces while the
backtest runs:

* the declared limits must themselves be within the policy's caps. A spec may
  declare a drawdown limit of up to 100%, which any backtest would meet, so
  the policy bounds what a declaration may claim;
* the backtest's largest drawdown and largest single position must not
  exceed the declared limits (equal to the limit passes);
* equity must never have been exhausted (zero or negative), where a
  position's share of equity has no meaning.

The measurements come from the event engine (``run_event_backtest`` with
``risk_metrics``).

The statistical gates (``evaluate``) judge the candidate against every
variant tried in its research family, from the trial matrix
(``trial_matrix``) and the statistics of ``validation_stats``:

* ``deflated_sharpe``: the deflated Sharpe ratio is at least ``min_dsr_bps``;
* ``overfitting``: the probability of backtest overfitting (CSCV) is at most
  ``max_pbo_bps``, from at least ``pbo_min_strategies`` distinct return
  series, with the first of ``pbo_splits`` whose blocks hold at least
  ``pbo_min_block_sessions`` sessions;
* ``haircut_sharpe``: the Sharpe ratio left after the ``haircut_method``
  adjustment for the family's trials is at least
  ``min_haircut_sharpe_annual_milli`` / 1000 a year;
* ``t_statistic``: the candidate's t-statistic (SR sqrt(T)) is at least
  ``min_t_statistic_milli`` / 1000;
* ``backtest_length``: the backtest is at least the minimum backtest length
  for the family's trials.

A Sharpe ratio is annualised only to compare it with a yearly threshold,
with the sessions a year of the strategy's dataset family
(``sessions_per_year``); a family the policy does not list fails the gates
that need it. A verdict is pre-holdout: it says whether the candidate may go
on to the one-shot holdout evaluation, not that it passed it. Cost stress is
not part of this policy.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .dataset_format import canonical_json
from .strategy_spec import RiskLimits

POLICY_SCHEME = "qs-vp-sha256-v1"
POLICY_ID = re.compile(rf"{POLICY_SCHEME}:[0-9a-f]{{64}}")
_BPS = 10_000


# The kind of each threshold: (smallest, largest) integer.
_INTEGER_FIELDS = {
    "max_declared_drawdown_bps": (1, _BPS),
    "max_declared_position_bps": (1, _BPS),
    "min_dsr_bps": (1, _BPS),
    "max_pbo_bps": (0, _BPS),
    "pbo_min_strategies": (2, 1_000),
    "pbo_min_block_sessions": (2, 10_000),
    "min_haircut_sharpe_annual_milli": (0, 100_000),
    "min_t_statistic_milli": (0, 100_000),
}


def _integer(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class ValidationPolicy:
    name: str
    # The largest drawdown and single position a strategy may declare, as
    # shares of equity.
    max_declared_drawdown_bps: int
    max_declared_position_bps: int
    # The deflated Sharpe ratio's floor, as a probability.
    min_dsr_bps: int
    # The probability of backtest overfitting's ceiling, from at least this
    # many distinct return series, with the first number of splits (even,
    # largest first) whose blocks hold at least this many sessions.
    max_pbo_bps: int
    pbo_min_strategies: int
    pbo_splits: tuple[int, ...]
    pbo_min_block_sessions: int
    # The haircut Sharpe ratio's floor, a year (thousandths), after the
    # multiple-testing adjustment named here (validation_stats.METHODS).
    haircut_method: str
    min_haircut_sharpe_annual_milli: int
    # The t-statistic's floor (thousandths).
    min_t_statistic_milli: int
    # Sessions a year by dataset family, as (family, sessions) pairs ordered
    # by family.
    sessions_per_year: tuple[tuple[str, int], ...]

    def __post_init__(self):
        from .validation_stats import METHODS

        if not isinstance(self.name, str) or not self.name:
            raise ValueError("a policy needs a name")
        for field, (low, high) in _INTEGER_FIELDS.items():
            value = getattr(self, field)
            if not _integer(value) or not low <= value <= high:
                raise ValueError(f"{field} must be an integer in {low}..{high}")
        splits = self.pbo_splits
        if (not isinstance(splits, tuple) or not splits
                or not all(_integer(s) and s >= 2 and s % 2 == 0 for s in splits)
                or list(splits) != sorted(set(splits), reverse=True)):
            raise ValueError("pbo_splits are distinct even integers of at least 2, largest first")
        if self.haircut_method not in METHODS:
            raise ValueError(f"haircut_method is one of {', '.join(METHODS)}")
        pairs = self.sessions_per_year
        if (not isinstance(pairs, tuple) or not pairs
                or not all(isinstance(p, tuple) and len(p) == 2 and isinstance(p[0], str) and p[0]
                           and _integer(p[1]) and 1 <= p[1] <= 366 for p in pairs)
                or [p[0] for p in pairs] != sorted({p[0] for p in pairs})):
            raise ValueError("sessions_per_year is (family, sessions in 1..366) pairs, one per family, in order")


POLICY_V1 = ValidationPolicy(
    name="qs-validation-policy-v1",
    max_declared_drawdown_bps=2_500,
    max_declared_position_bps=2_500,
    min_dsr_bps=9_500,
    max_pbo_bps=500,
    pbo_min_strategies=10,
    pbo_splits=(16, 12, 8),
    pbo_min_block_sessions=21,
    haircut_method="bhy",
    min_haircut_sharpe_annual_milli=500,
    min_t_statistic_milli=3_000,
    # Crypto trades every day; every other family on exchange sessions.
    sessions_per_year=(("CRYPTO_SPOT_EOD", 365), ("EQUITY_EOD", 252), ("FUND_NAV_EOD", 252),
                       ("FUTURES_CONTINUOUS_EOD", 252), ("FX_SPOT_EOD", 252), ("INDEX_EOD", 252)),
)


def policy_canonical(policy: ValidationPolicy) -> bytes:
    """The canonical JSON of the policy, which its id hashes."""
    return canonical_json(asdict(policy))


def policy_id(policy: ValidationPolicy) -> str:
    return f"{POLICY_SCHEME}:{hashlib.sha256(policy_canonical(policy)).hexdigest()}"


# --------------------------------------------------------------------------
# Risk gate
# --------------------------------------------------------------------------

def _share(value: Any) -> float | None:
    """A measured share of equity, or None if it is not one."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and value >= 0 else None


def _within(measured: float | None, limit_bps: int, description: str) -> dict:
    if measured is None:
        return {"passed": False, "value_bps": None, "limit_bps": limit_bps,
                "description": f"{description} (not measured, so the check fails)"}
    return {"passed": measured <= limit_bps / _BPS, "value_bps": measured * _BPS,
            "limit_bps": limit_bps, "description": description}


def risk_gate(limits: RiskLimits, risk: Mapping[str, Any] | None,
              policy: ValidationPolicy = POLICY_V1) -> dict:
    """Check a backtest's risk measurements (``risk`` from the event engine)
    against the limits its spec declares and the policy's caps on them."""
    risk = risk if isinstance(risk, Mapping) else {}
    exhausted = risk.get("equity_exhausted")
    measured = isinstance(exhausted, bool)
    checks = {
        "declared_drawdown_within_policy": {
            "passed": limits.max_drawdown_bps <= policy.max_declared_drawdown_bps,
            "value_bps": limits.max_drawdown_bps, "limit_bps": policy.max_declared_drawdown_bps,
            "description": "The declared drawdown limit is within the policy's cap"},
        "declared_position_within_policy": {
            "passed": limits.max_position_bps <= policy.max_declared_position_bps,
            "value_bps": limits.max_position_bps, "limit_bps": policy.max_declared_position_bps,
            "description": "The declared position limit is within the policy's cap"},
        "drawdown_within_declared": _within(
            _share(risk.get("max_drawdown")), limits.max_drawdown_bps,
            "The backtest's largest drawdown from a peak is within the declared limit"),
        "position_within_declared": _within(
            _share(risk.get("max_position_share")), limits.max_position_bps,
            "The backtest's largest single position, as a share of equity, is within the declared limit"),
        "equity_never_exhausted": {
            "passed": exhausted is False,
            "value": exhausted if measured else None,
            "description": "Equity never reached zero or below"
                           + ("" if measured else " (not measured, so the check fails)")},
    }
    return {"gate": "risk_limits", "policy": policy_id(policy),
            "passed": all(check["passed"] for check in checks.values()), "checks": checks}


# --------------------------------------------------------------------------
# Statistical gates
# --------------------------------------------------------------------------

STATISTICAL_GATES = ("deflated_sharpe", "overfitting", "haircut_sharpe", "t_statistic", "backtest_length")


def _refused(gate: dict, reason) -> dict:
    return {**gate, "passed": False, "reason": f"not evaluated, so the gate fails: {reason}"}


def _sessions_per_year(policy: ValidationPolicy, family: str) -> int:
    from .validation_stats import StatsError

    for listed, sessions in policy.sessions_per_year:
        if listed == family:
            return sessions
    raise StatsError(f"the policy gives no sessions a year for the {family!r} dataset family")


def deflated_sharpe_gate(columns, candidate: int, ledger: Mapping[str, Any], policy: ValidationPolicy) -> dict:
    """The candidate's deflated Sharpe ratio (validation_stats.deflated_sharpe)
    is at least the policy's floor."""
    from . import validation_stats as vs

    gate = {"minimum": policy.min_dsr_bps / _BPS,
            "description": "The deflated Sharpe ratio, after the family's trials, is at least the minimum"}
    try:
        found = vs.deflated_sharpe(columns, candidate, ledger.get("trials"), ledger.get("every_trial_evaluated"))
    except vs.StatsError as exc:
        return _refused(gate, exc)
    return {**gate, "passed": found["dsr"] >= gate["minimum"], "value": found["dsr"],
            **{key: found[key] for key in ("sharpe", "benchmark_sharpe", "n_trials", "n_trials_source",
                                           "ledger_trials", "distinct_series", "average_correlation",
                                           "variance", "variance_null", "variance_empirical",
                                           "skewness", "kurtosis")}}


def overfitting_gate(columns, policy: ValidationPolicy) -> dict:
    """The probability of backtest overfitting of the family's choice
    (validation_stats.pbo_cscv) is at most the policy's ceiling, from at
    least the policy's number of distinct return series, cut into the most
    blocks (the first of ``pbo_splits``) that are each long enough."""
    from . import validation_stats as vs

    gate = {"maximum": policy.max_pbo_bps / _BPS, "min_strategies": policy.pbo_min_strategies,
            "min_block_sessions": policy.pbo_min_block_sessions, "strategies": len(columns),
            "description": "The probability of backtest overfitting is at most the maximum"}
    if len(columns) < policy.pbo_min_strategies:
        return _refused(gate, f"{len(columns)} distinct return series, fewer than "
                              f"{policy.pbo_min_strategies}")
    sessions = len(columns[0])
    splits = next((s for s in policy.pbo_splits if sessions // s >= policy.pbo_min_block_sessions), None)
    if splits is None:
        return _refused(gate, f"{sessions} sessions leave blocks shorter than "
                              f"{policy.pbo_min_block_sessions} sessions at every number of splits")
    try:
        found = vs.pbo_cscv(columns, splits)
    except vs.StatsError as exc:
        return _refused(gate, exc)
    # overfit / combinations <= max_pbo_bps / 10,000, in integers.
    passed = found["overfit"] * _BPS <= policy.max_pbo_bps * found["combinations"]
    return {**gate, "passed": passed, "value": found["pbo"], **found}


def haircut_sharpe_gate(columns, candidate: int, ledger: Mapping[str, Any], family: str,
                        policy: ValidationPolicy) -> dict:
    """The candidate's Sharpe ratio after the policy's multiple-testing
    adjustment for the family's trials (validation_stats.haircut), a year,
    is at least the policy's floor.

    The other trials' p-values are those of the other distinct columns. A
    trial with no column of its own (a version that shares a column, the
    candidate's included, or a trial that left no evaluation) is unseen,
    p = 1, as is a column with no Sharpe ratio (a constant series). Giving a
    twin its column's p-value instead could only lower the candidate's
    adjusted p-value, and counting twins may only make the verdict
    stricter."""
    from . import validation_stats as vs

    gate = {"method": policy.haircut_method, "minimum_annual": policy.min_haircut_sharpe_annual_milli / 1_000,
            "description": "The Sharpe ratio left after adjusting for the family's trials is at least "
                           "the minimum a year"}
    try:
        sessions_a_year = _sessions_per_year(policy, family)
        sessions = len(columns[candidate])
        others = []
        for index, column in enumerate(columns):
            if index != candidate:
                try:
                    others.append(vs.p_value(vs.sharpe(column), sessions))
                except vs.StatsError:
                    others.append(1.0)
        found = vs.haircut(vs.sharpe(columns[candidate]), sessions, ledger.get("trials"),
                           policy.haircut_method, others)
    except vs.StatsError as exc:
        return _refused(gate, exc)
    annual = found["haircut_sharpe"] * math.sqrt(sessions_a_year)
    return {**gate, "passed": annual >= gate["minimum_annual"], "value_annual": annual,
            "sessions_per_year": sessions_a_year, **found}


def t_statistic_gate(columns, candidate: int, policy: ValidationPolicy) -> dict:
    """The candidate's t-statistic, SR sqrt(T) with its per-period Sharpe
    ratio over its T sessions, is at least the policy's floor."""
    from . import validation_stats as vs

    gate = {"minimum": policy.min_t_statistic_milli / 1_000,
            "description": "The t-statistic of the Sharpe ratio is at least the minimum"}
    try:
        sessions = len(columns[candidate])
        value = vs.sharpe(columns[candidate]) * math.sqrt(sessions)
    except vs.StatsError as exc:
        return _refused(gate, exc)
    return {**gate, "passed": value >= gate["minimum"], "value": value, "sessions": sessions}


def backtest_length_gate(columns, candidate: int, ledger: Mapping[str, Any]) -> dict:
    """The backtest is at least the minimum backtest length for the family's
    trials (the raw ledger count) and the candidate's Sharpe ratio
    (validation_stats.min_backtest_length)."""
    from . import validation_stats as vs

    gate = {"description": "The backtest is at least the minimum length for the family's trials"}
    try:
        sessions = len(columns[candidate])
        minimum = vs.min_backtest_length(ledger.get("trials"), vs.sharpe(columns[candidate]))
    except vs.StatsError as exc:
        return _refused(gate, exc)
    return {**gate, "passed": sessions >= minimum, "sessions": sessions, "minimum": minimum,
            "n_trials": ledger.get("trials")}


def evaluate(matrix: Mapping[str, Any], candidate_version: str, ledger: Mapping[str, Any], family: str,
             limits: RiskLimits, risk: Mapping[str, Any] | None,
             policy: ValidationPolicy = POLICY_V1) -> dict:
    """The policy's verdict on a candidate strategy version: every gate, and
    whether all of them passed.

    ``matrix`` is the family's trial matrix (trial_matrix.build; the caller
    checks its id), ``ledger`` the family's trials it was built from
    (trial_matrix.family_state), ``family`` the candidate's dataset family,
    and ``limits`` and ``risk`` its declared risk limits and its re-run's
    risk measurements. The verdict is pre-holdout: it does not include the
    one-shot holdout evaluation."""
    columns = [column["returns"] for column in matrix["columns"]]
    candidate = next((index for index, column in enumerate(matrix["columns"])
                      if candidate_version in column["strategy_version_ids"]), None)
    gates = {"risk_limits": risk_gate(limits, risk, policy)}
    if candidate is None:
        for name in STATISTICAL_GATES:
            gates[name] = _refused({}, "the candidate is not in the matrix")
    else:
        gates.update({
            "deflated_sharpe": deflated_sharpe_gate(columns, candidate, ledger, policy),
            "overfitting": overfitting_gate(columns, policy),
            "haircut_sharpe": haircut_sharpe_gate(columns, candidate, ledger, family, policy),
            "t_statistic": t_statistic_gate(columns, candidate, policy),
            "backtest_length": backtest_length_gate(columns, candidate, ledger),
        })
    return {"policy": policy_id(policy), "stage": "pre-holdout", "candidate": candidate_version,
            "passed": all(gate["passed"] for gate in gates.values()), "gates": gates}
