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

The statistical gates (the deflated Sharpe ratio, the probability of
backtest overfitting, the search budget) join this policy in later R3
changes. No verdict records a policy id until the validation job does, so
until then the policy's fields may still change.
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


@dataclass(frozen=True)
class ValidationPolicy:
    name: str
    # The largest drawdown and single position a strategy may declare, as
    # shares of equity.
    max_declared_drawdown_bps: int
    max_declared_position_bps: int

    def __post_init__(self):
        for field, value in asdict(self).items():
            if field == "name":
                if not isinstance(value, str) or not value:
                    raise ValueError("a policy needs a name")
            elif not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= _BPS:
                raise ValueError(f"{field} must be an integer number of basis points in 1..{_BPS}")


POLICY_V1 = ValidationPolicy(
    name="qs-validation-policy-v1",
    max_declared_drawdown_bps=2_500,
    max_declared_position_bps=2_500,
)


def policy_id(policy: ValidationPolicy) -> str:
    return f"{POLICY_SCHEME}:{hashlib.sha256(canonical_json(asdict(policy))).hexdigest()}"


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
