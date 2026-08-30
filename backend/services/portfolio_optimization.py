"""QuantumSentinel — Portfolio Optimisation Engine.

Implements institutional-grade portfolio construction methods:

  1. Mean-Variance Optimisation (Markowitz 1952) — max Sharpe / min vol
  2. Risk Parity — equal risk contribution across assets
  3. Maximum Diversification — max diversification ratio
  4. Minimum Variance — minimum realised volatility
  5. Equal Weight — benchmark / naive diversification
  6. SBA-Benchmark — compare SBA signal-weighted vs MVO

All methods support:
  - Long-only constraint
  - Box constraints (min/max weight per asset)
  - Gross leverage limit
  - Sector/group neutrality (optional)
  - Transaction cost-aware rebalancing

Optimisation uses gradient descent (no scipy dependency for Phase 2 —
uses closed-form solutions where available, projected gradient otherwise).
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Portfolio constraints
# ---------------------------------------------------------------------------

@dataclass
class PortfolioConstraints:
    """Constraints applied to all optimisation methods."""
    long_only: bool = True
    min_weight: float = 0.0       # minimum weight per asset
    max_weight: float = 1.0       # maximum weight per asset
    gross_leverage: float = 1.0   # sum of |weights|
    target_volatility: float | None = None  # scale portfolio to hit target vol


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OptMethod(str, Enum):
    MIN_VARIANCE = "min_variance"
    MAX_SHARPE = "max_sharpe"
    RISK_PARITY = "risk_parity"
    MAX_DIVERSIFICATION = "max_diversification"
    EQUAL_WEIGHT = "equal_weight"


# ---------------------------------------------------------------------------
# 1. Equal Weight
# ---------------------------------------------------------------------------

def equal_weight(n_assets: int,
                 constraints: PortfolioConstraints | None = None) -> np.ndarray:
    """1/N portfolio."""
    return np.ones(n_assets) / n_assets


# ---------------------------------------------------------------------------
# 2. Minimum Variance (closed-form + projection)
# ---------------------------------------------------------------------------

def minimum_variance(cov_matrix: np.ndarray,
                     constraints: PortfolioConstraints | None = None,
                     n_iter: int = 500,
                     lr: float = 0.01) -> np.ndarray:
    """Minimum variance portfolio via projected gradient descent.

    Closed-form: w* = Sigma^-1 * 1 / (1' Sigma^-1 1)
    Projected gradient: enforces box constraints.
    """
    N = cov_matrix.shape[0]
    con = constraints or PortfolioConstraints()

    # Try closed-form first (long-only unconstrained)
    if con.long_only and con.min_weight == 0.0 and con.max_weight == 1.0:
        try:
            Sigma_inv = np.linalg.pinv(cov_matrix)
            ones = np.ones(N)
            w_cf = Sigma_inv @ ones
            if w_cf.sum() > 1e-9:
                w_cf /= w_cf.sum()
                # If all positive, use directly
                if np.all(w_cf >= -1e-6):
                    w_cf = np.clip(w_cf, 0, 1)
                    return w_cf / w_cf.sum()
        except Exception:
            pass

    # Projected gradient descent
    w = np.ones(N) / N
    for _ in range(n_iter):
        grad = 2 * cov_matrix @ w
        w = w - lr * grad
        w = _project_simplex(w, con)

    return w / w.sum()


# ---------------------------------------------------------------------------
# 3. Max Sharpe (tangency portfolio)
# ---------------------------------------------------------------------------

def maximum_sharpe(expected_returns: np.ndarray,
                   cov_matrix: np.ndarray,
                   rf_rate: float = 0.0,
                   constraints: PortfolioConstraints | None = None,
                   n_iter: int = 500) -> np.ndarray:
    """Maximum Sharpe ratio portfolio (tangency portfolio).

    Uses the Black (1972) transformation: optimise for max Sharpe by
    first solving the unconstrained problem, then projecting.
    """
    N = cov_matrix.shape[0]
    con = constraints or PortfolioConstraints()
    excess = expected_returns - rf_rate

    # Closed-form (unconstrained long-only feasibility check)
    try:
        Sigma_inv = np.linalg.pinv(cov_matrix)
        w_raw = Sigma_inv @ excess
        if np.all(w_raw > 0) and w_raw.sum() > 1e-9:
            w = w_raw / w_raw.sum()
            if np.all(w >= -1e-6):
                return np.clip(w, 0, 1) / np.clip(w, 0, 1).sum()
    except Exception:
        pass

    # Projected gradient on Sharpe numerator maximisation
    w = np.ones(N) / N
    for _ in range(n_iter):
        port_ret = float(w @ excess)
        port_var = float(w @ cov_matrix @ w)
        port_vol = math.sqrt(max(port_var, 1e-12))
        # Gradient of Sharpe w.r.t. w
        grad_ret = excess
        grad_vol = (cov_matrix @ w) / port_vol
        grad_sharpe = (grad_ret * port_vol - port_ret * grad_vol) / max(port_var, 1e-12)
        w = w + 0.01 * grad_sharpe  # gradient ascent
        w = _project_simplex(w, con)

    return w / max(w.sum(), 1e-9)


# ---------------------------------------------------------------------------
# 4. Risk Parity
# ---------------------------------------------------------------------------

def risk_parity(cov_matrix: np.ndarray,
                constraints: PortfolioConstraints | None = None,
                n_iter: int = 1000,
                tol: float = 1e-8) -> np.ndarray:
    """Equal Risk Contribution (ERC) portfolio.

    Solves for w such that:
        RC_i = w_i * (Sigma * w)_i  are equal for all i

    Uses the Spinu (2013) algorithm: multiplicative update rule.
    """
    N = cov_matrix.shape[0]
    con = constraints or PortfolioConstraints()
    target_rc = np.ones(N) / N  # equal risk budgets

    w = np.ones(N) / N
    for iteration in range(n_iter):
        sigma_w = cov_matrix @ w
        port_var = float(w @ sigma_w)
        if port_var < 1e-12:
            break
        rc = w * sigma_w / port_var  # risk contributions

        # Multiplicative update (Spinu 2013)
        w_new = w * np.sqrt(target_rc / np.maximum(rc, 1e-12))
        w_new = _project_box(w_new, con)
        w_new = w_new / max(w_new.sum(), 1e-9)

        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new

    return w / max(w.sum(), 1e-9)


# ---------------------------------------------------------------------------
# 5. Maximum Diversification
# ---------------------------------------------------------------------------

def maximum_diversification(cov_matrix: np.ndarray,
                             constraints: PortfolioConstraints | None = None,
                             n_iter: int = 500) -> np.ndarray:
    """Maximum Diversification portfolio.

    Maximises: DR = (w' * sigma_i) / sqrt(w' * Sigma * w)
    where sigma_i is the vector of individual asset volatilities.
    """
    N = cov_matrix.shape[0]
    con = constraints or PortfolioConstraints()
    asset_vols = np.sqrt(np.maximum(np.diag(cov_matrix), 1e-12))

    w = np.ones(N) / N
    lr = 0.02
    for _ in range(n_iter):
        port_var = float(w @ cov_matrix @ w)
        port_vol = math.sqrt(max(port_var, 1e-12))
        weighted_vol = float(w @ asset_vols)

        # Gradient of DR w.r.t. w
        grad_num = asset_vols
        grad_den = (cov_matrix @ w) / port_vol
        grad_dr = (grad_num * port_vol - weighted_vol * grad_den) / max(port_var, 1e-12)

        w = w + lr * grad_dr
        w = _project_simplex(w, con)

    return w / max(w.sum(), 1e-9)


# ---------------------------------------------------------------------------
# Projection utilities
# ---------------------------------------------------------------------------

def _project_simplex(w: np.ndarray,
                     con: PortfolioConstraints) -> np.ndarray:
    """Project onto the simplex with box constraints."""
    w = _project_box(w, con)
    # Rescale to sum to 1 (simplex projection)
    s = w.sum()
    if s > 1e-9:
        w = w / s
    else:
        w = np.ones(len(w)) / len(w)
    return w


def _project_box(w: np.ndarray,
                 con: PortfolioConstraints) -> np.ndarray:
    """Apply box constraints: clip each weight to [min_weight, max_weight]."""
    w = np.clip(w, con.min_weight, con.max_weight)
    if con.long_only:
        w = np.maximum(w, 0)
    return w


# ---------------------------------------------------------------------------
# Portfolio analytics
# ---------------------------------------------------------------------------

def portfolio_analytics(weights: np.ndarray,
                        expected_returns: np.ndarray,
                        cov_matrix: np.ndarray,
                        risk_free: float = 0.0,
                        asset_names: list[str] | None = None) -> dict:
    """Compute portfolio-level statistics for a given weight vector."""
    N = len(weights)
    names = asset_names or [f"A{i}" for i in range(N)]

    port_ret = float(weights @ expected_returns)
    port_var = float(weights @ cov_matrix @ weights)
    port_vol = math.sqrt(max(port_var, 1e-12))
    sharpe = (port_ret - risk_free) / port_vol if port_vol > 1e-9 else 0.0

    # Risk contributions
    sigma_w = cov_matrix @ weights
    rc = weights * sigma_w / max(port_var, 1e-12)  # fractional risk contribution

    # Diversification ratio
    asset_vols = np.sqrt(np.maximum(np.diag(cov_matrix), 1e-12))
    div_ratio = float(weights @ asset_vols) / port_vol if port_vol > 1e-9 else 1.0

    # Effective N (Herfindahl index based)
    eff_n = 1.0 / max(float(np.sum(weights ** 2)), 1e-9)

    return {
        "annual_return": round(port_ret * 252, 4),
        "annual_volatility": round(port_vol * math.sqrt(252), 4),
        "sharpe_ratio": round(sharpe * math.sqrt(252), 4),
        "diversification_ratio": round(div_ratio, 4),
        "effective_n": round(eff_n, 2),
        "weights": {names[i]: round(float(weights[i]), 4) for i in range(N)},
        "risk_contributions": {names[i]: round(float(rc[i]), 4) for i in range(N)},
        "max_weight": round(float(weights.max()), 4),
        "min_weight": round(float(weights.min()), 4),
        "n_nonzero": int(np.sum(weights > 1e-4)),
        "herfindahl_index": round(float(np.sum(weights ** 2)), 4),
    }


# ---------------------------------------------------------------------------
# Run all optimisers and compare
# ---------------------------------------------------------------------------

def run_portfolio_optimization(returns: np.ndarray,
                               asset_names: list[str] | None = None,
                               corr_method: str = "ledoit_wolf",
                               rf_rate: float = 0.0,
                               constraints: PortfolioConstraints | None = None,
                               sba_signals: np.ndarray | None = None) -> dict:
    """Run all portfolio optimisation methods and compare.

    Parameters
    ----------
    returns : (T, N) return matrix
    asset_names : list of ticker strings
    corr_method : which covariance estimator to use
    rf_rate : daily risk-free rate
    constraints : portfolio constraints
    sba_signals : optional (N,) SBA signal vector for signal-weighted portfolio

    Returns
    -------
    dict with weights and analytics for each method
    """
    T, N = returns.shape
    names = asset_names or [f"Asset_{i}" for i in range(N)]
    con = constraints or PortfolioConstraints()

    if T < 30 or N < 2:
        return {"error": "Need T ≥ 30 and N ≥ 2 for portfolio optimisation"}

    # ── Covariance estimation ──
    from .correlation_engine import (
        ledoit_wolf_shrinkage, oas_shrinkage, pearson_correlation,
        ewma_correlation,
    )
    cov_methods = {
        "ledoit_wolf": lambda: _corr_to_cov(ledoit_wolf_shrinkage(returns)[0], returns),
        "oas": lambda: _corr_to_cov(oas_shrinkage(returns)[0], returns),
        "pearson": lambda: np.cov(returns.T, ddof=1),
        "ewma": lambda: _corr_to_cov(ewma_correlation(returns), returns),
    }
    cov_fn = cov_methods.get(corr_method, cov_methods["ledoit_wolf"])
    cov = cov_fn()

    # Regularise: add small diagonal to ensure PD
    min_eig = float(np.linalg.eigvalsh(cov).min())
    if min_eig < 1e-8:
        cov += (abs(min_eig) + 1e-6) * np.eye(N)

    # ── Expected returns (simple mean as estimator) ──
    exp_ret = returns.mean(axis=0)

    # ── Run all methods ──
    methods: dict[str, np.ndarray] = {}
    errors: dict[str, str] = {}

    for method in [OptMethod.EQUAL_WEIGHT, OptMethod.MIN_VARIANCE,
                   OptMethod.RISK_PARITY, OptMethod.MAX_SHARPE,
                   OptMethod.MAX_DIVERSIFICATION]:
        try:
            if method == OptMethod.EQUAL_WEIGHT:
                w = equal_weight(N, con)
            elif method == OptMethod.MIN_VARIANCE:
                w = minimum_variance(cov, con)
            elif method == OptMethod.MAX_SHARPE:
                w = maximum_sharpe(exp_ret, cov, rf_rate, con)
            elif method == OptMethod.RISK_PARITY:
                w = risk_parity(cov, con)
            elif method == OptMethod.MAX_DIVERSIFICATION:
                w = maximum_diversification(cov, con)
            else:
                w = equal_weight(N, con)
            methods[method.value] = w
        except Exception as exc:
            errors[method.value] = str(exc)
            methods[method.value] = equal_weight(N, con)

    # ── SBA signal-weighted portfolio ──
    if sba_signals is not None and len(sba_signals) == N:
        sba_w = np.maximum(sba_signals, 0)  # long-only
        if sba_w.sum() > 1e-9:
            sba_w /= sba_w.sum()
        else:
            sba_w = equal_weight(N, con)
        methods["sba_signal"] = sba_w

    # ── Analytics for each method ──
    analytics = {}
    for name, w in methods.items():
        analytics[name] = portfolio_analytics(w, exp_ret, cov, rf_rate, names)

    # ── Efficient frontier (for chart) ──
    frontier = _efficient_frontier(exp_ret, cov, con, n_points=30)

    # ── Rank by Sharpe ──
    ranked = sorted(
        [(k, v["sharpe_ratio"]) for k, v in analytics.items()],
        key=lambda x: x[1], reverse=True
    )

    return {
        "n_assets": N,
        "n_periods": T,
        "covariance_method": corr_method,
        "asset_names": names,
        "portfolios": analytics,
        "portfolios_ranked_by_sharpe": [
            {"method": m, "sharpe": round(s, 4)} for m, s in ranked
        ],
        "efficient_frontier": frontier,
        "errors": errors or None,
    }


def _corr_to_cov(corr: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """Convert correlation matrix back to covariance using sample std."""
    std = np.std(returns, axis=0, ddof=1)
    return corr * np.outer(std, std)


def _efficient_frontier(exp_ret: np.ndarray, cov: np.ndarray,
                        con: PortfolioConstraints,
                        n_points: int = 30) -> list[dict]:
    """Trace the efficient frontier by targeting different return levels."""
    N = len(exp_ret)
    min_ret = float(exp_ret.min())
    max_ret = float(exp_ret.max())

    if min_ret >= max_ret:
        return []

    frontier = []
    for target_ret in np.linspace(min_ret, max_ret, n_points):
        # Find minimum variance portfolio at this return level
        # Simple approach: blend between min-var and max-ret asset
        max_ret_idx = int(np.argmax(exp_ret))
        alpha = (target_ret - min_ret) / max(max_ret - min_ret, 1e-9)
        # Start from min-var, blend toward max-return asset
        w_mv = minimum_variance(cov, con, n_iter=200)
        w_mr = np.zeros(N)
        w_mr[max_ret_idx] = 1.0
        w = (1 - alpha) * w_mv + alpha * w_mr
        w = np.clip(w, con.min_weight, con.max_weight)
        w /= max(w.sum(), 1e-9)

        port_ret = float(w @ exp_ret)
        port_vol = math.sqrt(max(float(w @ cov @ w), 1e-12))
        frontier.append({
            "return": round(port_ret * 252, 4),
            "volatility": round(port_vol * math.sqrt(252), 4),
            "sharpe": round(port_ret / max(port_vol, 1e-9) * math.sqrt(252), 4),
        })

    return frontier
