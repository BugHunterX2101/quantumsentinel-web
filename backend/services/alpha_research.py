"""QuantumSentinel — Alpha Research Framework.

Measures the predictive power (alpha) of signals/factors before committing
to a strategy. Industry-standard metrics used at quant funds:

  - Information Coefficient (IC): Spearman rank correlation between
    predicted and realised returns.
  - Rank IC (RIC): IC computed on ranked signals — more robust to outliers.
  - IC Decay: How fast predictive power decays over holding horizons.
  - IC Information Ratio (ICIR): IC / std(IC) — signal consistency score.
  - Hit Rate: Fraction of correct directional predictions.
  - Factor Turnover: How often signal rankings change.
  - Quintile Analysis: Return spread between top and bottom signal quintiles.
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core IC computation
# ---------------------------------------------------------------------------

def _rank(arr: np.ndarray) -> np.ndarray:
    """Return ranks (1-indexed, average for ties)."""
    n = len(arr)
    temp = arr.argsort()
    ranks = np.empty_like(temp, dtype=float)
    ranks[temp] = np.arange(1, n + 1)
    # Handle ties — average rank
    unique_vals = np.unique(arr)
    for v in unique_vals:
        mask = arr == v
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    return ranks


def spearman_ic(signals: np.ndarray, forward_returns: np.ndarray) -> float:
    """Spearman rank correlation between signals and forward returns.

    IC > 0: signal predicts direction correctly.
    IC > 0.05 is generally considered useful in practice.
    IC > 0.10 is strong alpha.

    Parameters
    ----------
    signals : array of signal values at time t (one per asset)
    forward_returns : array of realised returns at time t+h (one per asset)

    Returns
    -------
    IC in [-1, 1]
    """
    n = len(signals)
    if n < 4:
        return 0.0
    r1 = _rank(signals)
    r2 = _rank(forward_returns)
    mean1, mean2 = r1.mean(), r2.mean()
    num = np.sum((r1 - mean1) * (r2 - mean2))
    den = math.sqrt(np.sum((r1 - mean1) ** 2) * np.sum((r2 - mean2) ** 2))
    if den < 1e-12:
        return 0.0
    return float(num / den)


def pearson_ic(signals: np.ndarray, forward_returns: np.ndarray) -> float:
    """Pearson correlation between signals and forward returns."""
    n = len(signals)
    if n < 4:
        return 0.0
    s_std = np.std(signals, ddof=1)
    r_std = np.std(forward_returns, ddof=1)
    if s_std < 1e-12 or r_std < 1e-12:
        return 0.0
    return float(np.corrcoef(signals, forward_returns)[0, 1])


# ---------------------------------------------------------------------------
# Time-series IC analysis (cross-sectional signal, panel data)
# ---------------------------------------------------------------------------

def compute_ic_series(signal_matrix: np.ndarray,
                      return_matrix: np.ndarray,
                      forward_horizon: int = 1) -> dict:
    """Compute IC time series from a signal panel.

    Parameters
    ----------
    signal_matrix : shape (T, N) — signal values, T time steps, N assets
    return_matrix : shape (T, N) — daily return matrix
    forward_horizon : forward return horizon in days

    Returns
    -------
    dict with ic_series, mean_ic, ic_std, icir, t_stat, p_value
    """
    T, N = signal_matrix.shape
    if N < 4 or T <= forward_horizon + 2:
        return {"ic_series": [], "mean_ic": 0, "ic_std": 0, "icir": 0,
                "t_stat": 0, "p_value": 1.0, "n_periods": 0}

    ic_list = []
    for t in range(T - forward_horizon):
        # Forward return: cumulative over horizon
        if forward_horizon == 1:
            fwd_ret = return_matrix[t + 1, :]
        else:
            fwd_ret = np.prod(1 + return_matrix[t + 1:t + 1 + forward_horizon, :],
                              axis=0) - 1
        sig = signal_matrix[t, :]
        # Only use assets where both signal and return are finite
        mask = np.isfinite(sig) & np.isfinite(fwd_ret)
        if mask.sum() < 4:
            continue
        ic = spearman_ic(sig[mask], fwd_ret[mask])
        ic_list.append(ic)

    if not ic_list:
        return {"ic_series": [], "mean_ic": 0, "ic_std": 0, "icir": 0,
                "t_stat": 0, "p_value": 1.0, "n_periods": 0}

    ics = np.array(ic_list)
    mean_ic = float(np.mean(ics))
    ic_std = float(np.std(ics, ddof=1))
    icir = mean_ic / ic_std if ic_std > 1e-9 else 0.0
    n = len(ics)
    t_stat = mean_ic / (ic_std / math.sqrt(n)) if ic_std > 1e-9 else 0.0
    from backend.services.stat_tests import _normal_cdf
    p_value = 2.0 * (1.0 - _normal_cdf(abs(t_stat)))

    return {
        "ic_series": [round(float(x), 5) for x in ics],
        "mean_ic": round(mean_ic, 5),
        "ic_std": round(ic_std, 5),
        "icir": round(icir, 4),
        "t_stat": round(t_stat, 4),
        "p_value": round(p_value, 6),
        "significant_5pct": p_value < 0.05,
        "n_periods": n,
        "positive_ic_pct": round(float(np.mean(ics > 0)), 4),
        "interpretation": _interpret_ic(mean_ic, icir),
    }


def _interpret_ic(mean_ic: float, icir: float) -> str:
    if abs(mean_ic) < 0.02:
        return "Negligible IC — signal has no meaningful predictive power."
    if abs(mean_ic) < 0.05:
        return f"Weak IC ({mean_ic:.3f}). Marginal alpha, likely not tradeable after costs."
    if abs(mean_ic) < 0.10:
        strength = "Moderate"
    else:
        strength = "Strong"
    direction = "positive" if mean_ic > 0 else "negative (fade the signal)"
    consistency = "consistent (ICIR > 0.5)" if abs(icir) > 0.5 else "inconsistent (ICIR < 0.5)"
    return (f"{strength} {direction} IC ({mean_ic:.3f}), {consistency}. "
            f"ICIR={icir:.2f}.")


# ---------------------------------------------------------------------------
# IC Decay analysis
# ---------------------------------------------------------------------------

def ic_decay(signal_matrix: np.ndarray,
             return_matrix: np.ndarray,
             max_horizon: int = 20) -> dict:
    """IC at multiple forward horizons to show alpha decay profile.

    Useful to calibrate holding period — alpha typically decays
    faster for short-term momentum and slower for value factors.
    """
    T, N = signal_matrix.shape
    horizons = list(range(1, min(max_horizon + 1, T // 3)))
    decay_ics = []
    for h in horizons:
        result = compute_ic_series(signal_matrix, return_matrix, h)
        decay_ics.append({
            "horizon": h,
            "mean_ic": result["mean_ic"],
            "icir": result["icir"],
            "n_periods": result["n_periods"],
        })

    # Half-life: horizon where IC decays to 50% of IC at horizon=1
    if decay_ics and abs(decay_ics[0]["mean_ic"]) > 1e-6:
        base_ic = decay_ics[0]["mean_ic"]
        half_life = None
        for item in decay_ics:
            if abs(item["mean_ic"]) <= abs(base_ic) * 0.5:
                half_life = item["horizon"]
                break
    else:
        half_life = None

    return {
        "horizons": horizons,
        "ic_by_horizon": decay_ics,
        "half_life_days": half_life,
        "recommended_holding_days": half_life or max(horizons),
    }


# ---------------------------------------------------------------------------
# Hit Rate
# ---------------------------------------------------------------------------

def hit_rate(signals: np.ndarray, forward_returns: np.ndarray,
             threshold: float = 0.0) -> dict:
    """Fraction of correct directional predictions.

    Counts cases where sign(signal) == sign(forward_return).
    A threshold can filter out near-zero signals.
    """
    if len(signals) < 2:
        return {"hit_rate": 0.5, "n": 0}

    mask = np.abs(signals) > threshold
    if mask.sum() < 2:
        return {"hit_rate": 0.5, "n": 0}

    sig = signals[mask]
    ret = forward_returns[mask]

    correct = np.sum(np.sign(sig) == np.sign(ret))
    n = mask.sum()
    hr = float(correct) / n

    # Binomial test: H0 = hit rate = 0.5
    # Using normal approximation for n > 30
    se = math.sqrt(0.25 / n)
    z = (hr - 0.5) / se
    from backend.services.stat_tests import _normal_cdf
    p_value = 2.0 * (1.0 - _normal_cdf(abs(z)))

    return {
        "hit_rate": round(hr, 4),
        "n": int(n),
        "correct": int(correct),
        "z_stat": round(z, 3),
        "p_value": round(p_value, 5),
        "significant_5pct": p_value < 0.05,
    }


# ---------------------------------------------------------------------------
# Quintile Analysis
# ---------------------------------------------------------------------------

def quintile_returns(signals: np.ndarray,
                     forward_returns: np.ndarray,
                     n_quantiles: int = 5) -> dict:
    """Bin assets by signal quantile and compute average forward returns.

    The spread between Q5 (top signal) and Q1 (bottom signal) is the
    key measure of factor efficacy — a monotonic Q1→Q5 return ladder
    indicates a robust cross-sectional signal.
    """
    if len(signals) < n_quantiles * 2:
        return {"quantile_returns": [], "spread": 0}

    # Assign quantile bins based on signal rank
    ranks = _rank(signals)
    n = len(signals)
    bin_size = n / n_quantiles

    q_returns = []
    for q in range(1, n_quantiles + 1):
        lo = (q - 1) * bin_size
        hi = q * bin_size
        mask = (ranks > lo) & (ranks <= hi)
        if mask.sum() == 0:
            q_returns.append(0.0)
        else:
            q_returns.append(float(np.mean(forward_returns[mask])))

    spread = q_returns[-1] - q_returns[0]  # Q5 - Q1
    monotonic = all(q_returns[i] <= q_returns[i + 1]
                    for i in range(len(q_returns) - 1))

    return {
        "quantile_returns": [round(r, 6) for r in q_returns],
        "quantile_labels": [f"Q{i + 1}" for i in range(n_quantiles)],
        "spread_q5_q1": round(spread, 6),
        "annualised_spread": round(spread * 252, 4),
        "monotonic": monotonic,
        "n_assets": len(signals),
    }


# ---------------------------------------------------------------------------
# Factor Turnover
# ---------------------------------------------------------------------------

def factor_turnover(signal_matrix: np.ndarray) -> dict:
    """Measure how much signal rankings change period-to-period.

    High turnover → higher transaction costs needed to exploit the signal.
    Calculated as average pairwise rank correlation change across periods.
    """
    T, N = signal_matrix.shape
    if T < 3 or N < 2:
        return {"avg_turnover": 0, "avg_rank_correlation": 1.0}

    rank_corrs = []
    for t in range(T - 1):
        sig_t = signal_matrix[t, :]
        sig_t1 = signal_matrix[t + 1, :]
        mask = np.isfinite(sig_t) & np.isfinite(sig_t1)
        if mask.sum() < 4:
            continue
        ic = spearman_ic(sig_t[mask], sig_t1[mask])
        rank_corrs.append(ic)

    if not rank_corrs:
        return {"avg_turnover": 0, "avg_rank_correlation": 1.0}

    avg_rc = float(np.mean(rank_corrs))
    avg_to = 1.0 - avg_rc  # turnover = 1 - autocorrelation of ranks

    return {
        "avg_rank_correlation": round(avg_rc, 4),
        "avg_turnover": round(avg_to, 4),
        "turnover_interpretation": (
            "Low turnover (trading cost friendly)" if avg_to < 0.3
            else "High turnover (requires low transaction costs to be profitable)"
        ),
    }


# ---------------------------------------------------------------------------
# Full alpha research report
# ---------------------------------------------------------------------------

def run_alpha_research(signal_matrix: np.ndarray,
                       return_matrix: np.ndarray,
                       max_horizon: int = 20) -> dict:
    """Run the complete alpha research pipeline.

    Parameters
    ----------
    signal_matrix : shape (T, N) — T time steps, N assets
    return_matrix : shape (T, N) — daily returns

    Returns
    -------
    Comprehensive alpha quality report
    """
    T, N = signal_matrix.shape

    # 1. IC at horizon=1
    ic_1d = compute_ic_series(signal_matrix, return_matrix, forward_horizon=1)

    # 2. IC decay (horizons 1..max_horizon)
    decay = ic_decay(signal_matrix, return_matrix, max_horizon=max_horizon)

    # 3. Hit rate (latest cross-section)
    if T >= 2:
        last_signals = signal_matrix[-2, :]
        last_returns = return_matrix[-1, :]
        mask = np.isfinite(last_signals) & np.isfinite(last_returns)
        if mask.sum() >= 4:
            hr = hit_rate(last_signals[mask], last_returns[mask])
        else:
            hr = {"hit_rate": 0.5, "n": 0}
    else:
        hr = {"hit_rate": 0.5, "n": 0}

    # 4. Quintile analysis (latest cross-section)
    if T >= 2:
        last_signals = signal_matrix[-2, :]
        last_returns = return_matrix[-1, :]
        mask = np.isfinite(last_signals) & np.isfinite(last_returns)
        if mask.sum() >= 10:
            quint = quintile_returns(last_signals[mask], last_returns[mask])
        else:
            quint = {"quantile_returns": [], "spread_q5_q1": 0}
    else:
        quint = {"quantile_returns": [], "spread_q5_q1": 0}

    # 5. Factor turnover
    to = factor_turnover(signal_matrix)

    # Alpha quality score (composite, 0-100)
    score = _alpha_quality_score(ic_1d, hr, quint, to)

    return {
        "ic_analysis": ic_1d,
        "ic_decay": decay,
        "hit_rate": hr,
        "quintile_analysis": quint,
        "factor_turnover": to,
        "alpha_quality_score": score,
        "n_assets": N,
        "n_periods": T,
    }


def _alpha_quality_score(ic_result: dict, hr_result: dict,
                         quint_result: dict, to_result: dict) -> dict:
    """Composite alpha quality score (0-100)."""
    score = 0.0

    # IC contribution (0-40)
    mean_ic = abs(ic_result.get("mean_ic", 0))
    icir = abs(ic_result.get("icir", 0))
    ic_score = min(mean_ic / 0.10 * 20, 20) + min(icir / 1.0 * 20, 20)
    score += ic_score

    # Hit rate contribution (0-25)
    hr = hr_result.get("hit_rate", 0.5)
    hr_score = min(max(hr - 0.5, 0) / 0.15 * 25, 25)
    score += hr_score

    # Quintile spread contribution (0-25)
    spread = abs(quint_result.get("annualised_spread", 0))
    monotonic = quint_result.get("monotonic", False)
    q_score = min(spread / 0.20 * 20, 20) + (5 if monotonic else 0)
    score += q_score

    # Turnover penalty (0 to -10)
    avg_to = to_result.get("avg_turnover", 0.5)
    score -= min(avg_to * 10, 10)

    score = max(0.0, min(100.0, score))

    rating = "Poor" if score < 25 else "Fair" if score < 50 else "Good" if score < 75 else "Excellent"
    return {
        "score": round(score, 1),
        "rating": rating,
        "components": {
            "ic_score": round(ic_score, 1),
            "hit_rate_score": round(hr_score, 1),
            "quintile_score": round(q_score, 1),
            "turnover_penalty": round(-min(avg_to * 10, 10), 1),
        },
    }
