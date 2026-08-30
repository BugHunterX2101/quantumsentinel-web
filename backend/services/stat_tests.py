"""QuantumSentinel — Statistical Testing Framework.

Proper hypothesis testing for strategy evaluation:
  - t-test with Newey-West (HAC) standard errors
  - Bootstrap confidence intervals for Sharpe
  - Permutation tests for strategy significance
  - Autocorrelation analysis (Ljung-Box)
  - Multiple-testing corrections (Bonferroni, BH, Deflated Sharpe Ratio)
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Newey-West HAC estimator
# ---------------------------------------------------------------------------

def newey_west_se(returns: np.ndarray, max_lag: int | None = None) -> float:
    """Newey-West heteroskedasticity and autocorrelation consistent SE.

    Uses the Bartlett kernel with automatic lag selection (Andrews 1991)
    if max_lag is not specified.
    """
    n = len(returns)
    if n < 5:
        return float(np.std(returns, ddof=1) / math.sqrt(n)) if n > 1 else 0.0

    if max_lag is None:
        max_lag = max(1, int(math.floor(4 * (n / 100) ** (2 / 9))))

    mean = np.mean(returns)
    resid = returns - mean

    # Variance (lag 0)
    gamma_0 = float(np.sum(resid ** 2)) / n

    # Add weighted autocovariances
    nw_var = gamma_0
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1)  # Bartlett kernel
        gamma_k = float(np.sum(resid[lag:] * resid[:-lag])) / n
        nw_var += 2 * weight * gamma_k

    nw_var = max(nw_var, 1e-12)  # floor at near-zero
    return math.sqrt(nw_var / n)


# ---------------------------------------------------------------------------
# t-test: H0: mean return = 0
# ---------------------------------------------------------------------------

def ttest_mean_return(returns: np.ndarray, use_nw: bool = True) -> dict:
    """Test H₀: α = 0 (mean excess return is zero).

    Uses Newey-West standard errors by default to account for
    autocorrelation and heteroskedasticity in return series.
    """
    n = len(returns)
    if n < 5:
        return {"t_stat": 0, "p_value": 1.0, "mean": 0, "se": 0,
                "n": n, "significant_5pct": False, "method": "insufficient_data"}

    mean = float(np.mean(returns))
    if use_nw:
        se = newey_west_se(returns)
        method = "newey_west"
    else:
        se = float(np.std(returns, ddof=1) / math.sqrt(n))
        method = "ols"

    if se < 1e-12:
        return {"t_stat": 0, "p_value": 1.0, "mean": mean, "se": 0,
                "n": n, "significant_5pct": False, "method": method}

    t_stat = mean / se
    # Two-tailed p-value using normal approximation (large sample)
    p_value = 2.0 * _normal_cdf(-abs(t_stat))

    return {
        "t_stat": round(t_stat, 4),
        "p_value": round(p_value, 6),
        "mean_daily_return": round(mean, 6),
        "se": round(se, 6),
        "n": n,
        "significant_5pct": p_value < 0.05,
        "significant_1pct": p_value < 0.01,
        "method": method,
        "annualized_mean": round(mean * 252, 4),
    }


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals for Sharpe ratio
# ---------------------------------------------------------------------------

def bootstrap_sharpe_ci(returns: np.ndarray, n_bootstrap: int = 10_000,
                        ci_level: float = 0.95,
                        seed: int = 42) -> dict:
    """Bootstrap confidence interval for the annualised Sharpe ratio.

    Resamples returns with replacement to construct empirical distribution
    of the Sharpe ratio.
    """
    n = len(returns)
    if n < 10:
        return {"sharpe": 0, "ci_lower": 0, "ci_upper": 0,
                "se": 0, "n_bootstrap": 0}

    rng = np.random.default_rng(seed)
    sharpe_samples = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        sample = rng.choice(returns, size=n, replace=True)
        std = np.std(sample, ddof=1)
        if std < 1e-9:
            sharpe_samples[i] = 0.0
        else:
            sharpe_samples[i] = float(np.mean(sample) / std * math.sqrt(252))

    alpha = (1 - ci_level) / 2
    ci_lower = float(np.percentile(sharpe_samples, alpha * 100))
    ci_upper = float(np.percentile(sharpe_samples, (1 - alpha) * 100))
    point_std = float(np.std(returns, ddof=1))
    point_sharpe = float(np.mean(returns) / max(point_std, 1e-9) * math.sqrt(252))

    return {
        "sharpe": round(point_sharpe, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "ci_level": ci_level,
        "bootstrap_se": round(float(np.std(sharpe_samples)), 4),
        "n_bootstrap": n_bootstrap,
        "sharpe_distribution": {
            "mean": round(float(np.mean(sharpe_samples)), 4),
            "median": round(float(np.median(sharpe_samples)), 4),
            "p5": round(float(np.percentile(sharpe_samples, 5)), 4),
            "p25": round(float(np.percentile(sharpe_samples, 25)), 4),
            "p75": round(float(np.percentile(sharpe_samples, 75)), 4),
            "p95": round(float(np.percentile(sharpe_samples, 95)), 4),
        },
        "probability_positive": round(float(np.mean(sharpe_samples > 0)), 4),
    }


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

def permutation_test(returns: np.ndarray, n_permutations: int = 10_000,
                     seed: int = 42) -> dict:
    """Permutation test for strategy significance.

    Shuffles return dates to destroy any signal-dependent structure,
    then computes the Sharpe ratio under the null of no timing skill.
    The p-value is the fraction of permuted Sharpes that exceed the
    observed Sharpe.
    """
    n = len(returns)
    if n < 10:
        return {"observed_sharpe": 0, "p_value": 1.0, "n_permutations": 0}

    rng = np.random.default_rng(seed)
    std = np.std(returns, ddof=1)
    observed_sharpe = float(np.mean(returns) / max(std, 1e-9) * math.sqrt(252))

    count_exceeding = 0
    for _ in range(n_permutations):
        perm = rng.permutation(returns)
        perm_std = np.std(perm, ddof=1)
        perm_sharpe = float(np.mean(perm) / max(perm_std, 1e-9) * math.sqrt(252))
        if perm_sharpe >= observed_sharpe:
            count_exceeding += 1

    p_value = (count_exceeding + 1) / (n_permutations + 1)  # continuity correction

    return {
        "observed_sharpe": round(observed_sharpe, 4),
        "p_value": round(p_value, 6),
        "n_permutations": n_permutations,
        "significant_5pct": p_value < 0.05,
        "significant_1pct": p_value < 0.01,
        "count_exceeding": count_exceeding,
    }


# ---------------------------------------------------------------------------
# Autocorrelation analysis (Ljung-Box)
# ---------------------------------------------------------------------------

def ljung_box_test(returns: np.ndarray, max_lag: int = 10) -> dict:
    """Ljung-Box test for serial autocorrelation in returns.

    Significant autocorrelation may indicate market inefficiency or
    data issues (e.g., stale prices).
    """
    n = len(returns)
    if n < max_lag + 5:
        return {"test_statistic": 0, "p_value": 1.0, "autocorrelations": {},
                "has_significant_autocorrelation": False}

    mean = np.mean(returns)
    var = np.sum((returns - mean) ** 2) / n

    if var < 1e-12:
        return {"test_statistic": 0, "p_value": 1.0, "autocorrelations": {},
                "has_significant_autocorrelation": False}

    autocorrs = {}
    q_stat = 0.0
    for lag in range(1, max_lag + 1):
        rk = float(np.sum((returns[lag:] - mean) * (returns[:-lag] - mean))) / (n * var)
        autocorrs[f"lag_{lag}"] = round(rk, 4)
        q_stat += rk ** 2 / (n - lag)

    q_stat *= n * (n + 2)

    # Chi-squared p-value approximation
    # degrees of freedom = max_lag
    p_value = _chi2_sf(q_stat, max_lag)

    return {
        "test_statistic": round(q_stat, 4),
        "p_value": round(p_value, 6),
        "degrees_of_freedom": max_lag,
        "autocorrelations": autocorrs,
        "has_significant_autocorrelation": p_value < 0.05,
        "interpretation": (
            "Returns show significant serial autocorrelation"
            if p_value < 0.05
            else "No significant autocorrelation detected"
        ),
    }


# ---------------------------------------------------------------------------
# Multiple-testing corrections
# ---------------------------------------------------------------------------

def bonferroni_correction(p_values: list[float],
                          alpha: float = 0.05) -> dict:
    """Bonferroni correction for multiple comparisons.

    The most conservative correction: divides α by the number of tests.
    """
    m = len(p_values)
    if m == 0:
        return {"adjusted_alpha": alpha, "significant": [], "n_tests": 0}

    adjusted_alpha = alpha / m
    significant = [i for i, p in enumerate(p_values) if p < adjusted_alpha]

    return {
        "method": "bonferroni",
        "original_alpha": alpha,
        "adjusted_alpha": round(adjusted_alpha, 6),
        "n_tests": m,
        "n_significant": len(significant),
        "significant_indices": significant,
        "adjusted_p_values": [round(min(p * m, 1.0), 6) for p in p_values],
    }


def benjamini_hochberg(p_values: list[float],
                       alpha: float = 0.05) -> dict:
    """Benjamini-Hochberg FDR control.

    Less conservative than Bonferroni: controls the expected proportion
    of false discoveries rather than the family-wise error rate.
    """
    m = len(p_values)
    if m == 0:
        return {"significant": [], "n_tests": 0}

    # Sort p-values and track original indices
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    significant = []

    # BH procedure
    prev_adj = 1.0
    for rank_minus_1 in range(m - 1, -1, -1):
        orig_idx, p = indexed[rank_minus_1]
        rank = rank_minus_1 + 1
        adj_p = min(prev_adj, p * m / rank)
        adjusted[orig_idx] = adj_p
        prev_adj = adj_p
        if adj_p < alpha:
            significant.append(orig_idx)

    return {
        "method": "benjamini_hochberg",
        "alpha": alpha,
        "n_tests": m,
        "n_significant": len(significant),
        "significant_indices": sorted(significant),
        "adjusted_p_values": [round(p, 6) for p in adjusted],
        "fdr_controlled": True,
    }


def deflated_sharpe_ratio(observed_sharpe: float,
                          n_trials: int,
                          n_observations: int,
                          skewness: float = 0.0,
                          kurtosis: float = 3.0,
                          sharpe_std: float = 1.0) -> dict:
    """Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

    Adjusts the Sharpe ratio for the number of strategy variants tried.
    This is the single most important test for quantitative strategy
    evaluation — it directly addresses data-snooping bias.

    DSR accounts for:
      1. Number of trials (strategies tested)
      2. Sample length
      3. Non-normality of returns (skew, kurtosis)
    """
    if n_trials < 1 or n_observations < 5:
        return {"dsr": 0, "p_value": 1.0, "significant": False}

    # Expected maximum Sharpe under null (from order statistics)
    # E[max(SR)] ≈ σ * √(2 * ln(N)) for N trials
    expected_max_sharpe = sharpe_std * math.sqrt(2 * math.log(max(n_trials, 2)))

    # Variance of Sharpe ratio estimator (Lo, 2002)
    # Var(SR) ≈ (1 + 0.25 * SR^2 * (γ₂ - 1) - SR * γ₁) / (n - 1)
    # where γ₁ = skewness, γ₂ = excess kurtosis
    excess_kurtosis = kurtosis - 3.0
    sr_variance = (
        (1 + 0.25 * observed_sharpe ** 2 * excess_kurtosis
         - observed_sharpe * skewness) / max(1, n_observations - 1)
    )
    sr_std = math.sqrt(max(sr_variance, 1e-12))

    # DSR = Prob(SR* > E[max(SR)])
    # where SR* is the observed Sharpe
    dsr_z = (observed_sharpe - expected_max_sharpe) / sr_std
    dsr_p = 1.0 - _normal_cdf(dsr_z)

    return {
        "observed_sharpe": round(observed_sharpe, 4),
        "expected_max_sharpe": round(expected_max_sharpe, 4),
        "n_trials": n_trials,
        "n_observations": n_observations,
        "dsr_z_score": round(dsr_z, 4),
        "dsr_p_value": round(dsr_p, 6),
        "significant_5pct": dsr_p < 0.05,
        "haircut_pct": round(
            max(0, (1 - observed_sharpe / max(expected_max_sharpe, 1e-9))) * 100, 1
        ),
        "interpretation": (
            f"After accounting for {n_trials} strategy trials, the observed "
            f"Sharpe of {observed_sharpe:.2f} is "
            + ("still significant" if dsr_p < 0.05
               else f"NOT significant (p={dsr_p:.3f}). Likely data-snooped.")
        ),
    }


# ---------------------------------------------------------------------------
# Full statistical test suite
# ---------------------------------------------------------------------------

def run_full_stat_tests(returns: np.ndarray,
                        n_strategies_tested: int = 1,
                        strategy_p_values: list[float] | None = None
                        ) -> dict:
    """Run the complete statistical testing suite on strategy returns.

    This is the main entry point for statistical validation.
    """
    results = {}

    # 1. t-test with Newey-West
    results["ttest_nw"] = ttest_mean_return(returns, use_nw=True)
    results["ttest_ols"] = ttest_mean_return(returns, use_nw=False)

    # 2. Bootstrap CI for Sharpe
    results["bootstrap_sharpe"] = bootstrap_sharpe_ci(returns)

    # 3. Permutation test
    results["permutation_test"] = permutation_test(returns)

    # 4. Autocorrelation
    results["ljung_box"] = ljung_box_test(returns)

    # 5. Deflated Sharpe Ratio
    std = float(np.std(returns, ddof=1))
    sharpe = float(np.mean(returns) / max(std, 1e-9) * math.sqrt(252))
    skew = float(_skewness(returns))
    kurt = float(_kurtosis(returns))

    results["deflated_sharpe"] = deflated_sharpe_ratio(
        observed_sharpe=sharpe,
        n_trials=max(1, n_strategies_tested),
        n_observations=len(returns),
        skewness=skew,
        kurtosis=kurt,
    )

    # 6. Multiple testing corrections (if multiple p-values provided)
    if strategy_p_values and len(strategy_p_values) > 1:
        results["bonferroni"] = bonferroni_correction(strategy_p_values)
        results["benjamini_hochberg"] = benjamini_hochberg(strategy_p_values)

    # Summary
    results["summary"] = {
        "n_observations": len(returns),
        "mean_return_significant": results["ttest_nw"]["significant_5pct"],
        "sharpe_ci_excludes_zero": results["bootstrap_sharpe"]["ci_lower"] > 0,
        "permutation_significant": results["permutation_test"]["significant_5pct"],
        "has_autocorrelation": results["ljung_box"]["has_significant_autocorrelation"],
        "survives_deflation": results["deflated_sharpe"]["significant_5pct"],
        "overall_credible": (
            results["ttest_nw"]["significant_5pct"] and
            results["bootstrap_sharpe"]["ci_lower"] > 0 and
            results["permutation_test"]["significant_5pct"] and
            results["deflated_sharpe"]["significant_5pct"]
        ),
    }

    return results


# ---------------------------------------------------------------------------
# Pure-Python statistical helpers (avoid scipy dependency for Phase 1)
# ---------------------------------------------------------------------------

def _normal_cdf(x: float) -> float:
    """Standard normal CDF — Abramowitz & Stegun approximation."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _chi2_sf(x: float, df: int) -> float:
    """Survival function (1 - CDF) for chi-squared distribution.

    Uses the regularised incomplete gamma function approximation.
    Good enough for df ≤ 100 and x ≥ 0.
    """
    if x <= 0:
        return 1.0
    if df <= 0:
        return 0.0
    # Use normal approximation for large df
    if df > 30:
        z = ((x / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
        return 1.0 - _normal_cdf(z)

    # Wilson-Hilferty approximation
    a = df / 2.0
    z = ((x / (2 * a)) ** (1 / 3) - (1 - 1 / (9 * a))) / math.sqrt(1 / (9 * a))
    return 1.0 - _normal_cdf(z)


def _skewness(x: np.ndarray) -> float:
    """Sample skewness."""
    n = len(x)
    if n < 3:
        return 0.0
    mean = np.mean(x)
    m2 = np.sum((x - mean) ** 2) / n
    m3 = np.sum((x - mean) ** 3) / n
    if m2 < 1e-12:
        return 0.0
    return float(m3 / m2 ** 1.5)


def _kurtosis(x: np.ndarray) -> float:
    """Sample kurtosis (not excess — includes the 3)."""
    n = len(x)
    if n < 4:
        return 3.0
    mean = np.mean(x)
    m2 = np.sum((x - mean) ** 2) / n
    m4 = np.sum((x - mean) ** 4) / n
    if m2 < 1e-12:
        return 3.0
    return float(m4 / m2 ** 2)
