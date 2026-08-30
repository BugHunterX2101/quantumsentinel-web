"""QuantumSentinel — Correlation & Covariance Engine.

Robust covariance/correlation estimation for multi-asset portfolios.
Naïve sample correlation is unreliable for quant portfolios (especially
with N ≈ T) — this module provides shrinkage and factor-based alternatives.

Methods implemented:
  - Pearson sample correlation
  - Spearman rank correlation (robust to outliers)
  - EWMA correlation (exponentially-weighted, captures regime changes)
  - Ledoit-Wolf analytical shrinkage (optimal for high-dimensional N)
  - Oracle Approximating Shrinkage (OAS) estimator
  - PCA-based factor covariance (low-rank + diagonal)
  - Minimum Variance Portfolio weights (from each estimator)
  - Condition number diagnostics (ill-conditioning detection)
"""
from __future__ import annotations

import math
import logging
from enum import Enum

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Correlation Methods Enum
# ---------------------------------------------------------------------------

class CorrMethod(str, Enum):
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    EWMA = "ewma"
    LEDOIT_WOLF = "ledoit_wolf"
    OAS = "oas"
    PCA = "pca"


# ---------------------------------------------------------------------------
# 1. Pearson sample correlation
# ---------------------------------------------------------------------------

def pearson_correlation(returns: np.ndarray) -> np.ndarray:
    """Standard Pearson sample correlation matrix. (T, N) → (N, N)."""
    if returns.shape[0] < 3:
        return np.eye(returns.shape[1])
    return np.corrcoef(returns.T)


# ---------------------------------------------------------------------------
# 2. Spearman rank correlation
# ---------------------------------------------------------------------------

def spearman_correlation(returns: np.ndarray) -> np.ndarray:
    """Spearman rank correlation — robust to heavy tails. (T, N) → (N, N)."""
    T, N = returns.shape
    if T < 3:
        return np.eye(N)

    # Rank each column (asset) independently
    ranked = np.zeros_like(returns)
    for j in range(N):
        col = returns[:, j]
        temp = col.argsort()
        ranks = np.empty_like(temp, dtype=float)
        ranks[temp] = np.arange(1, T + 1)
        ranked[:, j] = ranks

    return np.corrcoef(ranked.T)


# ---------------------------------------------------------------------------
# 3. EWMA correlation
# ---------------------------------------------------------------------------

def ewma_correlation(returns: np.ndarray, halflife: float = 60.0) -> np.ndarray:
    """Exponentially weighted moving average correlation.

    Uses RiskMetrics-style decay: lambda = exp(-ln2/halflife).
    More weight on recent observations — captures regime shifts.
    """
    T, N = returns.shape
    if T < 3:
        return np.eye(N)

    decay = math.exp(-math.log(2) / halflife)
    weights = np.array([decay ** (T - 1 - t) for t in range(T)])
    weights /= weights.sum()

    # Weighted covariance
    w_mean = (weights[:, None] * returns).sum(axis=0)
    deviations = returns - w_mean
    w_cov = (weights[:, None, None] *
             deviations[:, :, None] * deviations[:, None, :]).sum(axis=0)

    # Convert to correlation
    std = np.sqrt(np.diag(w_cov))
    outer_std = np.outer(std, std)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(outer_std > 1e-12, w_cov / outer_std, 0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


# ---------------------------------------------------------------------------
# 4. Ledoit-Wolf shrinkage (analytical — Ledoit & Wolf 2004)
# ---------------------------------------------------------------------------

def ledoit_wolf_shrinkage(returns: np.ndarray) -> tuple[np.ndarray, float]:
    """Analytical Ledoit-Wolf shrinkage estimator.

    Shrinks sample covariance toward scaled identity:
        Sigma_shrunk = (1-alpha)*S + alpha*mu*I

    where alpha is the optimal shrinkage intensity (analytical solution).

    Returns (shrunk_corr, shrinkage_intensity)
    """
    T, N = returns.shape
    if T < 3 or N < 2:
        return np.eye(N), 0.0

    # Sample covariance
    S = np.cov(returns.T, ddof=1)

    # Target: scaled identity (Ledoit-Wolf constant correlation model)
    mu = np.trace(S) / N

    # ── Oracle shrinkage intensity (Ledoit-Wolf analytical formula) ──
    # delta: average squared Frobenius distance between S and target
    X = returns - returns.mean(axis=0)
    phi_hat = 0.0
    for t in range(T):
        x = X[t:t+1, :].T  # (N, 1)
        phi_hat += float((x @ x.T - S @ (mu * np.eye(N))).ravel() ** 2 @ np.ones(N**2))
    phi_hat /= T ** 2

    # Analytical shrinkage coefficient
    delta_hat = float(np.sum((S - mu * np.eye(N)) ** 2))
    alpha = min(1.0, max(0.0, phi_hat / delta_hat if delta_hat > 1e-12 else 0))

    # Shrunk covariance
    S_shrunk = (1 - alpha) * S + alpha * mu * np.eye(N)

    # Convert to correlation
    std = np.sqrt(np.maximum(np.diag(S_shrunk), 1e-12))
    corr = S_shrunk / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1, 1)

    return corr, round(alpha, 4)


# ---------------------------------------------------------------------------
# 5. Oracle Approximating Shrinkage (OAS) — Chen et al. 2010
# ---------------------------------------------------------------------------

def oas_shrinkage(returns: np.ndarray) -> tuple[np.ndarray, float]:
    """OAS estimator — typically more accurate than LW for small T/N ratios."""
    T, N = returns.shape
    if T < 3 or N < 2:
        return np.eye(N), 0.0

    S = np.cov(returns.T, ddof=1)
    trace_S = np.trace(S)
    trace_S2 = np.trace(S @ S)
    mu = trace_S / N

    # OAS optimal alpha
    rho_numerator = (1 - 2 / N) * trace_S2 + trace_S ** 2
    rho_denominator = (T + 1 - 2 / N) * (trace_S2 - trace_S ** 2 / N)
    alpha = min(1.0, max(0.0,
        rho_numerator / rho_denominator if rho_denominator > 1e-12 else 0))

    S_shrunk = (1 - alpha) * S + alpha * mu * np.eye(N)
    std = np.sqrt(np.maximum(np.diag(S_shrunk), 1e-12))
    corr = S_shrunk / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1, 1)
    return corr, round(alpha, 4)


# ---------------------------------------------------------------------------
# 6. PCA-based factor covariance
# ---------------------------------------------------------------------------

def pca_covariance(returns: np.ndarray,
                   n_components: int | None = None) -> dict:
    """Low-rank factor covariance via PCA.

    Decomposes S = V * Lambda * V' + D  (factor + diagonal)
    where D is the specific (idiosyncratic) risk diagonal.

    Parameters
    ----------
    n_components : number of PCA factors (None = auto from explained var ≥ 80%)
    """
    T, N = returns.shape
    if T < 3 or N < 2:
        return {"correlation": np.eye(N).tolist(), "n_components": 0,
                "explained_variance": [], "components": []}

    # Standardise returns
    mn = returns.mean(axis=0)
    sd = returns.std(axis=0, ddof=1)
    sd = np.where(sd < 1e-9, 1.0, sd)
    Z = (returns - mn) / sd

    # SVD of standardised returns
    U, S_vals, Vt = np.linalg.svd(Z, full_matrices=False)
    eigenvalues = S_vals ** 2 / (T - 1)
    total_var = eigenvalues.sum()
    explained_ratio = eigenvalues / max(total_var, 1e-12)

    # Auto-select components
    if n_components is None:
        cum_var = np.cumsum(explained_ratio)
        n_components = int(np.searchsorted(cum_var, 0.80)) + 1
        n_components = min(n_components, N // 2, T // 2, 20)
        n_components = max(n_components, 1)

    # Factor loadings (N, K)
    loadings = Vt[:n_components, :].T * np.sqrt(eigenvalues[:n_components])

    # Factor + specific covariance
    F_cov = loadings @ loadings.T
    S_sample = np.cov(Z.T, ddof=1)
    D = np.diag(np.maximum(np.diag(S_sample - F_cov), 1e-6))
    S_pca = F_cov + D

    # Convert to correlation
    std = np.sqrt(np.maximum(np.diag(S_pca), 1e-12))
    corr = S_pca / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1, 1)

    return {
        "correlation": [[round(float(x), 4) for x in row] for row in corr],
        "covariance": [[round(float(x), 6) for x in row] for row in S_pca],
        "n_components": n_components,
        "explained_variance_ratio": [round(float(x), 4) for x in explained_ratio[:n_components]],
        "cumulative_explained": [round(float(x), 4)
                                  for x in np.cumsum(explained_ratio[:n_components])],
        "factor_loadings": [[round(float(x), 4) for x in row] for row in loadings.T],
    }


# ---------------------------------------------------------------------------
# Correlation diagnostics
# ---------------------------------------------------------------------------

def correlation_diagnostics(corr_matrix: np.ndarray,
                              method_name: str = "") -> dict:
    """Assess quality of a correlation matrix.

    Checks:
    - Positive definiteness (all eigenvalues > 0)
    - Condition number (high → numerically ill-conditioned)
    - Average absolute off-diagonal correlation
    - Effective number of independent bets (Menchero et al.)
    """
    N = corr_matrix.shape[0]
    eigenvalues = np.linalg.eigvalsh(corr_matrix)
    min_eig = float(eigenvalues.min())
    max_eig = float(eigenvalues.max())
    is_pd = min_eig > 1e-8

    cond_number = max_eig / max(abs(min_eig), 1e-12) if not is_pd else max_eig / min_eig

    # Average absolute off-diagonal correlation
    mask = ~np.eye(N, dtype=bool)
    avg_abs_corr = float(np.mean(np.abs(corr_matrix[mask])))

    # Effective number of uncorrelated bets: sum(eigenvalues)^2 / sum(eigenvalues^2)
    evs = np.maximum(eigenvalues, 0)
    eff_bets = (evs.sum() ** 2) / max(np.sum(evs ** 2), 1e-12)

    return {
        "method": method_name,
        "n_assets": N,
        "is_positive_definite": bool(is_pd),
        "min_eigenvalue": round(min_eig, 6),
        "max_eigenvalue": round(max_eig, 4),
        "condition_number": round(cond_number, 2),
        "avg_abs_correlation": round(avg_abs_corr, 4),
        "effective_uncorrelated_bets": round(float(eff_bets), 2),
        "warning": (
            "Matrix is NOT positive definite — use shrinkage!" if not is_pd
            else "Ill-conditioned matrix — moderate shrinkage recommended" if cond_number > 1000
            else None
        ),
    }


# ---------------------------------------------------------------------------
# Main correlation engine — compute all methods and compare
# ---------------------------------------------------------------------------

def run_correlation_engine(returns: np.ndarray,
                           ticker_names: list[str] | None = None,
                           ewma_halflife: float = 60.0,
                           pca_components: int | None = None) -> dict:
    """Compute correlations using multiple methods and compare diagnostics.

    Parameters
    ----------
    returns : (T, N) return matrix
    ticker_names : optional list of asset names
    ewma_halflife : EWMA half-life in days
    pca_components : PCA factors (None = auto)

    Returns
    -------
    dict with all correlation matrices, diagnostics, and shrinkage intensities
    """
    T, N = returns.shape
    names = ticker_names or [f"Asset_{i}" for i in range(N)]

    # ── Compute all correlation estimators ──
    pearson = pearson_correlation(returns)
    spearman = spearman_correlation(returns)
    ewma = ewma_correlation(returns, halflife=ewma_halflife)
    lw, lw_alpha = ledoit_wolf_shrinkage(returns)
    oas, oas_alpha = oas_shrinkage(returns)
    pca_result = pca_covariance(returns, n_components=pca_components)
    pca_corr = np.array(pca_result["correlation"])

    # ── Diagnostics for each method ──
    diag = {
        "pearson": correlation_diagnostics(pearson, "pearson"),
        "spearman": correlation_diagnostics(spearman, "spearman"),
        "ewma": correlation_diagnostics(ewma, f"ewma_{ewma_halflife}d"),
        "ledoit_wolf": correlation_diagnostics(lw, "ledoit_wolf"),
        "oas": correlation_diagnostics(oas, "oas"),
        "pca": correlation_diagnostics(pca_corr, f"pca_{pca_result['n_components']}f"),
    }

    # Recommend best estimator
    recommended = _recommend_estimator(T, N, diag)

    def to_list(mat: np.ndarray) -> list[list[float]]:
        return [[round(float(x), 4) for x in row] for row in mat]

    return {
        "n_assets": N,
        "n_periods": T,
        "ratio_T_over_N": round(T / N, 2),
        "ticker_names": names,
        "correlations": {
            "pearson": to_list(pearson),
            "spearman": to_list(spearman),
            "ewma": to_list(ewma),
            "ledoit_wolf": to_list(lw),
            "oas": to_list(oas),
            "pca": pca_result["correlation"],
        },
        "shrinkage_intensities": {
            "ledoit_wolf": lw_alpha,
            "oas": oas_alpha,
        },
        "pca_info": {
            "n_components": pca_result["n_components"],
            "explained_variance": pca_result.get("explained_variance_ratio", []),
            "cumulative_explained": pca_result.get("cumulative_explained", []),
        },
        "diagnostics": diag,
        "recommended_estimator": recommended,
    }


def _recommend_estimator(T: int, N: int, diagnostics: dict) -> dict:
    """Recommend the most appropriate estimator based on data dimensions."""
    ratio = T / N
    if ratio < 3:
        rec = "oas"
        reason = f"T/N={ratio:.1f} < 3: sample covariance is ill-conditioned; use OAS shrinkage"
    elif ratio < 10:
        rec = "ledoit_wolf"
        reason = f"T/N={ratio:.1f}: moderate dimensional problem; Ledoit-Wolf shrinkage recommended"
    elif not diagnostics["pearson"]["is_positive_definite"]:
        rec = "ledoit_wolf"
        reason = "Sample correlation is not PD; apply shrinkage"
    else:
        rec = "ewma"
        reason = f"T/N={ratio:.1f}: sufficient data; EWMA captures regime dynamics"

    return {
        "estimator": rec,
        "reason": reason,
        "is_pd": diagnostics[rec]["is_positive_definite"],
    }
