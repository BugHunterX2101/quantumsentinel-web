"""QuantumSentinel — Cross-Sectional Factor Model.

Implements Fama-MacBeth (1973) two-pass regression for factor pricing:

  Pass 1: Time-series regressions of each asset's returns on factors
          → estimate factor loadings (betas) per asset
  Pass 2: Cross-sectional regression of returns on betas at each t
          → estimate factor risk premia (lambda) per period
  Inference: Newey-West standard errors on the time-series of lambdas

Built-in factors:
  - Momentum (12-1 month)
  - Value proxy (earnings yield / price reversal)
  - Size (log market cap proxy)
  - Volatility (realised vol)
  - Quality (Sharpe-like consistency score)
  - SBA Signal (proprietary)

Also includes:
  - Barra-style risk model skeleton (factor + specific risk)
  - Factor exposure matrix computation
  - Factor return time series
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Factor computation from price/return matrix
# ---------------------------------------------------------------------------

def compute_factors(return_matrix: np.ndarray,
                    price_matrix: np.ndarray | None = None,
                    lookbacks: dict | None = None) -> dict[str, np.ndarray]:
    """Compute cross-sectional factor exposures at each time step.

    Parameters
    ----------
    return_matrix : (T, N) daily returns
    price_matrix  : (T, N) price levels (optional)
    lookbacks     : dict of factor-specific lookback windows

    Returns
    -------
    dict mapping factor name → (T, N) exposure matrix
    """
    T, N = return_matrix.shape
    lb = lookbacks or {}

    factors: dict[str, np.ndarray] = {}

    # ── Momentum: past 60-day cumulative return ──
    mom_lb = lb.get("momentum", 60)
    mom = np.full((T, N), np.nan)
    for t in range(mom_lb, T):
        window = return_matrix[t - mom_lb:t, :]
        cum_ret = np.prod(1 + window, axis=0) - 1
        mom[t, :] = cum_ret
    factors["momentum"] = mom

    # ── Short-term reversal: past 5-day return ──
    rev_lb = lb.get("reversal", 5)
    rev = np.full((T, N), np.nan)
    for t in range(rev_lb, T):
        window = return_matrix[t - rev_lb:t, :]
        cum_ret = np.prod(1 + window, axis=0) - 1
        rev[t, :] = -cum_ret  # reversal = negative of recent return
    factors["reversal"] = rev

    # ── Volatility: 21-day realised vol ──
    vol_lb = lb.get("volatility", 21)
    vol = np.full((T, N), np.nan)
    for t in range(vol_lb, T):
        window = return_matrix[t - vol_lb:t, :]
        vol[t, :] = np.std(window, axis=0, ddof=1) * math.sqrt(252)
    factors["volatility"] = vol

    # ── Low volatility (inverse of vol — anomaly) ──
    low_vol = np.full((T, N), np.nan)
    valid = ~np.isnan(vol)
    low_vol[valid] = -vol[valid]
    factors["low_volatility"] = low_vol

    # ── Quality: Sharpe-like consistency (60-day) ──
    qual_lb = lb.get("quality", 60)
    quality = np.full((T, N), np.nan)
    for t in range(qual_lb, T):
        window = return_matrix[t - qual_lb:t, :]
        mn = np.mean(window, axis=0)
        sd = np.std(window, axis=0, ddof=1) + 1e-9
        quality[t, :] = mn / sd * math.sqrt(252)
    factors["quality"] = quality

    # ── Size proxy: log of price (crude proxy; full impl needs mkt cap) ──
    if price_matrix is not None:
        log_price = np.full((T, N), np.nan)
        valid_p = price_matrix > 0
        log_price[valid_p] = np.log(price_matrix[valid_p])
        factors["size"] = log_price
    else:
        # Use cumulative return as size proxy
        cum = np.full((T, N), np.nan)
        for t in range(min(252, T), T):
            window = return_matrix[max(0, t - 252):t, :]
            cum[t, :] = np.prod(1 + window, axis=0) - 1
        factors["size"] = cum

    # Z-score normalise each factor cross-sectionally at each t
    for name in list(factors.keys()):
        mat = factors[name]
        normed = np.full_like(mat, np.nan)
        for t in range(T):
            row = mat[t, :]
            valid = np.isfinite(row)
            if valid.sum() < 4:
                continue
            mu = np.mean(row[valid])
            sd = np.std(row[valid], ddof=1)
            if sd < 1e-9:
                normed[t, valid] = 0.0
            else:
                normed[t, valid] = np.clip((row[valid] - mu) / sd, -3, 3)
        factors[name] = normed

    return factors


# ---------------------------------------------------------------------------
# Fama-MacBeth two-pass regression
# ---------------------------------------------------------------------------

def fama_macbeth(return_matrix: np.ndarray,
                 factor_matrices: dict[str, np.ndarray],
                 newey_west_lags: int = 4) -> dict:
    """Fama-MacBeth (1973) cross-sectional factor risk premia estimation.

    Pass 1 — estimate betas for each asset (rolling 60-bar OLS):
        R_{i,t} = alpha_i + sum_k beta_{i,k} * F_{k,t} + eps

    Pass 2 — cross-sectional regression each period:
        R_{i,t} = gamma_0 + sum_k lambda_k * beta_{i,k} + eps

    Inference: lambda_k = time-series mean of gamma_k
               SE(lambda_k) via Newey-West (accounts for autocorrelation)

    Parameters
    ----------
    return_matrix : (T, N) asset returns
    factor_matrices : dict of factor_name → (T, N) exposure matrices
    newey_west_lags : NW lag length for inference

    Returns
    -------
    dict with factor premia, t-stats, significance, R² decomposition
    """
    T, N = return_matrix.shape
    factor_names = list(factor_matrices.keys())
    K = len(factor_names)

    if T < 80 or N < K + 2:
        return {
            "error": "Insufficient data for Fama-MacBeth regression",
            "required_T": 80, "required_N": K + 2,
            "actual_T": T, "actual_N": N,
        }

    beta_lb = 60  # rolling window for Pass 1 betas
    gammas = {k: [] for k in factor_names}  # store cross-sec regression coeffs
    r2_series = []

    for t in range(beta_lb, T - 1):
        # ── Pass 1: estimate betas up to time t ──
        betas = np.full((N, K), np.nan)
        for i in range(N):
            y = return_matrix[max(0, t - beta_lb):t, i]
            X_cols = []
            for k, fname in enumerate(factor_names):
                f = factor_matrices[fname][max(0, t - beta_lb):t, i]
                X_cols.append(f)
            if not X_cols:
                continue
            X = np.column_stack(X_cols)
            valid = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
            if valid.sum() < K + 2:
                continue
            Xv, yv = X[valid], y[valid]
            # OLS: beta = (X'X)^-1 X'y
            try:
                XtX = Xv.T @ Xv
                XtXinv = np.linalg.pinv(XtX)
                beta_hat = XtXinv @ (Xv.T @ yv)
                betas[i, :] = beta_hat
            except Exception:
                continue

        # ── Pass 2: cross-sectional regression at t+1 ──
        y_xsec = return_matrix[t + 1, :]  # next period returns
        valid_assets = np.all(np.isfinite(betas), axis=1) & np.isfinite(y_xsec)
        if valid_assets.sum() < K + 2:
            continue

        B = betas[valid_assets, :]     # (n_valid, K)
        Y = y_xsec[valid_assets]       # (n_valid,)

        # Add intercept (market-wide return)
        Bx = np.column_stack([np.ones(len(Y)), B])
        try:
            BtBinv = np.linalg.pinv(Bx.T @ Bx)
            gamma = BtBinv @ (Bx.T @ Y)  # [intercept, lam_1, ..., lam_K]
        except Exception:
            continue

        # R² of cross-sectional fit
        y_hat = Bx @ gamma
        ss_res = np.sum((Y - y_hat) ** 2)
        ss_tot = np.sum((Y - Y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        r2_series.append(max(0.0, r2))

        for k, fname in enumerate(factor_names):
            gammas[fname].append(float(gamma[k + 1]))  # skip intercept

    if not any(gammas[k] for k in factor_names):
        return {"error": "No valid cross-sectional regressions completed"}

    # ── Inference: Newey-West SE on time series of gammas ──
    results = {}
    for fname in factor_names:
        g = np.array(gammas[fname])
        if len(g) < 5:
            results[fname] = {"lambda": 0, "t_stat": 0, "p_value": 1.0, "n": 0}
            continue
        lam = float(np.mean(g))
        nw_se = _nw_se(g, newey_west_lags)
        t_stat = lam / nw_se if nw_se > 1e-12 else 0.0
        from backend.services.stat_tests import _normal_cdf
        p_value = 2.0 * (1.0 - _normal_cdf(abs(t_stat)))
        results[fname] = {
            "lambda": round(lam, 6),
            "lambda_annualised": round(lam * 252, 4),
            "nw_se": round(nw_se, 6),
            "t_stat": round(t_stat, 4),
            "p_value": round(p_value, 6),
            "significant_5pct": p_value < 0.05,
            "significant_1pct": p_value < 0.01,
            "n_periods": len(g),
            "time_series": [round(float(x), 6) for x in g[-50:]],  # last 50 for charts
        }

    # Sort factors by |t-stat| descending
    ranked = sorted(
        [(fname, results[fname].get("t_stat", 0)) for fname in factor_names],
        key=lambda x: abs(x[1]), reverse=True
    )

    return {
        "factor_premia": results,
        "factors_ranked_by_significance": [{"factor": f, "t_stat": round(t, 4)}
                                            for f, t in ranked],
        "mean_cross_sectional_r2": round(float(np.mean(r2_series)) if r2_series else 0, 4),
        "n_cross_sections": len(r2_series),
        "n_assets": N,
        "factor_names": factor_names,
    }


def _nw_se(x: np.ndarray, max_lag: int) -> float:
    """Newey-West standard error."""
    n = len(x)
    if n < 3:
        return float(np.std(x, ddof=1) / math.sqrt(n)) if n > 1 else 1.0
    mean = np.mean(x)
    resid = x - mean
    gamma0 = float(np.sum(resid ** 2)) / n
    nw_var = gamma0
    for lag in range(1, max_lag + 1):
        w = 1.0 - lag / (max_lag + 1)
        gamma_k = float(np.sum(resid[lag:] * resid[:-lag])) / n
        nw_var += 2 * w * gamma_k
    nw_var = max(nw_var, 1e-12)
    return math.sqrt(nw_var / n)


# ---------------------------------------------------------------------------
# Risk model: factor + specific (idiosyncratic) risk
# ---------------------------------------------------------------------------

def barra_risk_decomposition(return_matrix: np.ndarray,
                              factor_matrices: dict[str, np.ndarray],
                              beta_lb: int = 60) -> dict:
    """Decompose portfolio variance into factor and specific components.

    Factor risk = beta' * F_cov * beta
    Specific risk = residual variance not explained by factors

    Useful for risk-budgeting and factor exposure reporting.
    """
    T, N = return_matrix.shape
    factor_names = list(factor_matrices.keys())
    K = len(factor_names)

    if T < beta_lb + 10 or N < K + 2:
        return {"error": "Insufficient data"}

    # Use the last beta_lb periods
    t = T - 1
    betas = np.full((N, K), np.nan)
    resid_vars = np.full(N, np.nan)

    for i in range(N):
        y = return_matrix[t - beta_lb:t, i]
        X_cols = [factor_matrices[fname][t - beta_lb:t, i] for fname in factor_names]
        X = np.column_stack(X_cols)
        valid = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
        if valid.sum() < K + 2:
            continue
        Xv, yv = X[valid], y[valid]
        try:
            beta_hat = np.linalg.lstsq(Xv, yv, rcond=None)[0]
            betas[i, :] = beta_hat
            resid = yv - Xv @ beta_hat
            resid_vars[i] = float(np.var(resid, ddof=1))
        except Exception:
            continue

    # Factor covariance matrix (from factor time series)
    factor_ts = np.full((T, K), np.nan)
    for k, fname in enumerate(factor_names):
        # Compute equal-weighted factor return: mean of factor × asset_return
        fmat = factor_matrices[fname]
        for t2 in range(T):
            row_f = fmat[t2, :]
            row_r = return_matrix[t2, :]
            valid = np.isfinite(row_f) & np.isfinite(row_r)
            if valid.sum() > 0:
                factor_ts[t2, k] = float(np.mean(row_f[valid] * row_r[valid]))

    valid_t = np.all(np.isfinite(factor_ts), axis=1)
    if valid_t.sum() < 10:
        return {"error": "Insufficient factor time series"}

    F_cov = np.cov(factor_ts[valid_t, :].T)  # (K, K) factor covariance
    if K == 1:
        F_cov = F_cov.reshape(1, 1)

    # Per-asset factor variance
    valid_assets = np.all(np.isfinite(betas), axis=1)
    asset_factor_var = []
    asset_specific_var = []
    for i in range(N):
        if not valid_assets[i]:
            continue
        b = betas[i, :]
        fv = float(b @ F_cov @ b)
        sv = float(resid_vars[i]) if np.isfinite(resid_vars[i]) else 0.0
        asset_factor_var.append(fv)
        asset_specific_var.append(sv)

    if not asset_factor_var:
        return {"error": "No valid assets"}

    avg_fv = float(np.mean(asset_factor_var))
    avg_sv = float(np.mean(asset_specific_var))
    total = avg_fv + avg_sv
    pct_factor = avg_fv / total if total > 0 else 0

    return {
        "factor_covariance_matrix": [[round(float(x), 6) for x in row] for row in F_cov],
        "factor_names": factor_names,
        "avg_factor_variance": round(avg_fv * 252, 6),
        "avg_specific_variance": round(avg_sv * 252, 6),
        "pct_factor_explained": round(pct_factor, 4),
        "pct_specific": round(1 - pct_factor, 4),
        "n_valid_assets": int(valid_assets.sum()),
    }
