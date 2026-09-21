"""QuantumSentinel — Market-Neutral & Factor-Neutral Strategy Engine.

Implements institutional market-neutral strategies:

  1. Statistical Arbitrage (pairs trading)
     - Cointegration test (Engle-Granger two-step)
     - Spread modelling: Z-score entry/exit signals
     - Kalman filter hedge ratio tracker

  2. Dollar-Neutral Long/Short
     - Cross-sectional momentum: long top decile, short bottom decile
     - Dollar-neutral: sum of positions = 0
     - Gross exposure constraint

  3. Factor-Neutral Strategy
     - Remove market beta from signal weights
     - Orthogonalise positions w.r.t. risk factors

  4. Long/Short Equity
     - Signal-ranked asset universe
     - Top N long, bottom N short
     - Volatility-scaled position sizing

All strategies produce:
  - position_weights: dict[asset → weight] (sum-to-zero for L/S)
  - gross_exposure: |longs| + |shorts|
  - net_exposure: longs + shorts
  - expected_alpha: signal-weighted return forecast
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kalman filter for hedge ratio estimation
# ---------------------------------------------------------------------------

class KalmanHedgeFilter:
    """Kalman filter for time-varying hedge ratio in pairs trading.

    State: beta (hedge ratio) — treated as random walk
    Observation: y_t = beta_t * x_t + epsilon_t

    Reference: Pole, West & Harrison (1994)
    """

    def __init__(self, delta: float = 1e-4, R: float = 1e-3):
        """
        delta : state noise (controls how fast beta changes)
        R     : observation noise
        """
        self.delta = delta  # state transition noise variance
        self.R = R          # observation noise variance
        self.beta = 0.0     # current hedge ratio estimate
        self.P = 1.0        # state covariance

    def update(self, y: float, x: float) -> float:
        """Update Kalman filter with one new observation.

        Returns updated hedge ratio estimate.
        """
        # Predict
        P_pred = self.P + self.delta

        # Innovation
        y_pred = self.beta * x
        innovation = y - y_pred
        innov_var = x ** 2 * P_pred + self.R

        # Kalman gain
        K = P_pred * x / max(innov_var, 1e-12)

        # Update
        self.beta = self.beta + K * innovation
        self.P = (1 - K * x) * P_pred

        return self.beta

    def fit(self, y: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Fit to full time series. Returns the posterior hedge-ratio
        history beta_{t|t} — the estimate *after* incorporating (y_t, x_t).
        Useful for reporting/diagnostics (e.g. convergence to a true beta),
        but NOT for building a tradeable spread — see fit_predictive().
        """
        betas = np.zeros(len(y))
        for t in range(len(y)):
            betas[t] = self.update(float(y[t]), float(x[t]))
        return betas

    def fit_predictive(self, y: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Fit to full time series. Returns the PRIOR hedge-ratio history
        beta_{t|t-1} — the estimate as of just *before* observing (y_t, x_t).

        This is what a tradeable spread must be built from. The posterior
        beta_{t|t} (returned by fit()) is chosen by the Kalman update
        specifically to explain away the very observation being tested, so
        using it makes the residual y_t - x_t*beta_{t|t} self-referential:
        algebraically it equals the true one-step-ahead innovation scaled
        by R/(x_t^2*P_pred+R) < 1, which collapses toward zero (empirically
        by ~3 orders of magnitude with typical delta/R) rather than
        reflecting a genuine, live-tradeable mispricing signal.
        """
        betas = np.zeros(len(y))
        for t in range(len(y)):
            betas[t] = self.beta  # prior, before this step's update
            self.update(float(y[t]), float(x[t]))
        return betas


# ---------------------------------------------------------------------------
# Cointegration test (Engle-Granger two-step, pure Python)
# ---------------------------------------------------------------------------

# MacKinnon (2010) response-surface coefficients for the Engle-Granger
# residual-based cointegration test with N=2 (one regressor + a constant in
# the first-stage OLS), regression="c" (constant, no trend). These are NOT
# the plain Dickey-Fuller "with constant" critical values (-3.43/-2.86/-2.57)
# — the EG test's critical values are more negative because the residual
# series being tested was itself derived from an estimated cointegrating
# vector, which adds estimation uncertainty that the plain ADF table doesn't
# account for.
#
# Finite-sample critical value: crit(T) = beta_inf + beta_1/T + beta_2/T^2.
# Coefficients below are MacKinnon (2010, Queen's Economics WP 1227) Table 2,
# N=2/"c" row — cross-checked against statsmodels' `tau_2010s['c'][1]` table,
# which digitises the same paper. The asymptotic-only values previously
# hardcoded here (-3.96/-3.37/-3.04) diverge from the true T→∞ limit
# (-3.896/-3.336/-3.044 respectively) and, more importantly, ignore that the
# finite-sample correction is large at the T~30-500 daily-bar windows this
# module actually runs on (e.g. at T=60 the true 5% critical value is
# -3.440, not -3.37 — a fixed asymptotic number understates how strict the
# test should be for short lookback windows and overstates it for long ones).
_EG_RS_COEFS = {
    0.01: (-3.89644, -10.9519, -33.527),
    0.05: (-3.33613, -6.1101, -6.823),
    0.10: (-3.04445, -4.2412, -2.72),
}


def _eg_critical_value(T: int, pct: float) -> float:
    """MacKinnon (2010) finite-sample-adjusted Engle-Granger critical value."""
    b_inf, b1, b2 = _EG_RS_COEFS[pct]
    return b_inf + b1 / T + b2 / T ** 2


def engle_granger_cointegration(y: np.ndarray, x: np.ndarray,
                                 max_lag: int = 1) -> dict:
    """Engle-Granger (1987) two-step cointegration test.

    Step 1: OLS regression y = alpha + beta * x + spread
    Step 2: ADF test on residual spread

    Returns p-value approximation (MacKinnon cointegration critical values).
    """
    T = len(y)
    if T < 30:
        return {"cointegrated": False, "error": "Insufficient data"}

    # Step 1: OLS
    X = np.column_stack([np.ones(T), x])
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
    except Exception:
        return {"cointegrated": False, "error": "OLS failed"}

    spread = y - X @ beta
    hedge_ratio = float(beta[1])
    alpha = float(beta[0])

    # Finite-sample-adjusted critical values (MacKinnon 2010 response
    # surface) computed from the original cointegrating-regression sample
    # size T — the convention the response surface itself was fit against.
    cv_1pct = _eg_critical_value(T, 0.01)
    cv_5pct = _eg_critical_value(T, 0.05)
    cv_10pct = _eg_critical_value(T, 0.10)

    # Step 2: ADF test on spread (no-constant, no-trend — the residual is
    # already mean-zero by construction from the step-1 OLS with intercept).
    adf_stat, adf_p = _eg_residual_adf_test(spread, cv_1pct, cv_5pct, cv_10pct, max_lag=max_lag)

    return {
        "cointegrated": adf_stat < cv_5pct,
        "cointegrated_1pct": adf_stat < cv_1pct,
        "adf_stat": round(adf_stat, 4),
        "adf_p_approx": round(adf_p, 4),
        "critical_value_1pct": round(cv_1pct, 4),
        "critical_value_5pct": round(cv_5pct, 4),
        "critical_value_10pct": round(cv_10pct, 4),
        "hedge_ratio": round(hedge_ratio, 6),
        "alpha": round(alpha, 6),
        "spread_mean": round(float(spread.mean()), 6),
        "spread_std": round(float(spread.std(ddof=1)), 6),
        "spread": spread.tolist(),
    }


def _eg_residual_adf_test(series: np.ndarray, cv_1pct: float, cv_5pct: float,
                           cv_10pct: float, max_lag: int = 1) -> tuple[float, float]:
    """(Augmented) Dickey-Fuller t-statistic for a unit root in an
    Engle-Granger cointegrating residual series, with a p-value
    approximation against the Engle-Granger (not plain-ADF) critical
    values.

    This is deliberately not a general-purpose ADF test: the regression run
    here has no constant term (see below) and the p-value bucketing uses
    the same finite-sample-adjusted cv_1pct/5pct/10pct the caller used for
    the "cointegrated" decision — the correct table for testing residuals of
    an estimated cointegrating relationship, not the plain "no constant"
    Dickey-Fuller table (whose asymptotic 5% value is ≈ -1.95, a very
    different number). Reusing this helper for a standalone unit-root test
    on a raw (non-residual) series would silently apply the wrong table.
    """
    T = len(series)
    dy = np.diff(series)
    y_lag = series[max_lag:-1] if max_lag > 0 else series[:-1]
    dy_dep = dy[max_lag:] if max_lag > 0 else dy

    # Build regressor matrix: [y_{t-1}, Δy_{t-1}, Δy_{t-2}, ...]
    X = y_lag.reshape(-1, 1)
    for lag in range(1, max_lag + 1):
        # Δy_{t-lag} aligned with dy_dep
        col = dy[max_lag - lag: len(dy) - lag] if lag <= max_lag else np.zeros(len(dy_dep))
        if len(col) == len(dy_dep):
            X = np.column_stack([X, col])

    y = dy_dep

    if len(y) < 5 or X.shape[0] != len(y):
        return 0.0, 1.0

    try:
        betas, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ betas
        resid = y - y_hat
        s2 = float(np.sum(resid ** 2)) / max(len(y) - X.shape[1], 1)
        XtXinv = np.linalg.pinv(X.T @ X)
        se_beta = math.sqrt(max(s2 * XtXinv[0, 0], 1e-12))
        t_stat = float(betas[0]) / se_beta
    except Exception:
        return 0.0, 1.0

    # Approximate p-value via the *same* Engle-Granger critical values used
    # for the "cointegrated" / "cointegrated_1pct" decision above — using a
    # different table here (as a previous version of this function did, with
    # the plain "constant, no trend" ADF critical values -3.48/-2.87/-2.57)
    # let adf_p_approx contradict the cointegrated flag: a spread could be
    # flagged cointegrated at 5% while its own reported p-value read back
    # as > 5%, because the two thresholds weren't the same test.
    if t_stat < cv_1pct:
        p = 0.005
    elif t_stat < cv_5pct:
        p = 0.025
    elif t_stat < cv_10pct:
        p = 0.075
    else:
        p = 0.20

    return t_stat, p


# ---------------------------------------------------------------------------
# Pairs Trading Signal Generator
# ---------------------------------------------------------------------------

def pairs_trading_signals(y: np.ndarray, x: np.ndarray,
                           entry_z: float = 2.0,
                           exit_z: float = 0.5,
                           use_kalman: bool = True,
                           window: int = 60) -> dict:
    """Generate entry/exit signals for a pairs trade.

    If use_kalman=True: uses Kalman filter for adaptive hedge ratio.
    Otherwise: rolling OLS with fixed window.

    Signal logic:
      +1: long y, short x (spread < -entry_z * std)
      -1: short y, long x (spread > +entry_z * std)
       0: flat (|spread| < exit_z * std)
    """
    T = len(y)
    if T < window + 10:
        return {"error": "Insufficient data for pairs trading"}

    # Cointegration test
    coint = engle_granger_cointegration(y, x)

    # Hedge ratio time series
    if use_kalman:
        kf = KalmanHedgeFilter(delta=1e-4, R=1e-3)
        # Use the PRIOR (pre-update) beta for each step, not the posterior
        # fit() would return — see fit_predictive()'s docstring for why the
        # posterior makes the spread self-referential and non-tradeable.
        hedge_ratios = kf.fit_predictive(y, x)
    else:
        hedge_ratios = np.full(T, np.nan)
        for t in range(window, T):
            X = np.column_stack([np.ones(window), x[t - window:t]])
            yy = y[t - window:t]
            try:
                b = np.linalg.lstsq(X, yy, rcond=None)[0]
                hedge_ratios[t] = b[1]
            except Exception:
                pass

    # Spread = y - hedge_ratio * x
    spread = y - hedge_ratios * x

    # Z-score spread (rolling)
    z_spread = np.full(T, np.nan)
    for t in range(window, T):
        w = spread[t - window:t]
        mn = np.mean(w)
        sd = np.std(w, ddof=1)
        if sd > 1e-9:
            z_spread[t] = (spread[t] - mn) / sd

    # Signals
    signals = np.zeros(T)
    position = 0
    for t in range(window, T):
        if not np.isfinite(z_spread[t]):
            continue
        z = z_spread[t]
        if position == 0:
            if z < -entry_z:
                position = 1   # long spread
            elif z > entry_z:
                position = -1  # short spread
        elif position == 1:
            if z > -exit_z:
                position = 0
        elif position == -1:
            if z < exit_z:
                position = 0
        signals[t] = position

    # P&L from signals
    spread_returns = np.diff(spread) / np.maximum(np.abs(spread[:-1]), 1e-9)
    strategy_returns = signals[1:] * spread_returns
    valid_r = strategy_returns[np.isfinite(strategy_returns)]

    sharpe = 0.0
    if len(valid_r) > 5 and valid_r.std(ddof=1) > 1e-9:
        sharpe = float(valid_r.mean() / valid_r.std(ddof=1) * math.sqrt(252))

    return {
        "cointegration": coint,
        "signals": [int(s) for s in signals],
        "z_spread": [round(float(z), 4) if np.isfinite(z) else None for z in z_spread],
        "hedge_ratios": [round(float(h), 6) if np.isfinite(h) else None for h in hedge_ratios],
        "entry_z": entry_z,
        "exit_z": exit_z,
        "n_trades": int((np.diff(signals) != 0).sum()),
        "strategy_sharpe": round(sharpe, 4),
        "use_kalman": use_kalman,
    }


# ---------------------------------------------------------------------------
# Cross-Sectional Long/Short Equity Strategy
# ---------------------------------------------------------------------------

def long_short_equity(signal_matrix: np.ndarray,
                      return_matrix: np.ndarray,
                      n_long: int = 10,
                      n_short: int = 10,
                      vol_scale: bool = True,
                      target_vol: float = 0.10,
                      asset_names: list[str] | None = None) -> dict:
    """Cross-sectional long/short equity strategy.

    At each period:
      - Rank assets by signal
      - Long top n_long, short bottom n_short
      - Dollar-neutral (sum weights = 0)
      - Optionally scale to target portfolio volatility

    Parameters
    ----------
    signal_matrix : (T, N) signal exposures
    return_matrix : (T, N) asset returns
    n_long, n_short : # long / short positions
    vol_scale : scale portfolio to target_vol
    target_vol : annualised target volatility
    """
    T, N = return_matrix.shape
    names = asset_names or [f"A{i}" for i in range(N)]

    portfolio_returns = []
    turnover = []
    prev_weights = np.zeros(N)

    for t in range(1, T):
        sig = signal_matrix[t - 1, :]
        valid = np.isfinite(sig)
        if valid.sum() < n_long + n_short + 2:
            portfolio_returns.append(0.0)
            turnover.append(0.0)
            continue

        # Rank and select
        ranked_idx = np.argsort(sig)
        short_idx = ranked_idx[:n_short]     # lowest signal → short
        long_idx = ranked_idx[-n_long:]      # highest signal → long

        weights = np.zeros(N)
        weights[long_idx] = 1.0 / n_long
        weights[short_idx] = -1.0 / n_short

        # Dollar neutral: already balanced if n_long * w == n_short * |w|
        # But normalise by gross exposure
        gross = np.abs(weights).sum()
        if gross > 1e-9:
            weights = weights / gross  # gross exposure = 1

        # Volatility scaling
        if vol_scale and t >= 22:
            hist_rets = (return_matrix[max(0, t - 21):t, :] * prev_weights[np.newaxis, :]).sum(axis=1)
            hist_vol = float(np.std(hist_rets, ddof=1) * math.sqrt(252))
            if hist_vol > 1e-9:
                scale = min(target_vol / hist_vol, 3.0)  # cap at 3x leverage
                weights = weights * scale

        # Portfolio return
        ret_t = float(return_matrix[t, :] @ weights)
        portfolio_returns.append(ret_t)

        # Turnover
        to = float(np.abs(weights - prev_weights).sum()) / 2.0
        turnover.append(to)
        prev_weights = weights.copy()

    pr = np.array(portfolio_returns)
    valid_pr = pr[np.isfinite(pr)]

    if len(valid_pr) < 10:
        return {"error": "Not enough valid portfolio returns"}

    ann_ret = float(valid_pr.mean() * 252)
    ann_vol = float(valid_pr.std(ddof=1) * math.sqrt(252))
    sharpe = ann_ret / max(ann_vol, 1e-9)

    # Max drawdown
    cum = np.cumprod(1 + valid_pr)
    running_max = np.maximum.accumulate(cum)
    dd = (cum - running_max) / np.maximum(running_max, 1e-9)
    max_dd = float(dd.min())

    # Monthly returns (approx 21-day buckets)
    n_months = len(valid_pr) // 21
    monthly_rets = [float(np.prod(1 + valid_pr[i*21:(i+1)*21]) - 1) for i in range(n_months)]

    return {
        "n_long": n_long,
        "n_short": n_short,
        "vol_scaled": vol_scale,
        "target_vol": target_vol,
        "annual_return": round(ann_ret, 4),
        "annual_volatility": round(ann_vol, 4),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
        "avg_daily_turnover": round(float(np.mean(turnover)), 4),
        "n_periods": len(valid_pr),
        "portfolio_returns": [round(float(r), 6) for r in portfolio_returns[-252:]],  # last year
        "monthly_returns": [round(r, 4) for r in monthly_rets],
    }


# ---------------------------------------------------------------------------
# Factor-neutral weight construction
# ---------------------------------------------------------------------------

def factor_neutralise(weights: np.ndarray,
                      factor_exposures: np.ndarray) -> np.ndarray:
    """Remove factor exposure from weight vector.

    Projects weights onto the null space of factor_exposures:
        w_neutral = w - B * (B'B)^-1 B' w

    where B is the (N, K) factor exposure matrix.

    Parameters
    ----------
    weights : (N,) weight vector
    factor_exposures : (N, K) factor exposure matrix

    Returns
    -------
    (N,) factor-neutral weights (sum may not be 1 after neutralisation)
    """
    if factor_exposures.ndim == 1:
        factor_exposures = factor_exposures.reshape(-1, 1)

    N, K = factor_exposures.shape
    if N != len(weights):
        raise ValueError("Weight and exposure dimensions mismatch")

    # Project out factor exposure
    try:
        BtBinv = np.linalg.pinv(factor_exposures.T @ factor_exposures)
        proj = factor_exposures @ BtBinv @ factor_exposures.T
        w_neutral = weights - proj @ weights
    except Exception:
        w_neutral = weights.copy()

    # Rescale so gross exposure matches original
    orig_gross = np.abs(weights).sum()
    new_gross = np.abs(w_neutral).sum()
    if new_gross > 1e-9 and orig_gross > 1e-9:
        w_neutral = w_neutral * (orig_gross / new_gross)

    return w_neutral


# ---------------------------------------------------------------------------
# Run neutral strategy pipeline
# ---------------------------------------------------------------------------

def run_neutral_strategies(signal_matrix: np.ndarray,
                            return_matrix: np.ndarray,
                            asset_names: list[str] | None = None,
                            factor_exposures: np.ndarray | None = None) -> dict:
    """Run cross-sectional long/short strategy with optional factor neutralisation.

    Parameters
    ----------
    signal_matrix : (T, N) signal matrix (e.g., momentum z-scores)
    return_matrix : (T, N) asset returns
    asset_names : optional ticker list
    factor_exposures : optional (N, K) factor exposure for neutralisation

    Returns
    -------
    Dict with base L/S result and factor-neutral L/S result (if exposures provided)
    """
    T, N = return_matrix.shape
    n_long = min(max(N // 5, 3), 15)
    n_short = n_long

    # Base long/short
    base = long_short_equity(signal_matrix, return_matrix,
                              n_long=n_long, n_short=n_short,
                              vol_scale=True, asset_names=asset_names)

    result: dict = {"base_long_short": base}

    # Factor-neutral version (if factor exposures available)
    if factor_exposures is not None and factor_exposures.shape[0] == N:
        # Neutralise signal matrix at each period
        neutral_signal = np.full_like(signal_matrix, np.nan)
        for t in range(T):
            sig = signal_matrix[t, :]
            valid = np.isfinite(sig)
            if valid.sum() > factor_exposures.shape[1] + 2:
                s_filled = np.where(valid, sig, 0.0)
                try:
                    neutral_signal[t, :] = factor_neutralise(s_filled, factor_exposures)
                    neutral_signal[t, ~valid] = np.nan
                except Exception:
                    neutral_signal[t, :] = sig
            else:
                neutral_signal[t, :] = sig

        factor_neutral = long_short_equity(neutral_signal, return_matrix,
                                            n_long=n_long, n_short=n_short,
                                            vol_scale=True, asset_names=asset_names)
        result["factor_neutral_long_short"] = factor_neutral

    return result
