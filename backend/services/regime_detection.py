"""QuantumSentinel — Market Regime Detection Engine.

Identifies market regimes using multiple statistical methods:

  1. Hidden Markov Model (HMM) — Gaussian emissions on returns/vol
     - Trained with Baum-Welch / Viterbi for state decoding
     - Pure-Python implementation (no hmmlearn needed for production)

  2. Volatility Regime (GARCH-proxy) — rolling vol percentiles
     - Low / medium / high volatility states

  3. Trend Regime — simple moving average regime
     - Bull / Bear / Sideways based on SMA crossovers + slope

  4. Correlation Regime — rolling pairwise correlation shift
     - Risk-on / Risk-off detection

Each detector produces:
  - regime_labels: array of state labels per time step
  - regime_probs: posterior probabilities per state
  - regime_stats: per-regime return, vol, Sharpe statistics
  - current_regime: most recent state
  - transition_matrix: probability of switching states
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class VolRegime(str, Enum):
    LOW = "low_vol"
    MED = "med_vol"
    HIGH = "high_vol"
    CRISIS = "crisis"


class TrendRegime(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"


class RiskRegime(str, Enum):
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    NEUTRAL = "neutral"


# ---------------------------------------------------------------------------
# 1. Pure-Python 2-state Gaussian HMM
# ---------------------------------------------------------------------------

class GaussianHMM2State:
    """2-state Gaussian HMM trained with Baum-Welch EM.

    States typically correspond to:
      State 0: Low-volatility / Bull regime
      State 1: High-volatility / Bear / Crisis regime

    Parameters
    ----------
    n_iter : max EM iterations
    tol    : convergence tolerance on log-likelihood
    """

    def __init__(self, n_iter: int = 100, tol: float = 1e-4):
        self.n_iter = n_iter
        self.tol = tol
        self.n_states = 2
        # Model parameters (initialised in fit)
        self.pi = np.array([0.5, 0.5])          # initial state probs
        self.A = np.array([[0.95, 0.05],         # transition matrix
                           [0.10, 0.90]])
        self.means = np.zeros(2)
        self.stds = np.ones(2)

    def _emission_probs(self, x: float) -> np.ndarray:
        """Gaussian emission probability for each state."""
        p = np.array([
            _gaussian_pdf(x, self.means[k], self.stds[k])
            for k in range(self.n_states)
        ])
        return np.maximum(p, 1e-300)

    def fit(self, observations: np.ndarray) -> "GaussianHMM2State":
        """Baum-Welch EM algorithm."""
        T = len(observations)
        K = self.n_states

        # Initialise means/stds from k-means-like split
        sorted_obs = np.sort(observations)
        self.means = np.array([
            float(np.mean(sorted_obs[:T // 2])),
            float(np.mean(sorted_obs[T // 2:])),
        ])
        self.stds = np.array([
            max(float(np.std(sorted_obs[:T // 2])), 1e-6),
            max(float(np.std(sorted_obs[T // 2:])), 1e-6),
        ])

        prev_ll = -np.inf

        for iteration in range(self.n_iter):
            # ── E-step: Forward-Backward ──
            alpha = np.zeros((T, K))
            beta = np.zeros((T, K))
            scale = np.zeros(T)

            # Forward pass
            em = np.array([self._emission_probs(observations[0])])
            alpha[0] = self.pi * em[0]
            scale[0] = alpha[0].sum()
            alpha[0] /= max(scale[0], 1e-300)

            for t in range(1, T):
                em_t = self._emission_probs(observations[t])
                alpha[t] = (alpha[t - 1] @ self.A) * em_t
                scale[t] = alpha[t].sum()
                alpha[t] /= max(scale[t], 1e-300)

            # Backward pass
            beta[T - 1] = 1.0
            for t in range(T - 2, -1, -1):
                em_t1 = self._emission_probs(observations[t + 1])
                beta[t] = self.A @ (em_t1 * beta[t + 1])
                beta[t] /= max(beta[t].sum(), 1e-300)

            # Posterior state probabilities (gamma)
            gamma = alpha * beta
            gamma_sum = gamma.sum(axis=1, keepdims=True)
            gamma /= np.maximum(gamma_sum, 1e-300)

            # Xi (joint probability of consecutive states)
            xi_sum = np.zeros((K, K))
            for t in range(T - 1):
                em_t1 = self._emission_probs(observations[t + 1])
                xi_t = np.outer(alpha[t], em_t1 * beta[t + 1]) * self.A
                xi_t /= max(xi_t.sum(), 1e-300)
                xi_sum += xi_t

            # ── M-step: Update parameters ──
            self.pi = gamma[0]
            # Transition matrix
            for i in range(K):
                row_sum = xi_sum[i].sum()
                if row_sum > 1e-12:
                    self.A[i] = xi_sum[i] / row_sum
            # Emission parameters
            for k in range(K):
                w = gamma[:, k]
                w_sum = max(w.sum(), 1e-12)
                self.means[k] = float((w * observations).sum() / w_sum)
                var = float((w * (observations - self.means[k]) ** 2).sum() / w_sum)
                self.stds[k] = max(math.sqrt(var), 1e-6)

            # Log-likelihood for convergence check
            ll = float(np.sum(np.log(np.maximum(scale, 1e-300))))
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        return self

    def predict(self, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Viterbi decoding — most likely state sequence.

        Returns (state_sequence, posterior_probs)
        """
        T = len(observations)
        K = self.n_states

        # Viterbi
        viterbi = np.zeros((T, K))
        backptr = np.zeros((T, K), dtype=int)

        em_0 = self._emission_probs(observations[0])
        viterbi[0] = np.log(np.maximum(self.pi, 1e-300)) + np.log(em_0)

        log_A = np.log(np.maximum(self.A, 1e-300))
        for t in range(1, T):
            em_t = self._emission_probs(observations[t])
            log_em_t = np.log(em_t)
            for k in range(K):
                candidates = viterbi[t - 1] + log_A[:, k]
                backptr[t, k] = int(np.argmax(candidates))
                viterbi[t, k] = candidates[backptr[t, k]] + log_em_t[k]

        # Backtrack
        states = np.zeros(T, dtype=int)
        states[T - 1] = int(np.argmax(viterbi[T - 1]))
        for t in range(T - 2, -1, -1):
            states[t] = backptr[t + 1, states[t + 1]]

        # Posterior probs via forward-backward (simplified: use emission ratio)
        alpha = np.zeros((T, K))
        scale = np.zeros(T)
        em_0 = self._emission_probs(observations[0])
        alpha[0] = self.pi * em_0
        scale[0] = alpha[0].sum()
        alpha[0] /= max(scale[0], 1e-300)
        for t in range(1, T):
            em_t = self._emission_probs(observations[t])
            alpha[t] = (alpha[t - 1] @ self.A) * em_t
            scale[t] = alpha[t].sum()
            alpha[t] /= max(scale[t], 1e-300)

        return states, alpha  # alpha serves as approx. posterior

    def steady_state(self) -> np.ndarray:
        """Compute steady-state distribution from transition matrix."""
        # Solve pi @ A = pi, sum(pi) = 1
        # Use eigenvalue approach
        vals, vecs = np.linalg.eig(self.A.T)
        idx = np.argmin(np.abs(vals - 1.0))
        ss = np.real(vecs[:, idx])
        ss = np.abs(ss) / np.abs(ss).sum()
        return ss


def _gaussian_pdf(x: float, mu: float, sigma: float) -> float:
    z = (x - mu) / max(sigma, 1e-9)
    return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2 * math.pi))


# ---------------------------------------------------------------------------
# 2. Volatility Regime Detector
# ---------------------------------------------------------------------------

def volatility_regime(returns: np.ndarray,
                      window: int = 21,
                      percentiles: tuple = (25, 75, 95)) -> dict:
    """Classify each period into Low/Med/High/Crisis volatility regimes.

    Uses rolling realised volatility and empirical percentile thresholds.
    """
    T = len(returns)
    if T < window + 2:
        return {"error": "Insufficient data"}

    # Rolling vol (annualised)
    rolling_vol = np.full(T, np.nan)
    for t in range(window, T):
        w = returns[t - window:t]
        rolling_vol[t] = float(np.std(w, ddof=1) * math.sqrt(252))

    valid = rolling_vol[np.isfinite(rolling_vol)]
    if len(valid) < 5:
        return {"error": "Not enough vol data"}

    # Percentile thresholds
    p25, p75, p95 = np.percentile(valid, list(percentiles))

    labels = []
    for v in rolling_vol:
        if not np.isfinite(v):
            labels.append("unknown")
        elif v <= p25:
            labels.append(VolRegime.LOW)
        elif v <= p75:
            labels.append(VolRegime.MED)
        elif v <= p95:
            labels.append(VolRegime.HIGH)
        else:
            labels.append(VolRegime.CRISIS)

    current = labels[-1]
    regime_returns = {}
    for regime in [VolRegime.LOW, VolRegime.MED, VolRegime.HIGH, VolRegime.CRISIS]:
        mask = np.array([l == regime for l in labels])
        r = returns[mask]
        if len(r) > 2:
            regime_returns[regime] = {
                "count": int(mask.sum()),
                "pct_of_time": round(float(mask.mean()), 4),
                "ann_return": round(float(r.mean() * 252), 4),
                "ann_vol": round(float(r.std(ddof=1) * math.sqrt(252)), 4),
                "sharpe": round(float(r.mean() / max(r.std(ddof=1), 1e-9) * math.sqrt(252)), 4),
            }

    # Transition matrix
    trans = _compute_transition_matrix(labels, [VolRegime.LOW, VolRegime.MED,
                                                  VolRegime.HIGH, VolRegime.CRISIS])

    return {
        "regime_labels": [str(l) for l in labels],
        "current_regime": str(current),
        "vol_thresholds": {"low_max": round(p25, 4), "med_max": round(p75, 4),
                           "high_max": round(p95, 4)},
        "regime_stats": regime_returns,
        "transition_matrix": trans,
    }


def _compute_transition_matrix(labels: list, states: list) -> dict:
    """Compute empirical state transition matrix."""
    n = len(states)
    counts = {s: {s2: 0 for s2 in states} for s in states}
    for i in range(len(labels) - 1):
        a, b = labels[i], labels[i + 1]
        if a in counts and b in counts[a]:
            counts[a][b] += 1
    # Normalise rows
    matrix = {}
    for s in states:
        row_total = sum(counts[s].values())
        if row_total > 0:
            matrix[str(s)] = {str(s2): round(counts[s][s2] / row_total, 4) for s2 in states}
    return matrix


# ---------------------------------------------------------------------------
# 3. Trend Regime Detector
# ---------------------------------------------------------------------------

def trend_regime(prices: np.ndarray,
                 fast: int = 50,
                 slow: int = 200,
                 slope_window: int = 20) -> dict:
    """Classify Bull / Bear / Sideways based on SMA crossovers + slope.

    Bull: fast > slow AND slope > 0
    Bear: fast < slow AND slope < 0
    Sideways: otherwise
    """
    T = len(prices)
    if T < slow + slope_window:
        return {"error": "Insufficient data for trend detection"}

    # SMAs
    sma_fast = np.full(T, np.nan)
    sma_slow = np.full(T, np.nan)
    for t in range(T):
        if t >= fast:
            sma_fast[t] = float(np.mean(prices[t - fast:t]))
        if t >= slow:
            sma_slow[t] = float(np.mean(prices[t - slow:t]))

    # Slope of fast SMA (normalised)
    slope = np.full(T, np.nan)
    for t in range(slope_window, T):
        if np.isfinite(sma_fast[t]) and np.isfinite(sma_fast[t - slope_window]):
            slope[t] = (sma_fast[t] - sma_fast[t - slope_window]) / max(
                abs(sma_fast[t - slope_window]), 1e-9)

    labels = []
    for t in range(T):
        if not (np.isfinite(sma_fast[t]) and np.isfinite(sma_slow[t]) and np.isfinite(slope[t])):
            labels.append("unknown")
        elif sma_fast[t] > sma_slow[t] and slope[t] > 0:
            labels.append(TrendRegime.BULL)
        elif sma_fast[t] < sma_slow[t] and slope[t] < 0:
            labels.append(TrendRegime.BEAR)
        else:
            labels.append(TrendRegime.SIDEWAYS)

    current = labels[-1]
    returns = np.diff(prices) / np.maximum(prices[:-1], 1e-9)
    regime_stats = {}
    for regime in [TrendRegime.BULL, TrendRegime.BEAR, TrendRegime.SIDEWAYS]:
        mask = np.array([l == regime for l in labels[1:]])
        r = returns[mask]
        if len(r) > 2:
            regime_stats[str(regime)] = {
                "count": int(mask.sum()),
                "pct_of_time": round(float(mask.mean()), 4),
                "ann_return": round(float(r.mean() * 252), 4),
                "ann_vol": round(float(r.std(ddof=1) * math.sqrt(252)), 4),
                "sharpe": round(float(r.mean() / max(r.std(ddof=1), 1e-9) * math.sqrt(252)), 4),
            }

    # Current SMA values
    current_fast = float(sma_fast[-1]) if np.isfinite(sma_fast[-1]) else None
    current_slow = float(sma_slow[-1]) if np.isfinite(sma_slow[-1]) else None

    return {
        "regime_labels": [str(l) for l in labels],
        "current_regime": str(current),
        "current_sma_fast": round(current_fast, 4) if current_fast else None,
        "current_sma_slow": round(current_slow, 4) if current_slow else None,
        "regime_stats": regime_stats,
    }


# ---------------------------------------------------------------------------
# 4. HMM-based regime (main entry point)
# ---------------------------------------------------------------------------

def hmm_regime(returns: np.ndarray,
               n_iter: int = 100) -> dict:
    """Fit a 2-state Gaussian HMM and decode regimes.

    Automatically labels states as Bull/Bear based on mean returns.
    Returns full regime history + current state + transition probabilities.
    """
    T = len(returns)
    if T < 60:
        return {"error": "Need ≥ 60 observations for HMM"}

    model = GaussianHMM2State(n_iter=n_iter)
    model.fit(returns)
    states, probs = model.predict(returns)

    # Label states: lower mean → bear, higher mean → bull
    bull_state = int(np.argmax(model.means))
    bear_state = 1 - bull_state

    labels = [TrendRegime.BULL if s == bull_state else TrendRegime.BEAR
              for s in states]
    current = str(labels[-1])

    # Per-regime statistics
    regime_stats = {}
    for regime_label, state_idx in [(TrendRegime.BULL, bull_state),
                                     (TrendRegime.BEAR, bear_state)]:
        mask = states == state_idx
        r = returns[mask]
        if len(r) > 2:
            regime_stats[str(regime_label)] = {
                "count": int(mask.sum()),
                "pct_of_time": round(float(mask.mean()), 4),
                "ann_return": round(float(r.mean() * 252), 4),
                "ann_vol": round(float(r.std(ddof=1) * math.sqrt(252)), 4),
                "sharpe": round(float(r.mean() / max(r.std(ddof=1), 1e-9) * math.sqrt(252)), 4),
                "hmm_mean": round(float(model.means[state_idx]), 6),
                "hmm_std": round(float(model.stds[state_idx]), 6),
            }

    # Steady-state probabilities
    try:
        ss = model.steady_state()
        steady_state = {
            str(TrendRegime.BULL): round(float(ss[bull_state]), 4),
            str(TrendRegime.BEAR): round(float(ss[bear_state]), 4),
        }
    except Exception:
        steady_state = None

    # Regime switches
    switches = int(sum(states[i] != states[i - 1] for i in range(1, T)))

    return {
        "method": "gaussian_hmm_2state",
        "regime_labels": [str(l) for l in labels],
        "regime_probs": [[round(float(p), 4) for p in probs[t]] for t in range(T)],
        "current_regime": current,
        "current_bull_prob": round(float(probs[-1][bull_state]), 4),
        "regime_stats": regime_stats,
        "transition_matrix": {
            str(TrendRegime.BULL): {
                str(TrendRegime.BULL): round(float(model.A[bull_state, bull_state]), 4),
                str(TrendRegime.BEAR): round(float(model.A[bull_state, bear_state]), 4),
            },
            str(TrendRegime.BEAR): {
                str(TrendRegime.BULL): round(float(model.A[bear_state, bull_state]), 4),
                str(TrendRegime.BEAR): round(float(model.A[bear_state, bear_state]), 4),
            },
        },
        "steady_state": steady_state,
        "n_regime_switches": switches,
        "hmm_params": {
            "means": [round(float(m), 6) for m in model.means],
            "stds": [round(float(s), 6) for s in model.stds],
        },
    }


# ---------------------------------------------------------------------------
# 5. Full regime dashboard
# ---------------------------------------------------------------------------

def run_regime_detection(returns: np.ndarray,
                         prices: np.ndarray | None = None,
                         hmm_iters: int = 100) -> dict:
    """Run all regime detectors and produce a unified dashboard.

    Parameters
    ----------
    returns : (T,) daily return series
    prices  : (T,) price series (optional, for trend regime)
    hmm_iters : HMM training iterations

    Returns
    -------
    Unified regime report with HMM, volatility, and trend regimes
    """
    T = len(returns)

    # HMM regime
    hmm = hmm_regime(returns, n_iter=hmm_iters)

    # Volatility regime
    vol = volatility_regime(returns, window=21)

    # Trend regime
    if prices is not None and len(prices) >= 60:
        trend = trend_regime(prices, fast=50, slow=200)
    else:
        trend = {"error": "Price data not provided"}

    # Regime alignment score: how consistent are the different detectors?
    alignment = _compute_regime_alignment(hmm, vol)

    return {
        "hmm_regime": hmm,
        "volatility_regime": vol,
        "trend_regime": trend,
        "regime_alignment": alignment,
        "current_summary": {
            "hmm": hmm.get("current_regime", "unknown"),
            "vol": vol.get("current_regime", "unknown"),
            "trend": trend.get("current_regime", "unknown"),
            "bull_prob_pct": round(hmm.get("current_bull_prob", 0.5) * 100, 1),
        },
        "n_periods": T,
    }


def _compute_regime_alignment(hmm: dict, vol: dict) -> dict:
    """Measure agreement between HMM and vol regime detectors."""
    hmm_labels = hmm.get("regime_labels", [])
    vol_labels = vol.get("regime_labels", [])
    n = min(len(hmm_labels), len(vol_labels))
    if n < 2:
        return {"agreement": 0.0}

    # Bull HMM + low/med vol = aligned bull
    # Bear HMM + high/crisis vol = aligned bear
    aligned = 0
    for i in range(n):
        h = hmm_labels[i]
        v = vol_labels[i]
        if h == str(TrendRegime.BULL) and v in (str(VolRegime.LOW), str(VolRegime.MED)):
            aligned += 1
        elif h == str(TrendRegime.BEAR) and v in (str(VolRegime.HIGH), str(VolRegime.CRISIS)):
            aligned += 1

    agreement = aligned / n
    return {
        "agreement_pct": round(agreement * 100, 1),
        "interpretation": (
            "High cross-method agreement — regime signal is reliable" if agreement > 0.70
            else "Moderate agreement — regime uncertainty, use wider thresholds"
            if agreement > 0.50
            else "Low agreement — conflicting signals, no clear regime"
        ),
    }
