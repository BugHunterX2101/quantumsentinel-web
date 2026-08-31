"""QuantumSentinel — C++ extension wrapper with graceful NumPy fallback.

Tries to import the compiled _qs_fast pybind11 extension; if unavailable
(no toolchain, first-run, CI environment) falls back to pure-NumPy
implementations that are numerically identical.

Public API (same regardless of which path is active)
-----------------------------------------------------
rolling_corr(X, window)            -> ndarray (T, N, N)
hmm_forward(obs, pi, A, means, stds) -> (alpha ndarray, log_likelihood float)
backtest_loop(prices, signals, commission, spread_bps) -> dict

CPP_AVAILABLE : bool  — True iff the compiled .pyd/.so was loaded
"""
from __future__ import annotations

import math
import logging
import sys
import os

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attempt native import
# ---------------------------------------------------------------------------

# Search order:
#   1. backend/services/ itself  (pyd copied here after build)
#   2. cpp/                     (in-place build output)
#   3. cpp/build/lib.*/         (setuptools out-of-place build)
_HERE = os.path.dirname(os.path.abspath(__file__))
_CPP_BUILD = os.path.normpath(os.path.join(_HERE, "..", "..", "cpp"))
_CPP_GLOB_BUILDS = []
try:
    import glob as _glob
    _CPP_GLOB_BUILDS = _glob.glob(os.path.join(_CPP_BUILD, "build", "lib.*"))
except Exception:
    pass

for _search_path in [_HERE, _CPP_BUILD] + _CPP_GLOB_BUILDS:
    if _search_path not in sys.path:
        sys.path.insert(0, _search_path)

# On Windows, DLLs (libwinpthread-1.dll, python313.dll) that live alongside
# the .pyd must be declared in the DLL search path explicitly.
if hasattr(os, "add_dll_directory"):
    for _dll_dir in [_HERE, _CPP_BUILD]:
        try:
            os.add_dll_directory(_dll_dir)
        except (OSError, ValueError):
            pass

try:
    from _qs_fast import (        # type: ignore[import]
        rolling_corr as _cpp_rolling_corr,
        hmm_forward  as _cpp_hmm_forward,
        backtest_loop as _cpp_backtest_loop,
    )
    CPP_AVAILABLE = True
    log.info("_qs_fast C++ extension loaded — hardware-accelerated kernels active")
except ImportError:
    CPP_AVAILABLE = False
    log.info(
        "_qs_fast C++ extension not found — using NumPy fallback implementations. "
        "To build: cd cpp && pip install -e . (requires g++ or MSVC)"
    )


# ---------------------------------------------------------------------------
# Pure-NumPy fallback implementations
# (numerically equivalent to the C++ kernels, tested jointly)
# ---------------------------------------------------------------------------

def _py_rolling_corr(X: np.ndarray, window: int) -> np.ndarray:
    """Rolling Pearson correlation matrix — pure NumPy fallback."""
    T, N = X.shape
    result = np.zeros((T, N, N))
    # Identity for bars without enough history
    for t in range(T):
        result[t] = np.eye(N)

    for t in range(window - 1, T):
        w = X[t - window + 1 : t + 1]          # (window, N)
        w_centered = w - w.mean(axis=0)
        cov = (w_centered.T @ w_centered) / max(window - 1, 1)
        stds = np.sqrt(np.maximum(np.diag(cov), 1e-12))
        denom = np.outer(stds, stds)
        corr = cov / np.where(denom > 1e-14, denom, 1.0)
        np.fill_diagonal(corr, 1.0)
        result[t] = np.clip(corr, -1.0, 1.0)

    return result


def _py_hmm_forward(obs: np.ndarray, pi: np.ndarray, A: np.ndarray,
                    means: np.ndarray, stds: np.ndarray
                    ) -> tuple[np.ndarray, float]:
    """Scaled HMM forward algorithm — pure NumPy fallback."""
    T = len(obs)
    K = len(pi)

    def emit(t: int) -> np.ndarray:
        z = (obs[t] - means) / np.maximum(stds, 1e-9)
        return np.exp(-0.5 * z * z) / (np.maximum(stds, 1e-9) * math.sqrt(2 * math.pi))

    alpha = np.zeros((T, K))
    scales = np.zeros(T)

    alpha[0] = pi * emit(0)
    scales[0] = max(alpha[0].sum(), 1e-300)
    alpha[0] /= scales[0]

    for t in range(1, T):
        alpha[t] = (alpha[t - 1] @ A) * emit(t)
        scales[t] = max(alpha[t].sum(), 1e-300)
        alpha[t] /= scales[t]

    log_likelihood = float(np.sum(np.log(np.maximum(scales, 1e-300))))
    return alpha, log_likelihood


def _py_backtest_loop(prices: np.ndarray, signals: np.ndarray,
                      commission: float = 0.001,
                      spread_bps: float = 5.0) -> dict:
    """Bar-by-bar backtest with 1-bar execution delay — pure NumPy fallback."""
    T = len(prices)
    equity = np.ones(T)
    returns = np.zeros(T - 1) if T > 1 else np.zeros(1)
    half_spread = spread_bps / 1e4

    position = 0.0
    total_cost = 0.0
    n_trades = 0

    for t in range(1, T):
        target = float(signals[t - 1])
        delta = target - position
        price_ret = (prices[t] - prices[t - 1]) / max(abs(prices[t - 1]), 1e-9)

        cost = 0.0
        if abs(delta) > 1e-6:
            cost = (commission + half_spread) * abs(delta)
            position = target
            n_trades += 1

        total_cost += cost
        port_ret = position * price_ret - cost
        equity[t] = equity[t - 1] * (1.0 + port_ret)
        if T > 1:
            returns[t - 1] = port_ret

    return {
        "equity_curve": equity,
        "daily_returns": returns,
        "n_trades": n_trades,
        "total_cost": total_cost,
    }


# ---------------------------------------------------------------------------
# Unified public API — dispatch to C++ or NumPy fallback
# ---------------------------------------------------------------------------

def rolling_corr(X: np.ndarray, window: int) -> np.ndarray:
    """Rolling Pearson correlation matrices (T, N, N).

    Parameters
    ----------
    X      : (T, N) float64 return matrix
    window : int lookback window (>= 2)

    Returns
    -------
    ndarray (T, N, N) — corr[t] is the N×N correlation at bar t.
    Bars 0..window-2 are identity matrices.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be 2-D (T, N)")
    if CPP_AVAILABLE:
        return _cpp_rolling_corr(X, window)
    return _py_rolling_corr(X, window)


def hmm_forward(obs: np.ndarray, pi: np.ndarray, A: np.ndarray,
                means: np.ndarray, stds: np.ndarray
                ) -> tuple[np.ndarray, float]:
    """Scaled HMM forward algorithm.

    Parameters
    ----------
    obs   : (T,)   observation sequence
    pi    : (K,)   initial state probabilities
    A     : (K, K) row-stochastic transition matrix
    means : (K,)   emission means
    stds  : (K,)   emission standard deviations

    Returns
    -------
    (alpha (T, K), log_likelihood float)
    """
    obs   = np.asarray(obs,   dtype=np.float64)
    pi    = np.asarray(pi,    dtype=np.float64)
    A     = np.asarray(A,     dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    stds  = np.asarray(stds,  dtype=np.float64)
    if CPP_AVAILABLE:
        return _cpp_hmm_forward(obs, pi, A, means, stds)
    return _py_hmm_forward(obs, pi, A, means, stds)


def backtest_loop(prices: np.ndarray, signals: np.ndarray,
                  commission: float = 0.001,
                  spread_bps: float = 5.0) -> dict:
    """Bar-by-bar backtest fill loop with 1-bar execution delay.

    Parameters
    ----------
    prices     : (T,) close price series
    signals    : (T,) position signal (-1, 0, +1) at each bar
    commission : float, fraction of notional per trade
    spread_bps : float, bid-ask half-spread in basis points

    Returns
    -------
    dict: equity_curve (T,), daily_returns (T-1,), n_trades, total_cost
    """
    prices  = np.asarray(prices,  dtype=np.float64)
    signals = np.asarray(signals, dtype=np.float64)
    if CPP_AVAILABLE:
        return _cpp_backtest_loop(prices, signals, commission, spread_bps)
    return _py_backtest_loop(prices, signals, commission, spread_bps)
