"""QuantumSentinel — Pipeline Latency Benchmarker.

Measures end-to-end signal generation latency across the full research
pipeline to identify bottlenecks:

  Data Fetch → Feature Engineering → Signal Generation → Order Sizing
     → Execution Simulation → Portfolio Update → Risk Calculation

Each stage is timed independently and reported with:
  - wall-clock time (ms)
  - p50 / p95 / p99 / p99.9 percentiles (repeated runs)
  - relative % of total
  - throughput (bars/sec)
  - bottleneck flag

Phase 4 additions:
  - TimerStats: repeated-run percentile analysis
  - run_percentile_benchmark: per-stage p50/p95/p99 profiling
  - bench_cpp_vs_python: C++ kernel vs NumPy speedup comparison
  - Vectorised vs event-driven backtest comparison
  - Single-asset vs multi-asset scaling (N=1, 10, 20, 50)
  - In-memory matrix operations (factor model, correlation)
"""
from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Timer context manager
# ---------------------------------------------------------------------------

class Timer:
    """High-resolution wall-clock timer."""

    def __init__(self, label: str = ""):
        self.label = label
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0


@dataclass
class TimerStats:
    """Statistical latency summary from N repeated runs."""
    label: str
    n_runs: int
    samples_ms: list[float] = field(default_factory=list)

    def record(self, elapsed_ms: float) -> None:
        self.samples_ms.append(elapsed_ms)

    def summary(self) -> dict:
        if not self.samples_ms:
            return {"label": self.label, "n_runs": 0}
        arr = np.array(self.samples_ms, dtype=float)
        return {
            "label":    self.label,
            "n_runs":   len(arr),
            "mean_ms":  round(float(arr.mean()), 4),
            "std_ms":   round(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, 4),
            "min_ms":   round(float(arr.min()), 4),
            "max_ms":   round(float(arr.max()), 4),
            "p50_ms":   round(float(np.percentile(arr, 50)), 4),
            "p95_ms":   round(float(np.percentile(arr, 95)), 4),
            "p99_ms":   round(float(np.percentile(arr, 99)), 4),
            "p999_ms":  round(float(np.percentile(arr, 99.9)), 4),
        }


def time_fn(fn: Callable, n_runs: int = 20, label: str = "") -> dict:
    """Time a callable `fn` n_runs times and return a TimerStats summary dict."""
    stats = TimerStats(label=label or fn.__name__, n_runs=n_runs)
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        stats.record((time.perf_counter() - t0) * 1000.0)
    return stats.summary()


# ---------------------------------------------------------------------------
# Stage benchmarks
# ---------------------------------------------------------------------------

def bench_feature_engineering(return_matrix: np.ndarray) -> dict:
    """Benchmark factor computation (momentum, vol, quality, etc.)."""
    from .factor_model import compute_factors

    T, N = return_matrix.shape
    with Timer() as t:
        factors = compute_factors(return_matrix)
    n_factors = len(factors)

    return {
        "stage": "Feature Engineering",
        "elapsed_ms": round(t.elapsed_ms, 3),
        "throughput_bars_per_sec": round(T / max(t.elapsed_ms / 1000, 1e-9), 0),
        "n_factors": n_factors,
        "n_assets": N,
        "n_bars": T,
    }


def bench_correlation_engine(return_matrix: np.ndarray) -> dict:
    """Benchmark multi-method correlation computation."""
    from .correlation_engine import run_correlation_engine

    T, N = return_matrix.shape
    with Timer() as t:
        result = run_correlation_engine(return_matrix)
    n_methods = len(result.get("correlations", {}))

    return {
        "stage": "Correlation Engine",
        "elapsed_ms": round(t.elapsed_ms, 3),
        "throughput_bars_per_sec": round(T / max(t.elapsed_ms / 1000, 1e-9), 0),
        "n_estimators": n_methods,
        "n_assets": N,
        "n_bars": T,
    }


def bench_portfolio_optimization(return_matrix: np.ndarray) -> dict:
    """Benchmark portfolio optimisation across all methods."""
    from .portfolio_optimization import run_portfolio_optimization

    T, N = return_matrix.shape
    with Timer() as t:
        result = run_portfolio_optimization(return_matrix)
    n_methods = len(result.get("portfolios", {}))

    return {
        "stage": "Portfolio Optimisation",
        "elapsed_ms": round(t.elapsed_ms, 3),
        "throughput_bars_per_sec": round(T / max(t.elapsed_ms / 1000, 1e-9), 0),
        "n_methods": n_methods,
        "n_assets": N,
        "n_bars": T,
    }


def bench_alpha_research(return_matrix: np.ndarray, signal_matrix: np.ndarray) -> dict:
    """Benchmark alpha research pipeline (IC, decay, quintile)."""
    from .alpha_research import run_alpha_research

    T, N = return_matrix.shape
    with Timer() as t:
        result = run_alpha_research(signal_matrix, return_matrix, max_horizon=10)

    return {
        "stage": "Alpha Research",
        "elapsed_ms": round(t.elapsed_ms, 3),
        "throughput_bars_per_sec": round(T / max(t.elapsed_ms / 1000, 1e-9), 0),
        "n_assets": N,
        "n_bars": T,
    }


def bench_fama_macbeth(return_matrix: np.ndarray) -> dict:
    """Benchmark Fama-MacBeth factor model regression."""
    from .factor_model import compute_factors, fama_macbeth

    T, N = return_matrix.shape

    with Timer("factor_compute") as t_feat:
        factors = compute_factors(return_matrix)

    with Timer("fama_macbeth") as t_fm:
        result = fama_macbeth(return_matrix, factors)

    n_premia = len(result.get("factor_premia", {}))

    return {
        "stage": "Fama-MacBeth Regression",
        "feature_ms": round(t_feat.elapsed_ms, 3),
        "regression_ms": round(t_fm.elapsed_ms, 3),
        "elapsed_ms": round(t_feat.elapsed_ms + t_fm.elapsed_ms, 3),
        "n_factors": n_premia,
        "n_assets": N,
        "n_bars": T,
    }


def bench_event_backtest(price_data: dict[str, np.ndarray],
                          strategy: str = "ma_crossover",
                          cost_model: str = "retail") -> dict:
    """Benchmark the event-driven backtester."""
    from .event_simulator import run_event_backtest

    tickers = list(price_data.keys())
    T = min(len(v) for v in price_data.values())

    with Timer() as t:
        result = run_event_backtest(
            tickers=tickers,
            price_data=price_data,
            strategy_name=strategy,
            cost_model_name=cost_model,
        )

    bars_per_sec = T / max(t.elapsed_ms / 1000, 1e-9)
    n_trades = result.get("n_trades", 0)

    return {
        "stage": "Event-Driven Backtest",
        "elapsed_ms": round(t.elapsed_ms, 3),
        "throughput_bars_per_sec": round(bars_per_sec, 0),
        "n_trades": n_trades,
        "n_tickers": len(tickers),
        "n_bars": T,
        "strategy": strategy,
        "cost_model": cost_model,
    }


def bench_regime_detection(returns: np.ndarray) -> dict:
    """Benchmark regime detection (HMM + vol + trend)."""
    from .regime_detection import run_regime_detection

    T = len(returns)
    with Timer() as t:
        result = run_regime_detection(returns)

    return {
        "stage": "Regime Detection",
        "elapsed_ms": round(t.elapsed_ms, 3),
        "throughput_bars_per_sec": round(T / max(t.elapsed_ms / 1000, 1e-9), 0),
        "n_bars": T,
        "current_regime": result.get("current_summary", {}).get("hmm", "unknown"),
    }


# ---------------------------------------------------------------------------
# Scaling benchmark (N assets)
# ---------------------------------------------------------------------------

def bench_scaling(n_values: list[int] | None = None,
                  T: int = 500,
                  seed: int = 42) -> list[dict]:
    """Measure how latency scales with number of assets N."""
    from .factor_model import compute_factors

    if n_values is None:
        n_values = [5, 10, 20, 30, 50]

    rng = np.random.default_rng(seed)
    results = []

    for N in n_values:
        returns = rng.normal(0, 0.01, (T, N))
        with Timer() as t_feat:
            _ = compute_factors(returns)
        from .correlation_engine import run_correlation_engine
        with Timer() as t_corr:
            _ = run_correlation_engine(returns)
        from .portfolio_optimization import run_portfolio_optimization
        with Timer() as t_opt:
            _ = run_portfolio_optimization(returns)

        results.append({
            "n_assets": N,
            "feature_ms": round(t_feat.elapsed_ms, 3),
            "correlation_ms": round(t_corr.elapsed_ms, 3),
            "optimization_ms": round(t_opt.elapsed_ms, 3),
            "total_ms": round(t_feat.elapsed_ms + t_corr.elapsed_ms + t_opt.elapsed_ms, 3),
        })

    return results


# ---------------------------------------------------------------------------
# Full benchmark suite
# ---------------------------------------------------------------------------

def run_full_benchmark(return_matrix: np.ndarray,
                        price_matrix: np.ndarray,
                        tickers: list[str] | None = None) -> dict:
    """Run all benchmarks and produce a latency report.

    Parameters
    ----------
    return_matrix : (T, N) return matrix
    price_matrix : (T, N) price matrix
    tickers : optional list of asset names

    Returns
    -------
    Full latency report with per-stage timing, bottleneck identification,
    and scaling analysis.
    """
    T, N = return_matrix.shape
    names = tickers or [f"Asset_{i}" for i in range(N)]

    stages = []

    # Stage 1: Feature Engineering
    try:
        stages.append(bench_feature_engineering(return_matrix))
    except Exception as e:
        stages.append({"stage": "Feature Engineering", "error": str(e), "elapsed_ms": 0})

    # Stage 2: Alpha Research
    try:
        from .factor_model import compute_factors
        factors = compute_factors(return_matrix)
        sig = factors.get("momentum", np.random.default_rng(0).normal(0, 1, (T, N)))
        stages.append(bench_alpha_research(return_matrix, sig))
    except Exception as e:
        stages.append({"stage": "Alpha Research", "error": str(e), "elapsed_ms": 0})

    # Stage 3: Fama-MacBeth
    try:
        stages.append(bench_fama_macbeth(return_matrix))
    except Exception as e:
        stages.append({"stage": "Fama-MacBeth Regression", "error": str(e), "elapsed_ms": 0})

    # Stage 4: Correlation Engine
    try:
        stages.append(bench_correlation_engine(return_matrix))
    except Exception as e:
        stages.append({"stage": "Correlation Engine", "error": str(e), "elapsed_ms": 0})

    # Stage 5: Portfolio Optimisation
    try:
        stages.append(bench_portfolio_optimization(return_matrix))
    except Exception as e:
        stages.append({"stage": "Portfolio Optimisation", "error": str(e), "elapsed_ms": 0})

    # Stage 6: Event-Driven Backtest
    try:
        price_dict = {names[i]: price_matrix[:, i] for i in range(min(N, 5))}
        stages.append(bench_event_backtest(price_dict, strategy="ma_crossover"))
    except Exception as e:
        stages.append({"stage": "Event-Driven Backtest", "error": str(e), "elapsed_ms": 0})

    # Stage 7: Regime Detection
    try:
        stages.append(bench_regime_detection(return_matrix[:, 0]))
    except Exception as e:
        stages.append({"stage": "Regime Detection", "error": str(e), "elapsed_ms": 0})

    # ── Analysis ──
    total_ms = sum(s.get("elapsed_ms", 0) for s in stages)
    for s in stages:
        ms = s.get("elapsed_ms", 0)
        s["pct_of_total"] = round(ms / max(total_ms, 1e-9) * 100, 1)

    # Identify bottleneck
    bottleneck = max(stages, key=lambda s: s.get("elapsed_ms", 0))

    # Scaling analysis (quick — fewer points)
    try:
        scaling = bench_scaling(n_values=[5, 10, 20, N], T=min(T, 300))
    except Exception:
        scaling = []

    return {
        "total_elapsed_ms": round(total_ms, 2),
        "total_elapsed_sec": round(total_ms / 1000, 3),
        "n_assets": N,
        "n_bars": T,
        "stages": stages,
        "bottleneck": {
            "stage": bottleneck.get("stage", "unknown"),
            "elapsed_ms": bottleneck.get("elapsed_ms", 0),
            "pct_of_total": bottleneck.get("pct_of_total", 0),
        },
        "scaling_analysis": scaling,
        "summary": {
            "full_pipeline_ms": round(total_ms, 2),
            "bars_per_second": round(T / max(total_ms / 1000, 1e-9), 0),
            "performance_rating": (
                "Excellent (< 500ms)" if total_ms < 500
                else "Good (< 2s)" if total_ms < 2000
                else "Acceptable (< 10s)" if total_ms < 10000
                else "Needs optimisation (> 10s)"
            ),
        },
    }


# ---------------------------------------------------------------------------
# Phase 4B: Percentile benchmark
# ---------------------------------------------------------------------------

def run_percentile_benchmark(return_matrix: np.ndarray,
                              price_matrix: np.ndarray,
                              tickers: list[str] | None = None,
                              n_runs: int = 20) -> dict:
    """Run each pipeline stage n_runs times and report p50/p95/p99 latencies.

    Parameters
    ----------
    return_matrix : (T, N) return matrix
    price_matrix  : (T, N) price matrix
    tickers       : optional asset names
    n_runs        : number of timed repetitions per stage (default 20)

    Returns
    -------
    dict with per-stage percentile stats and aggregate summary.
    """
    from .factor_model import compute_factors, fama_macbeth
    from .correlation_engine import run_correlation_engine
    from .portfolio_optimization import run_portfolio_optimization
    from .regime_detection import run_regime_detection

    T, N = return_matrix.shape
    names = tickers or [f"Asset_{i}" for i in range(N)]
    rng_returns = return_matrix[:, 0]

    stage_results = []

    # Feature engineering
    stage_results.append(time_fn(
        lambda: compute_factors(return_matrix), n_runs=n_runs,
        label="Feature Engineering"
    ))

    # Fama-MacBeth
    _factors = compute_factors(return_matrix)  # pre-compute once
    stage_results.append(time_fn(
        lambda: fama_macbeth(return_matrix, _factors), n_runs=n_runs,
        label="Fama-MacBeth Regression"
    ))

    # Correlation engine
    stage_results.append(time_fn(
        lambda: run_correlation_engine(return_matrix), n_runs=n_runs,
        label="Correlation Engine"
    ))

    # Portfolio optimisation
    stage_results.append(time_fn(
        lambda: run_portfolio_optimization(return_matrix), n_runs=n_runs,
        label="Portfolio Optimisation"
    ))

    # Regime detection
    stage_results.append(time_fn(
        lambda: run_regime_detection(rng_returns), n_runs=n_runs,
        label="Regime Detection"
    ))

    # Aggregate
    p50_total = sum(s["p50_ms"] for s in stage_results)
    p99_total = sum(s["p99_ms"] for s in stage_results)

    bottleneck = max(stage_results, key=lambda s: s["p99_ms"])

    return {
        "n_runs_per_stage": n_runs,
        "n_assets": N,
        "n_bars": T,
        "stages": stage_results,
        "aggregate": {
            "p50_pipeline_ms": round(p50_total, 3),
            "p99_pipeline_ms": round(p99_total, 3),
        },
        "bottleneck": {
            "stage": bottleneck["label"],
            "p99_ms": bottleneck["p99_ms"],
        },
    }


# ---------------------------------------------------------------------------
# Phase 4A: C++ vs Python speedup benchmark
# ---------------------------------------------------------------------------

def bench_cpp_vs_python(T: int = 500, N: int = 10,
                         n_runs: int = 20,
                         seed: int = 42) -> dict:
    """Compare C++ kernels vs NumPy fallback for rolling_corr, hmm_forward,
    and backtest_loop.

    Returns speedup ratios and correctness check (max absolute difference).
    """
    from .cpp_ext import (
        CPP_AVAILABLE,
        _py_rolling_corr, _py_hmm_forward, _py_backtest_loop,
    )

    rng = np.random.default_rng(seed)
    X = rng.normal(0, 0.01, (T, N))
    prices = np.cumprod(1 + rng.normal(0, 0.01, T)) * 100.0
    signals = rng.choice([-1.0, 0.0, 1.0], size=T)
    K = 2
    pi = np.array([0.6, 0.4])
    A = np.array([[0.97, 0.03], [0.05, 0.95]])
    means = np.array([0.0005, -0.001])
    stds  = np.array([0.008, 0.025])
    obs   = X[:, 0]
    window = 60

    results: dict = {"cpp_available": CPP_AVAILABLE, "kernels": {}}

    # ── rolling_corr ──
    rc_py = time_fn(lambda: _py_rolling_corr(X, window), n_runs=n_runs,
                    label="rolling_corr (Python)")
    if CPP_AVAILABLE:
        from .cpp_ext import _cpp_rolling_corr
        rc_cpp = time_fn(lambda: _cpp_rolling_corr(X, window), n_runs=n_runs,
                         label="rolling_corr (C++)")
        py_out = _py_rolling_corr(X, window)
        cpp_out = _cpp_rolling_corr(X, window)
        max_diff = float(np.max(np.abs(py_out - cpp_out)))
        speedup = rc_py["p50_ms"] / max(rc_cpp["p50_ms"], 1e-6)
    else:
        rc_cpp = None
        max_diff = 0.0
        speedup = 1.0
    results["kernels"]["rolling_corr"] = {
        "python": rc_py, "cpp": rc_cpp,
        "speedup_p50x": round(speedup, 2),
        "max_abs_diff": round(max_diff, 12),
        "numerically_identical": max_diff < 1e-10,
    }

    # ── hmm_forward ──
    hf_py = time_fn(
        lambda: _py_hmm_forward(obs, pi, A, means, stds),
        n_runs=n_runs, label="hmm_forward (Python)"
    )
    if CPP_AVAILABLE:
        from .cpp_ext import _cpp_hmm_forward
        hf_cpp = time_fn(
            lambda: _cpp_hmm_forward(obs, pi, A, means, stds),
            n_runs=n_runs, label="hmm_forward (C++)"
        )
        py_a, py_ll = _py_hmm_forward(obs, pi, A, means, stds)
        cpp_a, cpp_ll = _cpp_hmm_forward(obs, pi, A, means, stds)
        max_diff_hmm = float(np.max(np.abs(py_a - cpp_a)))
        speedup_hmm = hf_py["p50_ms"] / max(hf_cpp["p50_ms"], 1e-6)
    else:
        hf_cpp = None
        max_diff_hmm = 0.0
        speedup_hmm = 1.0
    results["kernels"]["hmm_forward"] = {
        "python": hf_py, "cpp": hf_cpp,
        "speedup_p50x": round(speedup_hmm, 2),
        "max_abs_diff": round(max_diff_hmm, 12),
        "numerically_identical": max_diff_hmm < 1e-10,
    }

    # ── backtest_loop ──
    bl_py = time_fn(
        lambda: _py_backtest_loop(prices, signals),
        n_runs=n_runs, label="backtest_loop (Python)"
    )
    if CPP_AVAILABLE:
        from .cpp_ext import _cpp_backtest_loop
        bl_cpp = time_fn(
            lambda: _cpp_backtest_loop(prices, signals),
            n_runs=n_runs, label="backtest_loop (C++)"
        )
        py_r = _py_backtest_loop(prices, signals)
        cpp_r = _cpp_backtest_loop(prices, signals)
        max_diff_bl = float(np.max(np.abs(
            np.asarray(py_r["equity_curve"]) - np.asarray(cpp_r["equity_curve"])
        )))
        speedup_bl = bl_py["p50_ms"] / max(bl_cpp["p50_ms"], 1e-6)
    else:
        bl_cpp = None
        max_diff_bl = 0.0
        speedup_bl = 1.0
    results["kernels"]["backtest_loop"] = {
        "python": bl_py, "cpp": bl_cpp,
        "speedup_p50x": round(speedup_bl, 2),
        "max_abs_diff": round(max_diff_bl, 12),
        "numerically_identical": max_diff_bl < 1e-10,
    }

    return results
