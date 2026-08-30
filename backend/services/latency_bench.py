"""QuantumSentinel — Pipeline Latency Benchmarker.

Measures end-to-end signal generation latency across the full research
pipeline to identify bottlenecks:

  Data Fetch → Feature Engineering → Signal Generation → Order Sizing
     → Execution Simulation → Portfolio Update → Risk Calculation

Each stage is timed independently and reported with:
  - wall-clock time (ms)
  - relative % of total
  - throughput (bars/sec)
  - bottleneck flag

Also benchmarks:
  - Vectorised vs event-driven backtest comparison
  - Single-asset vs multi-asset scaling (N=1, 10, 20, 50)
  - In-memory matrix operations (factor model, correlation)
"""
from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass

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
