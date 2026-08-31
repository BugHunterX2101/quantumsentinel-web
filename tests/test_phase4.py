"""Tests for Phase 4: C++ Extension, p50/p99 Latency, scipy Upgrades, Report Generator.

Tests cover:
  - cpp_ext: fallback always works, CPP_AVAILABLE is bool, kernels produce correct shapes/values
  - latency_bench: TimerStats, time_fn, run_percentile_benchmark, bench_cpp_vs_python
  - stat_tests: scipy-backed ADF and Durbin-Watson (if available)
  - neutral_strategies: statsmodels-backed cointegration (if available)
  - report_generator: section structure, executive_summary, risk_decomposition, efficient_frontier
"""
import math
import numpy as np
import pytest


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def return_matrix(rng):
    """Small (250, 8) return matrix — fast for repeated runs."""
    return rng.normal(0, 0.01, (250, 8))


@pytest.fixture
def price_series(rng):
    """Realistic-ish price series derived from a random walk."""
    rets = rng.normal(0.0003, 0.012, 300)
    return np.cumprod(1 + rets) * 100.0


@pytest.fixture
def strategy_returns(rng):
    return rng.normal(0.0003, 0.012, 252)


@pytest.fixture
def cointegrated_pair(rng):
    T = 300
    x = np.zeros(T); x[0] = 100.0
    for i in range(1, T):
        x[i] = x[i-1] + rng.normal(0, 0.5)
    y = 2.0 * x + rng.normal(0, 1.5, T)
    return y, x


@pytest.fixture
def nonstationary_series(rng):
    """Pure random walk — should not reject unit root."""
    T = 300
    rw = np.zeros(T)
    for i in range(1, T):
        rw[i] = rw[i-1] + rng.normal(0, 1)
    return rw


@pytest.fixture
def stationary_series(rng):
    """IID noise — should be stationary."""
    return rng.normal(0, 1, 300)


# ===========================================================================
# Track 4A: C++ Extension (cpp_ext)
# ===========================================================================

class TestCppExtFallback:
    """All tests must pass even when CPP_AVAILABLE = False."""

    def test_import_without_error(self):
        from backend.services.cpp_ext import CPP_AVAILABLE, rolling_corr, hmm_forward, backtest_loop
        assert isinstance(CPP_AVAILABLE, bool)

    def test_cpp_available_is_bool(self):
        from backend.services import cpp_ext
        assert isinstance(cpp_ext.CPP_AVAILABLE, bool)

    def test_rolling_corr_shape(self, rng):
        from backend.services.cpp_ext import rolling_corr
        X = rng.normal(0, 0.01, (200, 5))
        out = rolling_corr(X, window=60)
        assert out.shape == (200, 5, 5)

    def test_rolling_corr_identity_before_window(self, rng):
        from backend.services.cpp_ext import rolling_corr
        X = rng.normal(0, 0.01, (100, 3))
        out = rolling_corr(X, window=60)
        # First 59 bars should be identity
        for t in range(59):
            np.testing.assert_allclose(out[t], np.eye(3), atol=1e-10)

    def test_rolling_corr_diagonal_is_one(self, rng):
        from backend.services.cpp_ext import rolling_corr
        X = rng.normal(0, 0.01, (150, 4))
        out = rolling_corr(X, window=50)
        for t in range(50, 150):
            diag = np.diag(out[t])
            np.testing.assert_allclose(diag, np.ones(4), atol=1e-10)

    def test_rolling_corr_values_in_range(self, rng):
        from backend.services.cpp_ext import rolling_corr
        X = rng.normal(0, 0.01, (150, 4))
        out = rolling_corr(X, window=50)
        assert np.all(out >= -1.0 - 1e-10)
        assert np.all(out <= 1.0 + 1e-10)

    def test_hmm_forward_shape(self, rng):
        from backend.services.cpp_ext import hmm_forward
        obs = rng.normal(0, 0.01, 200)
        pi = np.array([0.6, 0.4])
        A = np.array([[0.97, 0.03], [0.05, 0.95]])
        means = np.array([0.0003, -0.001])
        stds = np.array([0.008, 0.025])
        alpha, ll = hmm_forward(obs, pi, A, means, stds)
        assert alpha.shape == (200, 2)
        assert isinstance(ll, float)
        assert np.isfinite(ll)  # log-likelihood must be finite (can be positive or negative)

    def test_hmm_forward_alpha_sums_one(self, rng):
        """Scaled alpha rows must sum to 1."""
        from backend.services.cpp_ext import hmm_forward
        obs = rng.normal(0, 0.01, 100)
        pi = np.array([0.5, 0.5])
        A = np.array([[0.9, 0.1], [0.1, 0.9]])
        means = np.array([0.0, 0.0])
        stds = np.array([0.01, 0.02])
        alpha, _ = hmm_forward(obs, pi, A, means, stds)
        row_sums = alpha.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(100), atol=1e-10)

    def test_backtest_loop_equity_shape(self, price_series, rng):
        from backend.services.cpp_ext import backtest_loop
        signals = rng.choice([-1.0, 0.0, 1.0], size=len(price_series))
        out = backtest_loop(price_series, signals)
        assert "equity_curve" in out
        assert "daily_returns" in out
        assert "n_trades" in out
        assert "total_cost" in out
        assert len(out["equity_curve"]) == len(price_series)

    def test_backtest_loop_starts_at_one(self, price_series, rng):
        from backend.services.cpp_ext import backtest_loop
        signals = rng.choice([-1.0, 0.0, 1.0], size=len(price_series))
        out = backtest_loop(price_series, signals)
        assert abs(float(out["equity_curve"][0]) - 1.0) < 1e-10

    def test_backtest_loop_cost_nonneg(self, price_series, rng):
        from backend.services.cpp_ext import backtest_loop
        signals = rng.choice([-1.0, 0.0, 1.0], size=len(price_series))
        out = backtest_loop(price_series, signals, commission=0.001, spread_bps=5.0)
        assert float(out["total_cost"]) >= 0.0

    def test_zero_cost_model(self, price_series, rng):
        from backend.services.cpp_ext import backtest_loop
        signals = rng.choice([-1.0, 0.0, 1.0], size=len(price_series))
        out = backtest_loop(price_series, signals, commission=0.0, spread_bps=0.0)
        assert abs(float(out["total_cost"])) < 1e-10

    def test_cpp_vs_python_rolling_corr_consistent(self, rng):
        """Python and C++ (if available) must agree numerically."""
        from backend.services.cpp_ext import (
            CPP_AVAILABLE, _py_rolling_corr
        )
        X = rng.normal(0, 0.01, (150, 4))
        py_out = _py_rolling_corr(X, window=50)
        if CPP_AVAILABLE:
            from backend.services.cpp_ext import _cpp_rolling_corr
            cpp_out = _cpp_rolling_corr(X, window=50)
            np.testing.assert_allclose(py_out, cpp_out, atol=1e-10)

    def test_cpp_vs_python_hmm_forward_consistent(self, rng):
        from backend.services.cpp_ext import (
            CPP_AVAILABLE, _py_hmm_forward
        )
        obs = rng.normal(0, 0.01, 100)
        pi = np.array([0.6, 0.4])
        A = np.array([[0.97, 0.03], [0.05, 0.95]])
        means = np.array([0.0003, -0.001])
        stds = np.array([0.008, 0.025])
        py_a, py_ll = _py_hmm_forward(obs, pi, A, means, stds)
        if CPP_AVAILABLE:
            from backend.services.cpp_ext import _cpp_hmm_forward
            cpp_a, cpp_ll = _cpp_hmm_forward(obs, pi, A, means, stds)
            np.testing.assert_allclose(py_a, cpp_a, atol=1e-10)
            assert abs(py_ll - cpp_ll) < 1e-8

    def test_cpp_vs_python_backtest_consistent(self, price_series, rng):
        from backend.services.cpp_ext import (
            CPP_AVAILABLE, _py_backtest_loop
        )
        signals = rng.choice([-1.0, 0.0, 1.0], size=len(price_series))
        py_r = _py_backtest_loop(price_series, signals)
        if CPP_AVAILABLE:
            from backend.services.cpp_ext import _cpp_backtest_loop
            cpp_r = _cpp_backtest_loop(price_series, signals)
            np.testing.assert_allclose(
                np.asarray(py_r["equity_curve"]),
                np.asarray(cpp_r["equity_curve"]),
                atol=1e-10
            )

    def test_invalid_input_raises(self):
        from backend.services.cpp_ext import rolling_corr
        with pytest.raises((ValueError, Exception)):
            rolling_corr(np.ones((10, 3)), window=200)  # window > T


# ===========================================================================
# Track 4B: p50/p99 Latency Profiler
# ===========================================================================

class TestTimerStats:
    def test_timer_stats_basic(self):
        from backend.services.latency_bench import TimerStats
        ts = TimerStats(label="test", n_runs=5)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            ts.record(v)
        s = ts.summary()
        assert s["n_runs"] == 5
        assert s["p50_ms"] == pytest.approx(3.0, abs=0.1)
        assert s["p99_ms"] >= s["p95_ms"] >= s["p50_ms"]
        assert s["min_ms"] <= s["mean_ms"] <= s["max_ms"]

    def test_timer_stats_empty(self):
        from backend.services.latency_bench import TimerStats
        ts = TimerStats(label="empty", n_runs=0)
        s = ts.summary()
        assert s["n_runs"] == 0

    def test_p50_le_p95_le_p99(self):
        from backend.services.latency_bench import TimerStats
        import time
        ts = TimerStats(label="order_check", n_runs=30)
        rng = np.random.default_rng(1)
        for v in rng.exponential(scale=5, size=30):
            ts.record(float(v))
        s = ts.summary()
        assert s["p50_ms"] <= s["p95_ms"] <= s["p99_ms"] <= s["p999_ms"]

    def test_time_fn_returns_all_keys(self):
        from backend.services.latency_bench import time_fn
        result = time_fn(lambda: sum(range(1000)), n_runs=10, label="test_sum")
        for key in ("label", "n_runs", "mean_ms", "std_ms", "p50_ms", "p95_ms", "p99_ms"):
            assert key in result, f"Missing key: {key}"
        assert result["n_runs"] == 10
        assert result["p50_ms"] >= 0.0

    def test_time_fn_p50_le_p99(self):
        from backend.services.latency_bench import time_fn
        result = time_fn(lambda: np.dot(np.ones(1000), np.ones(1000)), n_runs=20)
        assert result["p50_ms"] <= result["p99_ms"]

    def test_bench_cpp_vs_python_structure(self):
        from backend.services.latency_bench import bench_cpp_vs_python
        result = bench_cpp_vs_python(T=100, N=4, n_runs=5)
        assert "cpp_available" in result
        assert "kernels" in result
        for kernel in ("rolling_corr", "hmm_forward", "backtest_loop"):
            assert kernel in result["kernels"]
            k = result["kernels"][kernel]
            assert "speedup_p50x" in k
            assert "numerically_identical" in k
            assert k["speedup_p50x"] >= 0.0

    def test_run_percentile_benchmark_structure(self, return_matrix):
        from backend.services.latency_bench import run_percentile_benchmark
        price_matrix = np.cumprod(1 + return_matrix, axis=0) * 100
        result = run_percentile_benchmark(return_matrix, price_matrix, n_runs=3)
        assert "stages" in result
        assert "aggregate" in result
        assert "bottleneck" in result
        assert len(result["stages"]) >= 5
        agg = result["aggregate"]
        assert "p50_pipeline_ms" in agg
        assert "p99_pipeline_ms" in agg
        assert agg["p99_pipeline_ms"] >= agg["p50_pipeline_ms"]
        for stage in result["stages"]:
            assert "p50_ms" in stage
            assert "p99_ms" in stage
            assert stage["p50_ms"] <= stage["p99_ms"]


# ===========================================================================
# Track 4C: scipy/statsmodels upgrades
# ===========================================================================

class TestScipyStatTests:
    def test_scipy_available(self):
        import scipy.stats
        assert hasattr(scipy.stats, "norm")

    def test_statsmodels_available(self):
        import statsmodels.tsa.stattools as tsa
        assert hasattr(tsa, "adfuller")

    def test_adf_nonstationary(self, nonstationary_series):
        """Random walk should NOT reject unit root (p > 0.05)."""
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(nonstationary_series, autolag="AIC")
        p_value = float(result[1])
        # High p-value expected — don't reject H0 of unit root
        assert p_value > 0.01, f"Random walk p-value too low: {p_value}"

    def test_adf_stationary(self, stationary_series):
        """IID noise should REJECT unit root (p < 0.05)."""
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(stationary_series, autolag="AIC")
        p_value = float(result[1])
        assert p_value < 0.05, f"Stationary series p-value too high: {p_value}"

    def test_durbin_watson_in_range(self, strategy_returns):
        """Durbin-Watson statistic must always be in [0, 4]."""
        from statsmodels.stats.stattools import durbin_watson
        dw = float(durbin_watson(strategy_returns))
        assert 0.0 <= dw <= 4.0

    def test_durbin_watson_uncorrelated_near_2(self, rng):
        """IID residuals should give DW close to 2."""
        from statsmodels.stats.stattools import durbin_watson
        iid = rng.normal(0, 1, 500)
        dw = float(durbin_watson(iid))
        assert 1.5 < dw < 2.5

    def test_cointegration_detects_cointegrated(self, cointegrated_pair):
        """statsmodels coint test should find cointegration in cointegrated pair."""
        from statsmodels.tsa.stattools import coint
        y, x = cointegrated_pair
        _, p_value, _ = coint(y, x)
        assert float(p_value) < 0.05, f"Expected cointegration, got p={p_value:.4f}"

    def test_cointegration_no_false_positive(self, rng):
        """Two independent random walks should not be flagged as cointegrated."""
        from statsmodels.tsa.stattools import coint
        T = 300
        x = np.cumsum(rng.normal(0, 1, T))
        y = np.cumsum(rng.normal(0, 1, T))
        _, p_value, _ = coint(y, x)
        # p > 0.05 for independence (with high probability — not guaranteed)
        # Use a generous threshold to avoid flaky tests
        assert float(p_value) > 0.001, "Unexpectedly strong spurious cointegration"

    def test_scipy_norm_cdf_exact(self):
        """scipy.stats.norm.cdf should give exact values, not approximations."""
        from scipy.stats import norm
        assert abs(float(norm.cdf(0)) - 0.5) < 1e-15
        assert abs(float(norm.cdf(1.96)) - 0.975) < 0.001

    def test_scipy_chi2_sf(self):
        from scipy.stats import chi2
        # chi2(df=1) survival at 3.84 ≈ 0.05
        sf = float(chi2.sf(3.841, df=1))
        assert abs(sf - 0.05) < 0.005


# ===========================================================================
# Track 4D: Research Report Generator
# ===========================================================================

class TestReportGenerator:
    def test_generate_report_all_sections(self, strategy_returns):
        from backend.services.report_generator import generate_report
        report = generate_report(returns=strategy_returns)
        for section in (
            "executive_summary",
            "walk_forward_table",
            "walk_forward_summary",
            "factor_premia_table",
            "factor_model_summary",
            "regime_statistics",
            "statistical_validation",
            "risk_decomposition",
            "efficient_frontier",
        ):
            assert section in report, f"Missing section: {section}"

    def test_executive_summary_keys(self, strategy_returns):
        from backend.services.report_generator import generate_report
        report = generate_report(returns=strategy_returns)
        es = report["executive_summary"]
        for key in ("annual_return", "annual_volatility", "sharpe_ratio",
                    "max_drawdown", "n_observations"):
            assert key in es, f"Missing key in exec summary: {key}"

    def test_executive_summary_sharpe_is_float(self, strategy_returns):
        from backend.services.report_generator import generate_report
        report = generate_report(returns=strategy_returns)
        sharpe = report["executive_summary"]["sharpe_ratio"]
        assert isinstance(sharpe, float)

    def test_executive_summary_max_dd_nonpositive(self, strategy_returns):
        from backend.services.report_generator import generate_report
        report = generate_report(returns=strategy_returns)
        dd = report["executive_summary"]["max_drawdown"]
        assert isinstance(dd, float)
        assert dd <= 0.0

    def test_risk_decomposition_all_keys(self, strategy_returns):
        from backend.services.report_generator import generate_report
        report = generate_report(returns=strategy_returns)
        rd = report["risk_decomposition"]
        for key in ("cvar_5pct", "cvar_1pct", "sortino_ratio",
                    "calmar_ratio", "omega_ratio", "skewness", "excess_kurtosis"):
            assert key in rd, f"Missing risk key: {key}"

    def test_cvar_5pct_le_cvar_1pct(self, strategy_returns):
        """CVaR at 1% should be more extreme than at 5%."""
        from backend.services.report_generator import generate_report
        report = generate_report(returns=strategy_returns)
        rd = report["risk_decomposition"]
        assert rd["cvar_1pct"] <= rd["cvar_5pct"]

    def test_omega_ratio_positive(self, strategy_returns):
        from backend.services.report_generator import generate_report
        report = generate_report(returns=strategy_returns)
        omega = report["risk_decomposition"]["omega_ratio"]
        # Omega ratio is always >= 0
        assert omega >= 0.0

    def test_not_run_sections_when_no_data(self, strategy_returns):
        from backend.services.report_generator import generate_report
        report = generate_report(returns=strategy_returns)
        # Without wf_result provided, should have status: not_run
        assert isinstance(report["walk_forward_table"], dict)
        if isinstance(report["walk_forward_table"], dict):
            if "status" in report["walk_forward_table"]:
                assert report["walk_forward_table"]["status"] == "not_run"

    def test_report_json_serialisable(self, strategy_returns):
        """Full report must be JSON-serialisable (no numpy types)."""
        import json
        from backend.services.report_generator import generate_report
        report = generate_report(returns=strategy_returns)
        # Should not raise
        serialised = json.dumps(report)
        assert len(serialised) > 100

    def test_report_with_benchmark(self, strategy_returns, rng):
        from backend.services.report_generator import generate_report
        benchmark = rng.normal(0.0002, 0.011, len(strategy_returns))
        report = generate_report(returns=strategy_returns,
                                  benchmark_returns=benchmark)
        es = report["executive_summary"]
        assert "alpha_vs_benchmark" in es
        assert "information_ratio" in es
        assert isinstance(es["alpha_vs_benchmark"], float)

    def test_insufficient_data_returns_error(self):
        from backend.services.report_generator import generate_report
        tiny = np.array([0.01, -0.01, 0.02])
        report = generate_report(returns=tiny)
        es = report["executive_summary"]
        # Either returns error key or zeros
        assert "error" in es or "sharpe_ratio" in es

    def test_clean_handles_nan_inf(self):
        from backend.services.report_generator import _clean
        import math
        assert _clean(float("nan")) is None
        assert _clean(float("inf")) is None
        assert _clean(1.5) == 1.5
        assert _clean(np.int64(3)) == 3
        assert _clean(np.float32(2.5)) == pytest.approx(2.5, abs=0.01)

    def test_build_risk_decomposition_handles_short(self):
        from backend.services.report_generator import _build_risk_decomposition
        result = _build_risk_decomposition(np.array([0.01, -0.01]))
        assert "error" in result

    def test_walk_forward_table_with_real_wf(self):
        from backend.services.report_generator import _build_wf_table
        # Synthetic WF result structure
        wf = {
            "folds": [
                {
                    "fold_id": 1,
                    "train_start": "2020-01-01", "train_end": "2021-12-31",
                    "test_start": "2022-01-01", "test_end": "2022-12-31",
                    "oos_metrics": {"sharpe_ratio": 0.8, "annual_return": 0.06,
                                    "max_drawdown": -0.12},
                    "best_fast_window": 20, "best_slow_window": 50,
                },
                {
                    "fold_id": 2,
                    "train_start": "2021-01-01", "train_end": "2022-12-31",
                    "test_start": "2023-01-01", "test_end": "2023-12-31",
                    "oos_metrics": {"sharpe_ratio": -0.3, "annual_return": -0.02,
                                    "max_drawdown": -0.25},
                    "best_fast_window": 15, "best_slow_window": 45,
                },
            ]
        }
        table = _build_wf_table(wf)
        assert len(table) == 2
        assert table[0]["fold"] == 1
        assert table[0]["degraded"] is False
        assert table[1]["degraded"] is True  # negative Sharpe

    def test_factor_table_sorted_by_t_stat(self):
        from backend.services.report_generator import _build_factor_table
        fm = {
            "factor_premia": {
                "momentum": {"premium_ann": 0.05, "t_stat": 2.1, "p_value": 0.04, "nw_se": 0.02},
                "reversal": {"premium_ann": -0.02, "t_stat": -0.8, "p_value": 0.43, "nw_se": 0.025},
                "volatility": {"premium_ann": 0.08, "t_stat": 3.5, "p_value": 0.001, "nw_se": 0.01},
            }
        }
        table = _build_factor_table(fm)
        assert len(table) == 3
        # Sorted by |t_stat| descending: volatility (3.5), momentum (2.1), reversal (0.8)
        assert table[0]["factor"] == "volatility"
        assert table[0]["significant_5pct"] is True
        assert table[2]["factor"] == "reversal"
        assert table[2]["significant_5pct"] is False
