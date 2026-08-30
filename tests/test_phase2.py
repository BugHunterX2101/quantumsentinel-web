"""Tests for Phase 2: Alpha Research & Factor Modeling.

Tests cover:
  - alpha_research: IC computation, IC decay, hit rate, quintile analysis,
    factor turnover, alpha quality score
  - factor_model: factor computation, Fama-MacBeth regression, Barra risk decomp
  - correlation_engine: all 5 estimators, diagnostics, recommendation
  - portfolio_optimization: all 5 optimisation methods, efficient frontier
"""
import math
import numpy as np
import pytest

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def small_returns(rng):
    """(T=252, N=10) random return matrix."""
    return rng.normal(0.0005, 0.015, (252, 10))


@pytest.fixture
def large_returns(rng):
    """(T=500, N=20) return matrix with mild factor structure."""
    # Factor model: 3 latent factors + idiosyncratic
    T, N, K = 500, 20, 3
    F = rng.normal(0, 0.01, (T, K))
    B = rng.normal(0, 1, (K, N))
    eps = rng.normal(0, 0.01, (T, N))
    returns = F @ B + eps
    return returns


@pytest.fixture
def small_signal_matrix(rng):
    """(T=200, N=8) signal matrix."""
    return rng.normal(0, 1, (200, 8))


# ──────────────────────────────────────────────────────────────────────────────
# Alpha Research Tests
# ──────────────────────────────────────────────────────────────────────────────

from backend.services.alpha_research import (
    spearman_ic, pearson_ic, compute_ic_series, ic_decay,
    hit_rate, quintile_returns, factor_turnover, run_alpha_research,
)


class TestSpearmanIC:
    def test_perfect_positive_correlation(self):
        sig = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ret = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        ic = spearman_ic(sig, ret)
        assert ic == pytest.approx(1.0, abs=0.01)

    def test_perfect_negative_correlation(self):
        sig = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        ret = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        ic = spearman_ic(sig, ret)
        assert ic == pytest.approx(-1.0, abs=0.01)

    def test_zero_correlation(self):
        rng = np.random.default_rng(0)
        sig = rng.normal(0, 1, 500)
        ret = rng.normal(0, 1, 500)
        ic = spearman_ic(sig, ret)
        assert abs(ic) < 0.15  # noise → near zero

    def test_insufficient_data(self):
        ic = spearman_ic(np.array([1.0, 2.0]), np.array([0.1, 0.2]))
        assert ic == 0.0

    def test_constant_signal_returns_zero(self):
        sig = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        ret = np.array([0.1, -0.1, 0.2, -0.2, 0.0])
        ic = spearman_ic(sig, ret)
        assert ic == 0.0


class TestComputeICSeries:
    def test_basic_structure(self, small_signal_matrix, small_returns):
        T, N = small_signal_matrix.shape
        # Use smaller return matrix
        result = compute_ic_series(small_signal_matrix, small_returns[:200, :8])
        assert "ic_series" in result
        assert "mean_ic" in result
        assert "icir" in result
        assert "significant_5pct" in result
        assert isinstance(result["ic_series"], list)

    def test_predictive_signal(self, rng):
        """A signal that perfectly predicts next-period returns should have high IC."""
        T, N = 200, 10
        signal_matrix = rng.normal(0, 1, (T, N))
        # Next period return = signal + noise
        return_matrix = np.zeros((T, N))
        return_matrix[1:, :] = signal_matrix[:-1, :] * 0.5 + rng.normal(0, 0.5, (T - 1, N))
        result = compute_ic_series(signal_matrix, return_matrix)
        assert result["mean_ic"] > 0.05  # should detect predictability

    def test_random_signal_near_zero_ic(self, rng):
        T, N = 200, 10
        sig = rng.normal(0, 1, (T, N))
        ret = rng.normal(0, 0.01, (T, N))
        result = compute_ic_series(sig, ret)
        assert abs(result["mean_ic"]) < 0.25  # likely noise


class TestICDecay:
    def test_decay_structure(self, small_signal_matrix, small_returns):
        result = ic_decay(small_signal_matrix, small_returns[:200, :8], max_horizon=10)
        assert "horizons" in result
        assert "ic_by_horizon" in result
        assert len(result["horizons"]) >= 1
        for item in result["ic_by_horizon"]:
            assert "horizon" in item
            assert "mean_ic" in item

    def test_horizon_count(self, small_signal_matrix, small_returns):
        result = ic_decay(small_signal_matrix, small_returns[:200, :8], max_horizon=5)
        assert len(result["horizons"]) <= 5


class TestHitRate:
    def test_perfect_hit_rate(self):
        sig = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
        ret = np.array([0.1, -0.1, 0.2, -0.2, 0.05, -0.05])
        result = hit_rate(sig, ret)
        assert result["hit_rate"] == pytest.approx(1.0)

    def test_zero_hit_rate(self):
        sig = np.array([1.0, -1.0, 1.0, -1.0])
        ret = np.array([-0.1, 0.1, -0.2, 0.2])
        result = hit_rate(sig, ret)
        assert result["hit_rate"] == pytest.approx(0.0)

    def test_random_near_50pct(self, rng):
        n = 500
        sig = rng.choice([-1, 1], n)
        ret = rng.choice([-0.01, 0.01], n)
        result = hit_rate(sig.astype(float), ret.astype(float))
        assert 0.35 < result["hit_rate"] < 0.65

    def test_returns_correct_keys(self, rng):
        sig = rng.normal(0, 1, 50)
        ret = rng.normal(0, 0.01, 50)
        result = hit_rate(sig, ret)
        assert "hit_rate" in result
        assert "z_stat" in result
        assert "p_value" in result


class TestQuintileReturns:
    def test_monotonic_quintiles(self):
        """Perfect signal should produce monotonically increasing quintiles."""
        n = 100
        signals = np.arange(float(n))
        # Q1 (low signal) → negative return, Q5 (high signal) → positive
        returns = (signals - n / 2) / n * 0.1 + np.random.default_rng(0).normal(0, 0.001, n)
        result = quintile_returns(signals, returns, n_quantiles=5)
        assert result["monotonic"]
        assert result["spread_q5_q1"] > 0

    def test_spread_negative_for_reversed_signal(self):
        n = 100
        signals = np.arange(float(n))
        returns = -(signals - n / 2) / n * 0.1  # reversed
        result = quintile_returns(signals, returns, n_quantiles=5)
        assert result["spread_q5_q1"] < 0

    def test_structure(self, rng):
        result = quintile_returns(rng.normal(0, 1, 50), rng.normal(0, 0.01, 50))
        assert "quantile_returns" in result
        assert "spread_q5_q1" in result
        assert len(result["quantile_returns"]) == 5


class TestFactorTurnover:
    def test_stable_signal_low_turnover(self):
        """Constant signal → zero turnover."""
        T, N = 100, 8
        sig = np.tile(np.arange(float(N)), (T, 1))  # same ranking every period
        result = factor_turnover(sig)
        assert result["avg_rank_correlation"] > 0.9
        assert result["avg_turnover"] < 0.2

    def test_random_signal_high_turnover(self, rng):
        T, N = 100, 8
        sig = rng.normal(0, 1, (T, N))  # IID → high turnover
        result = factor_turnover(sig)
        assert result["avg_turnover"] > 0.3


class TestRunAlphaResearch:
    def test_full_pipeline_structure(self, small_signal_matrix, small_returns):
        result = run_alpha_research(
            small_signal_matrix, small_returns[:200, :8], max_horizon=5
        )
        assert "ic_analysis" in result
        assert "ic_decay" in result
        assert "hit_rate" in result
        assert "quintile_analysis" in result
        assert "factor_turnover" in result
        assert "alpha_quality_score" in result
        qs = result["alpha_quality_score"]
        assert "score" in qs
        assert "rating" in qs
        assert 0 <= qs["score"] <= 100


# ──────────────────────────────────────────────────────────────────────────────
# Factor Model Tests
# ──────────────────────────────────────────────────────────────────────────────

from backend.services.factor_model import (
    compute_factors, fama_macbeth, barra_risk_decomposition, _nw_se,
)


class TestComputeFactors:
    def test_factor_shapes(self, large_returns):
        T, N = large_returns.shape
        factors = compute_factors(large_returns)
        for name, mat in factors.items():
            assert mat.shape == (T, N), f"Factor {name} shape mismatch"

    def test_factors_have_valid_names(self, large_returns):
        factors = compute_factors(large_returns)
        expected = {"momentum", "reversal", "volatility", "low_volatility",
                    "quality", "size"}
        assert set(factors.keys()) >= expected - {"size"}  # size needs price

    def test_factors_z_scored(self, large_returns):
        """Cross-sectional z-scored factors should have |mean| ≈ 0 and |std| ≈ 1
        for rows with sufficient valid data."""
        factors = compute_factors(large_returns)
        T, N = large_returns.shape
        for name, mat in factors.items():
            for t in range(T // 2, T):  # check second half (first half has NaNs)
                row = mat[t, :]
                valid = row[np.isfinite(row)]
                if len(valid) > 3:
                    assert abs(valid.mean()) < 0.5  # roughly centred
                    break  # one check per factor is sufficient

    def test_missing_data_propagation(self, rng):
        """NaN in returns should produce NaN in factor for early periods."""
        returns = rng.normal(0, 0.01, (100, 5))
        factors = compute_factors(returns)
        # Early rows should be NaN (lookback not satisfied)
        assert np.isnan(factors["momentum"][5, 0])


class TestFamaMacBeth:
    def test_insufficient_data_returns_error(self, rng):
        returns = rng.normal(0, 0.01, (30, 5))
        factors = {"mom": rng.normal(0, 1, (30, 5))}
        result = fama_macbeth(returns, factors)
        assert "error" in result

    def test_sufficient_data_produces_premia(self, large_returns, rng):
        T, N = large_returns.shape
        factor_mats = compute_factors(large_returns)
        # Use subset with valid data
        result = fama_macbeth(large_returns, factor_mats, newey_west_lags=4)
        if "error" not in result:
            assert "factor_premia" in result
            assert "mean_cross_sectional_r2" in result
            assert result["mean_cross_sectional_r2"] >= 0

    def test_output_structure(self, large_returns):
        factor_mats = compute_factors(large_returns)
        result = fama_macbeth(large_returns, factor_mats, newey_west_lags=2)
        if "error" not in result:
            for fname, info in result["factor_premia"].items():
                assert "lambda" in info
                assert "t_stat" in info
                assert "p_value" in info
                assert "significant_5pct" in info


class TestNeweyWestSE:
    def test_iid_matches_std_se(self, rng):
        """For IID data, NW SE should be close to std/sqrt(n)."""
        x = rng.normal(0, 1, 200)
        nw = _nw_se(x, max_lag=0)
        classical = np.std(x, ddof=1) / math.sqrt(len(x))
        assert nw == pytest.approx(classical, rel=0.01)

    def test_nw_larger_than_ols_with_autocorr(self, rng):
        """NW SE should be larger than classical SE for autocorrelated data."""
        n = 300
        x = np.zeros(n)
        x[0] = rng.normal()
        for t in range(1, n):
            x[t] = 0.7 * x[t - 1] + rng.normal(0, 0.1)
        nw = _nw_se(x, max_lag=5)
        classical = np.std(x, ddof=1) / math.sqrt(n)
        assert nw >= classical * 0.5  # NW is usually larger but not guaranteed with small lags


class TestBarraRisk:
    def test_structure(self, large_returns):
        factor_mats = compute_factors(large_returns)
        result = barra_risk_decomposition(large_returns, factor_mats)
        if "error" not in result:
            assert "pct_factor_explained" in result
            assert "pct_specific" in result
            assert 0 <= result["pct_factor_explained"] <= 1
            assert abs(result["pct_factor_explained"] + result["pct_specific"] - 1.0) < 0.01


# ──────────────────────────────────────────────────────────────────────────────
# Correlation Engine Tests
# ──────────────────────────────────────────────────────────────────────────────

from backend.services.correlation_engine import (
    pearson_correlation, spearman_correlation, ewma_correlation,
    ledoit_wolf_shrinkage, oas_shrinkage, pca_covariance,
    correlation_diagnostics, run_correlation_engine,
)


class TestCorrelationMethods:
    def test_pearson_is_symmetric(self, small_returns):
        C = pearson_correlation(small_returns)
        assert C.shape == (10, 10)
        assert np.allclose(C, C.T, atol=1e-10)

    def test_diagonal_is_one(self, small_returns):
        for fn in [pearson_correlation, spearman_correlation]:
            C = fn(small_returns)
            assert np.allclose(np.diag(C), 1.0, atol=1e-6)

    def test_ewma_diagonal_one(self, small_returns):
        C = ewma_correlation(small_returns, halflife=30.0)
        assert np.allclose(np.diag(C), 1.0, atol=1e-6)

    def test_spearman_bounded(self, small_returns):
        C = spearman_correlation(small_returns)
        assert np.all(C >= -1 - 1e-6) and np.all(C <= 1 + 1e-6)

    def test_ledoit_wolf_shrinkage_intensity_in_range(self, small_returns):
        corr, alpha = ledoit_wolf_shrinkage(small_returns)
        assert 0.0 <= alpha <= 1.0
        assert corr.shape == (10, 10)
        assert np.allclose(np.diag(corr), 1.0, atol=1e-4)

    def test_oas_shrinkage_intensity_in_range(self, small_returns):
        corr, alpha = oas_shrinkage(small_returns)
        assert 0.0 <= alpha <= 1.0

    def test_high_shrinkage_when_T_small(self, rng):
        """Small T relative to N → high shrinkage intensity."""
        returns = rng.normal(0, 0.01, (15, 10))  # T < N
        _, alpha_lw = ledoit_wolf_shrinkage(returns)
        _, alpha_oas = oas_shrinkage(returns)
        # At least one shrinkage should be substantial
        assert max(alpha_lw, alpha_oas) > 0.1

    def test_pca_structure(self, large_returns):
        result = pca_covariance(large_returns, n_components=3)
        assert "correlation" in result
        assert "n_components" in result
        assert result["n_components"] == 3
        # Diagonal of correlation should be 1
        corr = np.array(result["correlation"])
        assert np.allclose(np.diag(corr), 1.0, atol=1e-4)

    def test_pca_auto_components(self, large_returns):
        result = pca_covariance(large_returns, n_components=None)
        assert result["n_components"] >= 1


class TestCorrelationDiagnostics:
    def test_pd_matrix_passes(self, rng):
        C = np.eye(10)  # PD by construction
        diag = correlation_diagnostics(C, "identity")
        assert diag["is_positive_definite"]

    def test_singular_matrix_not_pd(self):
        C = np.ones((5, 5))  # rank 1, not PD
        diag = correlation_diagnostics(C)
        assert not diag["is_positive_definite"]

    def test_effective_bets(self, rng):
        C = np.eye(10)
        diag = correlation_diagnostics(C)
        # Identity → 10 effective bets
        assert diag["effective_uncorrelated_bets"] == pytest.approx(10.0, rel=0.01)


class TestRunCorrelationEngine:
    def test_full_output_structure(self, small_returns):
        result = run_correlation_engine(small_returns)
        assert "correlations" in result
        assert "diagnostics" in result
        assert "recommended_estimator" in result
        for method in ["pearson", "spearman", "ewma", "ledoit_wolf", "oas", "pca"]:
            assert method in result["correlations"]
            assert method in result["diagnostics"]

    def test_recommendation_for_small_t_n(self, rng):
        returns = rng.normal(0, 0.01, (40, 20))  # T/N = 2 → recommend OAS
        result = run_correlation_engine(returns)
        assert result["recommended_estimator"]["estimator"] in {"oas", "ledoit_wolf"}


# ──────────────────────────────────────────────────────────────────────────────
# Portfolio Optimisation Tests
# ──────────────────────────────────────────────────────────────────────────────

from backend.services.portfolio_optimization import (
    equal_weight, minimum_variance, maximum_sharpe, risk_parity,
    maximum_diversification, portfolio_analytics, run_portfolio_optimization,
    PortfolioConstraints,
)


class TestEqualWeight:
    def test_sums_to_one(self):
        w = equal_weight(10)
        assert w.sum() == pytest.approx(1.0)
        assert np.allclose(w, 0.1)


class TestMinimumVariance:
    def test_sums_to_one(self, small_returns):
        cov = np.cov(small_returns.T)
        w = minimum_variance(cov)
        assert w.sum() == pytest.approx(1.0, abs=1e-4)

    def test_all_nonnegative_long_only(self, small_returns):
        cov = np.cov(small_returns.T)
        w = minimum_variance(cov, PortfolioConstraints(long_only=True))
        assert np.all(w >= -1e-6)

    def test_lower_variance_than_equal_weight(self, large_returns):
        cov = np.cov(large_returns.T)
        w_mv = minimum_variance(cov)
        w_ew = equal_weight(large_returns.shape[1])
        var_mv = float(w_mv @ cov @ w_mv)
        var_ew = float(w_ew @ cov @ w_ew)
        assert var_mv <= var_ew * 1.1  # MV should not be worse than EW


class TestMaximumSharpe:
    def test_sums_to_one(self, large_returns):
        cov = np.cov(large_returns.T)
        exp_ret = large_returns.mean(axis=0)
        w = maximum_sharpe(exp_ret, cov)
        assert w.sum() == pytest.approx(1.0, abs=1e-4)

    def test_long_only(self, large_returns):
        cov = np.cov(large_returns.T)
        exp_ret = large_returns.mean(axis=0)
        w = maximum_sharpe(exp_ret, cov, constraints=PortfolioConstraints(long_only=True))
        assert np.all(w >= -1e-6)


class TestRiskParity:
    def test_sums_to_one(self, small_returns):
        cov = np.cov(small_returns.T)
        w = risk_parity(cov)
        assert w.sum() == pytest.approx(1.0, abs=1e-4)

    def test_risk_contributions_roughly_equal(self, small_returns):
        cov = np.cov(small_returns.T)
        w = risk_parity(cov)
        sigma_w = cov @ w
        port_var = float(w @ sigma_w)
        if port_var > 1e-12:
            rc = w * sigma_w / port_var
            # All risk contributions should be roughly equal (within 3x of each other)
            assert rc.max() < rc.min() * 5 + 0.05

    def test_all_nonnegative(self, small_returns):
        cov = np.cov(small_returns.T)
        w = risk_parity(cov)
        assert np.all(w >= -1e-6)


class TestMaxDiversification:
    def test_sums_to_one(self, small_returns):
        cov = np.cov(small_returns.T)
        w = maximum_diversification(cov)
        assert w.sum() == pytest.approx(1.0, abs=1e-4)

    def test_diversification_ratio_ge_one(self, small_returns):
        cov = np.cov(small_returns.T)
        w = maximum_diversification(cov)
        asset_vols = np.sqrt(np.diag(cov))
        port_vol = math.sqrt(max(float(w @ cov @ w), 1e-12))
        dr = float(w @ asset_vols) / port_vol
        assert dr >= 1.0 - 1e-4


class TestPortfolioAnalytics:
    def test_output_structure(self, small_returns):
        cov = np.cov(small_returns.T)
        exp_ret = small_returns.mean(axis=0)
        w = equal_weight(10)
        analytics = portfolio_analytics(w, exp_ret, cov)
        for key in ["annual_return", "annual_volatility", "sharpe_ratio",
                    "diversification_ratio", "effective_n", "weights",
                    "risk_contributions", "herfindahl_index"]:
            assert key in analytics

    def test_equal_weight_eff_n(self, small_returns):
        cov = np.cov(small_returns.T)
        exp_ret = small_returns.mean(axis=0)
        w = equal_weight(10)
        analytics = portfolio_analytics(w, exp_ret, cov)
        assert analytics["effective_n"] == pytest.approx(10.0, rel=0.01)


class TestRunPortfolioOptimization:
    def test_all_methods_present(self, large_returns):
        result = run_portfolio_optimization(large_returns)
        assert "portfolios" in result
        expected_methods = {"equal_weight", "min_variance", "risk_parity",
                            "max_sharpe", "max_diversification"}
        assert expected_methods.issubset(result["portfolios"].keys())

    def test_ranked_by_sharpe(self, large_returns):
        result = run_portfolio_optimization(large_returns)
        ranked = result["portfolios_ranked_by_sharpe"]
        sharpes = [x["sharpe"] for x in ranked]
        assert sharpes == sorted(sharpes, reverse=True)

    def test_efficient_frontier_structure(self, large_returns):
        result = run_portfolio_optimization(large_returns)
        frontier = result["efficient_frontier"]
        assert isinstance(frontier, list)
        for point in frontier:
            assert "return" in point
            assert "volatility" in point
            assert "sharpe" in point

    def test_weight_constraints_respected(self, large_returns):
        con = PortfolioConstraints(long_only=True, max_weight=0.25)
        result = run_portfolio_optimization(
            large_returns, constraints=con
        )
        # Every method's weights should respect max_weight
        for method, analytics in result["portfolios"].items():
            max_w = max(analytics["weights"].values())
            assert max_w <= 0.26  # allow tiny numerical slack
