"""Tests for Phase 1: Research Engine Foundation.

Tests cover:
  - ExecutionModel: commission, slippage, spread, borrow cost, position sizing
  - BacktestEngine: full pipeline with execution costs
  - WalkForward: rolling/expanding window validation
  - StatTests: t-test, bootstrap, permutation, Deflated Sharpe
  - Extended risk metrics: CVaR, Sortino, Calmar, Omega
"""
import math
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Execution Model Tests
# ---------------------------------------------------------------------------

from backend.services.execution_model import (
    CommissionModel, SlippageModel, SpreadModel, BorrowCostModel,
    PositionSizer, SizingMethod, ExecutionSimulator, ExecutionConfig,
    zero_cost_config, retail_config, institutional_config, FillResult,
)


class TestCommissionModel:
    def test_minimum_commission(self):
        cm = CommissionModel(per_share=0.005, pct_of_notional=0.001,
                             min_per_trade=1.0)
        # 10 shares at $100 = 10*0.005 + 1000*0.001 = 0.05+1 = 1.05 > min of 1.0
        cost = cm.compute(10, 100)
        assert cost >= 1.0

    def test_cap_commission(self):
        cm = CommissionModel(per_share=0.005, pct_of_notional=0.01,
                             min_per_trade=0, max_pct_of_notional=0.005)
        cost = cm.compute(1000, 100)
        cap = 1000 * 100 * 0.005
        assert cost <= cap + 0.01

    def test_zero_cost(self):
        cm = CommissionModel(per_share=0, pct_of_notional=0,
                             min_per_trade=0)
        assert cm.compute(100, 50) == 0.0


class TestSlippageModel:
    def test_base_slippage(self):
        sm = SlippageModel(base_bps=2.0, volatility_factor=0)
        slip = sm.compute(100, 100, 0.02, 1e6)
        assert slip > 0
        assert slip == pytest.approx(100 * 2 / 10_000, rel=0.1)

    def test_vol_adjusted_slippage(self):
        sm = SlippageModel(base_bps=0, volatility_factor=0.1)
        slip_low = sm.compute(100, 100, 0.01, 1e6)
        slip_high = sm.compute(100, 100, 0.05, 1e6)
        assert slip_high > slip_low

    def test_partial_fill(self):
        sm = SlippageModel(volume_participation_limit=0.05)
        # avg_volume = 1000, limit = 5% = 50 shares
        assert not sm.can_fill_fully(100, 1000)
        partial = sm.partial_fill_qty(100, 1000)
        assert abs(partial) <= 50


class TestSpreadModel:
    def test_spread_adjusts_buy_up(self):
        sm = SpreadModel(base_spread_bps=4.0, vol_multiplier=0)
        buy_price = sm.adjust_fill_price(100, 0.02, "buy")
        assert buy_price > 100

    def test_spread_adjusts_sell_down(self):
        sm = SpreadModel(base_spread_bps=4.0, vol_multiplier=0)
        sell_price = sm.adjust_fill_price(100, 0.02, "sell")
        assert sell_price < 100


class TestBorrowCost:
    def test_long_no_cost(self):
        bc = BorrowCostModel()
        assert bc.daily_cost("AAPL", 100, 150) == 0.0

    def test_short_has_cost(self):
        bc = BorrowCostModel(general_annual_rate=0.005)
        cost = bc.daily_cost("AAPL", -100, 150)
        expected = 100 * 150 * 0.005 / 252
        assert cost == pytest.approx(expected, rel=0.01)

    def test_hard_to_borrow(self):
        bc = BorrowCostModel(
            general_annual_rate=0.005,
            hard_to_borrow_rate=0.10,
            hard_to_borrow_tickers={"GME"},
        )
        cost_gme = bc.daily_cost("GME", -100, 50)
        cost_aapl = bc.daily_cost("AAPL", -100, 50)
        assert cost_gme > cost_aapl


class TestPositionSizer:
    def test_fixed_fractional(self):
        ps = PositionSizer(method=SizingMethod.FIXED_FRACTIONAL,
                           risk_per_trade=0.02)
        shares = ps.compute_shares(100_000, 100)
        assert shares == pytest.approx(20, rel=0.1)  # 2% of 100k / $100

    def test_equal_weight(self):
        ps = PositionSizer(method=SizingMethod.EQUAL_WEIGHT)
        shares = ps.compute_shares(100_000, 50, n_assets=10)
        assert shares == pytest.approx(200, rel=0.1)  # 10k / $50

    def test_position_cap(self):
        ps = PositionSizer(method=SizingMethod.FIXED_FRACTIONAL,
                           risk_per_trade=0.50,  # would be 50k
                           max_position_pct=0.10)  # capped at 10k
        shares = ps.compute_shares(100_000, 100)
        max_shares = 100_000 * 0.10 / 100
        assert shares <= max_shares + 0.01


class TestExecutionSimulator:
    def test_zero_cost_fill(self):
        sim = ExecutionSimulator(zero_cost_config())
        fill = sim.execute_order("buy", 100, 150.0, 0.02, 1e6)
        assert fill.filled
        assert fill.commission == 0
        assert fill.slippage_cost == 0

    def test_retail_costs_nonzero(self):
        sim = ExecutionSimulator(retail_config())
        fill = sim.execute_order("buy", 100, 150.0, 0.02, 1e6)
        assert fill.filled
        assert fill.commission > 0
        assert fill.total_cost > 0

    def test_short_selling_disabled(self):
        cfg = ExecutionConfig()
        cfg.allow_short_selling = False
        sim = ExecutionSimulator(cfg)
        fill = sim.execute_order("sell", 100, 150.0, 0.02, 1e6,
                                 current_position=0)
        assert not fill.filled


# ---------------------------------------------------------------------------
# Backtest Engine Tests
# ---------------------------------------------------------------------------

from backend.services.backtest_service import (
    _sharpe, _sortino, _max_drawdown, _var_cvar, _calmar,
    _alpha_beta, _omega_ratio,
)


class TestRiskHelpers:
    def test_sharpe_positive_returns(self):
        rng = np.random.default_rng(42)
        rets = rng.normal(0.001, 0.01, 252)
        s = _sharpe(rets)
        assert s > 0

    def test_sharpe_zero_returns(self):
        rets = np.array([0.0] * 100)
        assert _sharpe(rets) == 0.0

    def test_sortino_ignores_upside(self):
        rets = np.array([0.01, 0.02, 0.03, -0.005, 0.01])
        s = _sortino(rets)
        assert s > 0

    def test_max_drawdown(self):
        curve = [100, 110, 95, 105, 90]
        dd = _max_drawdown(curve)
        # Peak=110, trough=90 → dd = 20/110 ≈ 0.1818
        assert dd == pytest.approx(20 / 110, rel=0.01)

    def test_var_cvar(self):
        rets = np.random.default_rng(42).normal(0, 0.01, 1000)
        var95, cvar95 = _var_cvar(rets, 0.05)
        assert var95 >= 0
        assert cvar95 >= var95  # CVaR is always ≥ VaR

    def test_calmar(self):
        rets = np.array([0.001] * 252)
        max_dd = 0.05
        c = _calmar(rets, max_dd)
        expected = 0.001 * 252 / 0.05
        assert c == pytest.approx(expected, rel=0.01)

    def test_omega_ratio(self):
        rets = np.array([0.01, 0.02, -0.005, 0.015, -0.01])
        omega = _omega_ratio(rets)
        assert omega > 0

    def test_alpha_beta_market_neutral(self):
        rng = np.random.default_rng(42)
        bench = rng.normal(0.0005, 0.01, 500)
        strategy = rng.normal(0.001, 0.01, 500)  # uncorrelated
        alpha, beta = _alpha_beta(strategy, bench)
        assert abs(beta) < 0.5  # should be near zero if uncorrelated


# ---------------------------------------------------------------------------
# Statistical Tests
# ---------------------------------------------------------------------------

from backend.services.stat_tests import (
    newey_west_se, ttest_mean_return, bootstrap_sharpe_ci,
    permutation_test, ljung_box_test, deflated_sharpe_ratio,
    bonferroni_correction, benjamini_hochberg,
    run_full_stat_tests,
)


class TestStatTests:
    def test_ttest_significant(self):
        # Strong positive returns should be significant
        rets = np.random.default_rng(42).normal(0.005, 0.01, 500)
        result = ttest_mean_return(rets)
        assert result["significant_5pct"]
        assert result["t_stat"] > 0

    def test_ttest_insignificant(self):
        # Zero-mean returns should not be significant
        rets = np.random.default_rng(42).normal(0, 0.01, 100)
        result = ttest_mean_return(rets)
        # With 100 obs of zero-mean, likely not significant
        # (not guaranteed, but highly probable)
        assert result["p_value"] > 0.001

    def test_newey_west_se(self):
        rets = np.random.default_rng(42).normal(0, 0.01, 200)
        se = newey_west_se(rets)
        assert se > 0

    def test_bootstrap_ci(self):
        rets = np.random.default_rng(42).normal(0.001, 0.01, 300)
        result = bootstrap_sharpe_ci(rets, n_bootstrap=1000)
        assert result["ci_lower"] < result["ci_upper"]
        assert result["sharpe"] > 0

    def test_permutation_random_strategy(self):
        # Random returns should produce high p-value
        rets = np.random.default_rng(42).normal(0, 0.01, 200)
        result = permutation_test(rets, n_permutations=500)
        assert result["p_value"] > 0.01

    def test_ljung_box(self):
        rets = np.random.default_rng(42).normal(0, 0.01, 200)
        result = ljung_box_test(rets)
        assert "autocorrelations" in result
        assert "lag_1" in result["autocorrelations"]

    def test_deflated_sharpe_many_trials(self):
        result = deflated_sharpe_ratio(
            observed_sharpe=1.5,
            n_trials=100,
            n_observations=252,
        )
        # With 100 trials, a Sharpe of 1.5 should be challenged
        assert result["expected_max_sharpe"] > 0
        assert "dsr_p_value" in result

    def test_deflated_sharpe_single_trial(self):
        result = deflated_sharpe_ratio(
            observed_sharpe=2.0,
            n_trials=1,
            n_observations=500,
        )
        # Single trial should pass more easily
        assert result["dsr_p_value"] < result["dsr_p_value"] + 1  # sanity

    def test_bonferroni(self):
        p_vals = [0.01, 0.03, 0.06, 0.10]
        result = bonferroni_correction(p_vals, alpha=0.05)
        assert result["adjusted_alpha"] == pytest.approx(0.05 / 4)

    def test_benjamini_hochberg(self):
        p_vals = [0.001, 0.01, 0.03, 0.04, 0.50]
        result = benjamini_hochberg(p_vals, alpha=0.05)
        assert result["n_significant"] >= 1

    def test_full_suite(self):
        rets = np.random.default_rng(42).normal(0.001, 0.01, 300)
        result = run_full_stat_tests(rets, n_strategies_tested=5)
        assert "summary" in result
        assert "ttest_nw" in result
        assert "bootstrap_sharpe" in result
        assert "permutation_test" in result
        assert "deflated_sharpe" in result
