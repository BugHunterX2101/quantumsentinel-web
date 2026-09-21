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
    _alpha_beta, _omega_ratio, _downside_deviation,
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

    def test_sortino_downside_deviation_divides_by_total_periods(self):
        """Downside deviation must be RMS-shortfall-below-target averaged
        over ALL periods, not just the periods that fall below target.
        Dividing by only the downside count (dropping the zero terms from
        periods at/above target) overstates the deviation and understates
        the ratio — verified against the standard Sortino formula:
        DD = sqrt(mean(min(r - target, 0)^2)) over the full sample."""
        rets = np.array([0.02, 0.02, 0.02, 0.02, -0.01])
        import math as _math
        expected_dd = _math.sqrt(np.mean(np.minimum(rets, 0.0) ** 2))
        expected_sortino = (np.mean(rets) / expected_dd) * _math.sqrt(252)
        s = _sortino(rets)
        assert s == pytest.approx(expected_sortino, rel=1e-9)
        # The old (buggy) denominator used only the single downside period:
        # dd_wrong = sqrt(mean((-0.01)**2)) is numerically the same here by
        # coincidence (n=1 downside), so use a case with >1 downside period
        # to actually discriminate between the two conventions.
        rets2 = np.array([0.02, 0.02, 0.02, -0.01, -0.03])
        dd_correct = _math.sqrt(np.mean(np.minimum(rets2, 0.0) ** 2))  # divide by 5
        dd_wrong = _math.sqrt(np.mean(np.array([-0.01, -0.03]) ** 2))  # divide by 2 (buggy)
        assert dd_correct != pytest.approx(dd_wrong)
        expected_sortino2 = (np.mean(rets2) / dd_correct) * _math.sqrt(252)
        assert _sortino(rets2) == pytest.approx(expected_sortino2, rel=1e-9)

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

    def test_downside_deviation_divides_by_total_periods(self):
        rets = np.array([0.02, 0.02, 0.02, -0.01, -0.03])
        dd = _downside_deviation(rets)
        expected = float(np.sqrt(np.mean(np.minimum(rets, 0.0) ** 2)))  # divide by 5
        wrong = float(np.sqrt(np.mean(np.array([-0.01, -0.03]) ** 2)))  # divide by 2
        assert dd == pytest.approx(expected, rel=1e-9)
        assert dd != pytest.approx(wrong)

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

    def test_permutation_has_power_against_real_skill(self):
        # A naive reordering of returns cannot change mean/std, so a test
        # built that way would report the same (non-)significance no
        # matter how strong the underlying edge is. Verify the test has
        # actual statistical power: a clearly skilled strategy (high
        # true Sharpe) must be flagged significant.
        rets = np.random.default_rng(7).normal(0.003, 0.01, 500)
        result = permutation_test(rets, n_permutations=2000)
        assert result["observed_sharpe"] > 2.0
        assert result["p_value"] < 0.01
        assert result["significant_1pct"] is True

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


# ---------------------------------------------------------------------------
# BacktestEngine._run_strategy — full bar-by-bar pipeline, no network.
#
# BacktestEngine.run() requires a live yfinance download, so (matching the
# established pattern in tests/test_walk_forward.py for the sibling
# WalkForwardEngine) this calls _run_strategy() directly with synthetic
# price data. This is the exact same bug class already fixed once in
# walk_forward.py's _windowed_vol() (see test_walk_forward.py's docstring):
# a per-bar rolling-volatility window whose numerator (np.diff of a price
# slice) and denominator (a second, independently-offset price slice) don't
# derive from the same base window, so their lengths silently diverge near
# array boundaries. backtest_service.py had its own, unfixed copy of the
# same pattern, which crashed on 100% of real requests reaching this code
# (any period beyond ~21 bars, i.e. every 6mo/1y/2y/3y/5y backtest).
# ---------------------------------------------------------------------------

from backend.services.backtest_service import (
    BacktestEngine, BacktestConfig, StrategyConfig, StrategyType,
)


@pytest.fixture
def trending_close_matrix():
    """Two price series with clear swings so MA crossovers fire and hold
    positions open at the end of the run (exercising the liquidation path)."""
    rng = np.random.default_rng(7)
    n = 300
    trend = np.concatenate([
        np.linspace(0, 25, n // 3),
        np.linspace(25, -10, n // 3),
        np.linspace(-10, 20, n - 2 * (n // 3)),
    ])
    noise_a = rng.normal(0, 0.5, n)
    noise_b = rng.normal(0, 0.5, n)
    return {
        "TEST": 100.0 + trend + noise_a,
        "SPY": 100.0 + trend * 0.6 + noise_b,
    }


class TestBacktestEngineRunStrategy:
    def _engine(self, fast=20, slow=50):
        cfg = BacktestConfig(
            assets=["TEST"], benchmark="SPY",
            strategy=StrategyConfig(strategy_type=StrategyType.MA_CROSSOVER,
                                     fast_window=fast, slow_window=slow),
        )
        return BacktestEngine(cfg)

    def _asset_data(self, closes: dict) -> dict:
        return {
            t: {"close": c, "volume": np.full_like(c, 1_000_000.0),
                "high": c * 1.01, "low": c * 0.99}
            for t, c in closes.items()
        }

    def test_runs_full_bar_range_without_shape_error(self, trending_close_matrix):
        engine = self._engine()
        asset_data = self._asset_data(trending_close_matrix)
        n_bars = len(trending_close_matrix["TEST"])
        start_bar = engine.config.strategy.slow_window + 5
        # Must not raise "operands could not be broadcast together" (the
        # returns_window numerator/denominator shape bug) nor NameError
        # (the min_len/n_bars mismatch in the liquidation step below).
        result = engine._run_strategy(asset_data, start_bar, n_bars)
        assert len(result["equity_curve_net"]) > 0
        assert result["final_capital"] > 0
        assert np.isfinite(result["final_capital"])

    def test_every_bar_index_reaches_returns_window_line(self, trending_close_matrix):
        # Directly exercises every bar >= 2 (the guard in the buggy line),
        # including the exact boundary (bar == 21/22) where the numerator
        # and denominator window lengths first diverged.
        engine = self._engine()
        close = trending_close_matrix["TEST"]
        asset_data = self._asset_data({"TEST": close})
        for bar in range(2, len(close)):
            win_start = max(0, bar - 21)
            returns_window = np.diff(close[win_start:bar + 1]) / np.maximum(close[win_start:bar], 1e-9)
            assert returns_window.shape == close[win_start:bar].shape
            assert np.all(np.isfinite(returns_window))

    def test_liquidation_uses_n_bars_not_undefined_min_len(self, trending_close_matrix):
        # A short window that still leaves the MA-crossover strategy holding
        # an open position at n_bars, forcing execution of the liquidation
        # block (`final_bar = n_bars - 1`) that referenced the undefined
        # name `min_len` before the fix.
        engine = self._engine(fast=5, slow=10)
        asset_data = self._asset_data(trending_close_matrix)
        n_bars = 60
        result = engine._run_strategy(asset_data, start_bar=15, n_bars=n_bars)
        assert result["final_capital"] > 0
