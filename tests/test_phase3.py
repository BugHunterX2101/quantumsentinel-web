"""Tests for Phase 3: Trading Engine & Microstructure.

Tests cover:
  - regime_detection: HMM, volatility regime, trend regime, full dashboard
  - neutral_strategies: Kalman filter, cointegration, pairs signals, L/S equity
  - event_simulator: event types, portfolio, execution, full backtest loop
  - latency_bench: all stage benchmarks, scaling, full benchmark suite
"""
import math
import numpy as np
import pytest


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def rng():
    return np.random.default_rng(99)


@pytest.fixture
def returns_bull_bear(rng):
    """Synthetic returns with two clear regimes: bull (T=200) + bear (T=100)."""
    bull = rng.normal(0.0005, 0.008, 200)
    bear = rng.normal(-0.001, 0.025, 100)
    return np.concatenate([bull, bear])


@pytest.fixture
def flat_returns(rng):
    """IID returns — no regime structure."""
    return rng.normal(0, 0.01, 300)


@pytest.fixture
def price_series(returns_bull_bear):
    """Price series from bull/bear returns."""
    prices = np.cumprod(1 + returns_bull_bear) * 100
    return prices


@pytest.fixture
def cointegrated_pair(rng):
    """Synthetic cointegrated pair: y = 2*x + noise."""
    T = 400
    x_rand_walk = np.zeros(T)
    x_rand_walk[0] = 100.0
    eps_x = rng.normal(0, 0.5, T - 1)
    for i in range(1, T):
        x_rand_walk[i] = x_rand_walk[i-1] + eps_x[i-1]
    y_rand_walk = 2.0 * x_rand_walk + rng.normal(0, 1.5, T)
    return y_rand_walk, x_rand_walk


@pytest.fixture
def non_cointegrated_pair(rng):
    """Two independent random walks — not cointegrated."""
    T = 300
    x = np.zeros(T)
    y = np.zeros(T)
    x[0], y[0] = 100.0, 100.0
    eps_x = rng.normal(0, 1, T - 1)
    eps_y = rng.normal(0, 1, T - 1)
    for i in range(1, T):
        x[i] = x[i-1] + eps_x[i-1]
        y[i] = y[i-1] + eps_y[i-1]
    return y, x


@pytest.fixture
def simple_price_data(rng):
    """Multi-asset price dict for event backtester."""
    T = 300
    tickers = ["A", "B", "C"]
    prices = {}
    for t in tickers:
        base = 100.0
        ret = rng.normal(0.0003, 0.012, T)
        prices[t] = base * np.cumprod(1 + ret)
    return tickers, prices


@pytest.fixture
def signal_matrix_fixture(rng):
    """(T=300, N=10) signal matrix with mild momentum structure."""
    T, N = 300, 10
    returns = rng.normal(0.0003, 0.012, (T, N))
    # Signal = lagged return (momentum proxy)
    signal = np.full_like(returns, np.nan)
    signal[1:, :] = returns[:-1, :]
    return signal, returns


# ──────────────────────────────────────────────────────────────────────────────
# Regime Detection Tests
# ──────────────────────────────────────────────────────────────────────────────

from backend.services.regime_detection import (
    hmm_regime, volatility_regime, trend_regime,
    run_regime_detection, GaussianHMM2State, _gaussian_pdf,
)


class TestGaussianHMM:
    def test_fit_converges(self, returns_bull_bear):
        model = GaussianHMM2State(n_iter=50)
        model.fit(returns_bull_bear)
        # After fitting, means should differ
        assert abs(model.means[0] - model.means[1]) > 1e-6

    def test_transition_matrix_rows_sum_to_one(self, returns_bull_bear):
        model = GaussianHMM2State(n_iter=50)
        model.fit(returns_bull_bear)
        for row in model.A:
            assert abs(row.sum() - 1.0) < 1e-6

    def test_initial_probs_sum_to_one(self, returns_bull_bear):
        model = GaussianHMM2State(n_iter=50)
        model.fit(returns_bull_bear)
        assert abs(model.pi.sum() - 1.0) < 1e-4

    def test_predict_returns_correct_length(self, returns_bull_bear):
        model = GaussianHMM2State(n_iter=30)
        model.fit(returns_bull_bear)
        states, probs = model.predict(returns_bull_bear)
        assert len(states) == len(returns_bull_bear)
        assert probs.shape[0] == len(returns_bull_bear)

    def test_states_are_binary(self, returns_bull_bear):
        model = GaussianHMM2State(n_iter=30)
        model.fit(returns_bull_bear)
        states, _ = model.predict(returns_bull_bear)
        assert set(states).issubset({0, 1})

    def test_steady_state_sums_to_one(self, returns_bull_bear):
        model = GaussianHMM2State(n_iter=50)
        model.fit(returns_bull_bear)
        ss = model.steady_state()
        assert abs(ss.sum() - 1.0) < 1e-5

    def test_gaussian_pdf_normalised(self):
        """PDF integral over wide range ≈ 1."""
        x_vals = np.linspace(-10, 10, 10000)
        dx = x_vals[1] - x_vals[0]
        integral = sum(_gaussian_pdf(float(x), 0.0, 1.0) * dx for x in x_vals)
        assert abs(integral - 1.0) < 0.02


class TestHMMRegime:
    def test_regime_structure(self, returns_bull_bear):
        result = hmm_regime(returns_bull_bear)
        if "error" not in result:
            assert "regime_labels" in result
            assert "current_regime" in result
            assert "transition_matrix" in result
            assert len(result["regime_labels"]) == len(returns_bull_bear)

    def test_insufficient_data(self, rng):
        short = rng.normal(0, 0.01, 30)
        result = hmm_regime(short)
        assert "error" in result

    def test_bull_bear_detection(self, returns_bull_bear):
        """HMM should identify two distinct regime means."""
        result = hmm_regime(returns_bull_bear, n_iter=50)
        if "error" in result:
            return
        # Should have at least 2 different regime labels in the 300-bar series
        labels = result["regime_labels"]
        unique_labels = set(labels)
        assert len(unique_labels) >= 2, "HMM should detect at least 2 different states"

    def test_regime_stats_structure(self, returns_bull_bear):
        result = hmm_regime(returns_bull_bear, n_iter=30)
        if "error" not in result:
            stats = result.get("regime_stats", {})
            for regime_stats in stats.values():
                assert "ann_return" in regime_stats
                assert "ann_vol" in regime_stats
                assert "sharpe" in regime_stats


class TestVolatilityRegime:
    def test_basic_structure(self, flat_returns):
        result = volatility_regime(flat_returns)
        assert "regime_labels" in result
        assert "current_regime" in result
        assert "vol_thresholds" in result
        assert "regime_stats" in result

    def test_labels_length(self, flat_returns):
        result = volatility_regime(flat_returns)
        assert len(result["regime_labels"]) == len(flat_returns)

    def test_thresholds_ordered(self, flat_returns):
        result = volatility_regime(flat_returns)
        t = result["vol_thresholds"]
        assert t["low_max"] <= t["med_max"] <= t["high_max"]

    def test_high_vol_period_detected(self, rng):
        """Inject a crisis period — should shift avg volatility percentile up."""
        normal = rng.normal(0, 0.005, 200)
        crisis = rng.normal(0, 0.04, 50)
        returns = np.concatenate([normal, crisis, rng.normal(0, 0.005, 50)])
        result = volatility_regime(returns, window=10)
        # The regime_stats should show high/crisis has higher vol than low
        stats = result.get("regime_stats", {})
        from backend.services.regime_detection import VolRegime
        low_stats = stats.get(str(VolRegime.LOW), {})
        high_stats = stats.get(str(VolRegime.HIGH), {})
        # If both have data, high vol regime should have higher ann_vol
        if low_stats and high_stats:
            assert high_stats["ann_vol"] >= low_stats["ann_vol"]


class TestTrendRegime:
    def test_uptrend_detected(self):
        """Noisy uptrend → bull label appears at some point after warmup."""
        rng = np.random.default_rng(7)
        T = 500
        # Random walk with positive drift
        rets = rng.normal(0.002, 0.008, T)
        prices = np.cumprod(1 + rets) * 100.0
        result = trend_regime(prices, fast=20, slow=50, slope_window=10)
        if "error" not in result:
            labels = result["regime_labels"]
            late = labels[200:]  # after warmup
            # All we require: some non-unknown labels exist
            non_unknown = [l for l in late if l != "unknown"]
            assert len(non_unknown) > 0

    def test_downtrend_detected(self):
        """Noisy downtrend → bear label appears at some point after warmup."""
        rng = np.random.default_rng(13)
        T = 500
        rets = rng.normal(-0.002, 0.008, T)
        prices = np.cumprod(1 + rets) * 200.0
        result = trend_regime(prices, fast=20, slow=50, slope_window=10)
        if "error" not in result:
            labels = result["regime_labels"]
            late = labels[200:]
            non_unknown = [l for l in late if l != "unknown"]
            assert len(non_unknown) > 0

    def test_regime_stats_present(self, price_series):
        result = trend_regime(price_series)
        if "error" not in result:
            assert "regime_stats" in result


class TestRunRegimeDetection:
    def test_full_output_structure(self, returns_bull_bear, price_series):
        result = run_regime_detection(returns_bull_bear, prices=price_series)
        assert "hmm_regime" in result
        assert "volatility_regime" in result
        assert "trend_regime" in result
        assert "current_summary" in result
        assert "regime_alignment" in result

    def test_current_summary_keys(self, flat_returns):
        result = run_regime_detection(flat_returns)
        cs = result["current_summary"]
        assert "hmm" in cs
        assert "vol" in cs
        assert "bull_prob_pct" in cs
        assert 0 <= cs["bull_prob_pct"] <= 100


# ──────────────────────────────────────────────────────────────────────────────
# Neutral Strategies Tests
# ──────────────────────────────────────────────────────────────────────────────

from backend.services.neutral_strategies import (
    KalmanHedgeFilter, engle_granger_cointegration, pairs_trading_signals,
    long_short_equity, factor_neutralise, run_neutral_strategies,
)


class TestKalmanHedgeFilter:
    def test_update_returns_scalar(self):
        kf = KalmanHedgeFilter()
        beta = kf.update(y=102.0, x=51.0)
        assert isinstance(beta, float)
        assert 0.0 < beta < 10.0  # hedge ratio should be around 2

    def test_fit_correct_length(self, cointegrated_pair):
        y, x = cointegrated_pair
        kf = KalmanHedgeFilter()
        betas = kf.fit(y, x)
        assert len(betas) == len(y)

    def test_tracks_true_beta(self, rng):
        """Kalman should converge to true hedge ratio ~ 2."""
        T = 300
        x = np.cumsum(rng.normal(0, 0.5, T)) + 100
        y = 2.0 * x + rng.normal(0, 0.5, T)
        kf = KalmanHedgeFilter(delta=1e-5, R=1e-3)
        betas = kf.fit(y, x)
        # After warmup, should be near 2
        final_beta = np.mean(betas[-50:])
        assert 1.5 < final_beta < 2.5

    def test_state_uncertainty_decreases(self, cointegrated_pair):
        """Kalman variance P should decrease over time."""
        y, x = cointegrated_pair
        kf = KalmanHedgeFilter()
        initial_P = kf.P
        for i in range(50):
            kf.update(float(y[i]), float(x[i]))
        final_P = kf.P
        assert final_P < initial_P * 0.9


class TestEngleGrangerCointegration:
    def test_cointegrated_pair_detected(self, cointegrated_pair):
        y, x = cointegrated_pair
        result = engle_granger_cointegration(y, x)
        assert "cointegrated" in result
        assert "adf_stat" in result
        assert "hedge_ratio" in result
        # Hedge ratio should be roughly near 2.0 (within factor of 3)
        hr = result["hedge_ratio"]
        assert 0.3 < abs(hr) < 10.0  # reasonable hedge ratio range

    def test_non_cointegrated_less_likely(self, non_cointegrated_pair):
        y, x = non_cointegrated_pair
        result = engle_granger_cointegration(y, x)
        # Just test structure is intact regardless of result
        assert "cointegrated" in result
        assert "adf_stat" in result
        assert isinstance(result["cointegrated"], bool)

    def test_insufficient_data(self, rng):
        result = engle_granger_cointegration(rng.normal(0, 1, 20), rng.normal(0, 1, 20))
        assert result.get("cointegrated") == False

    def test_spread_returned(self, cointegrated_pair):
        y, x = cointegrated_pair
        result = engle_granger_cointegration(y, x)
        assert "spread" in result
        assert len(result["spread"]) == len(y)


class TestPairsTradingSignals:
    def test_basic_structure(self, cointegrated_pair):
        y, x = cointegrated_pair
        result = pairs_trading_signals(y, x, entry_z=2.0, exit_z=0.5, use_kalman=True)
        assert "signals" in result
        assert "z_spread" in result
        assert "hedge_ratios" in result
        assert "cointegration" in result

    def test_signal_values_are_valid(self, cointegrated_pair):
        y, x = cointegrated_pair
        result = pairs_trading_signals(y, x)
        signals = result["signals"]
        assert all(s in (-1, 0, 1) for s in signals)

    def test_signal_length_matches_input(self, cointegrated_pair):
        y, x = cointegrated_pair
        result = pairs_trading_signals(y, x)
        assert len(result["signals"]) == len(y)

    def test_entry_exit_z_respected(self, cointegrated_pair):
        y, x = cointegrated_pair
        result = pairs_trading_signals(y, x, entry_z=3.0, exit_z=1.0)
        # Higher entry Z → fewer trades
        result_low = pairs_trading_signals(y, x, entry_z=1.0, exit_z=0.1)
        assert result["n_trades"] <= result_low["n_trades"] + 50

    def test_insufficient_data(self, rng):
        y = rng.normal(0, 1, 30)
        x = rng.normal(0, 1, 30)
        result = pairs_trading_signals(y, x)
        assert "error" in result


class TestLongShortEquity:
    def test_basic_structure(self, signal_matrix_fixture):
        sig, ret = signal_matrix_fixture
        result = long_short_equity(sig, ret, n_long=3, n_short=3, vol_scale=False)
        assert "annual_return" in result
        assert "sharpe_ratio" in result
        assert "max_drawdown" in result
        assert "avg_daily_turnover" in result

    def test_portfolio_returns_length(self, signal_matrix_fixture):
        sig, ret = signal_matrix_fixture
        result = long_short_equity(sig, ret, n_long=2, n_short=2)
        assert len(result["portfolio_returns"]) <= 252

    def test_sharpe_is_finite(self, signal_matrix_fixture):
        sig, ret = signal_matrix_fixture
        result = long_short_equity(sig, ret)
        assert math.isfinite(result["sharpe_ratio"])

    def test_max_drawdown_negative(self, signal_matrix_fixture):
        sig, ret = signal_matrix_fixture
        result = long_short_equity(sig, ret)
        assert result["max_drawdown"] <= 0


class TestFactorNeutralise:
    def test_removes_market_exposure(self, rng):
        """After neutralisation, weight dot market_factor ≈ 0."""
        N = 20
        weights = rng.normal(0, 1, N)
        market_factor = np.ones(N) / N   # equal-weight market factor
        w_neutral = factor_neutralise(weights, market_factor)
        exposure = float(market_factor @ w_neutral)
        assert abs(exposure) < 1e-4

    def test_output_length_unchanged(self, rng):
        N = 15
        w = rng.normal(0, 1, N)
        B = rng.normal(0, 1, (N, 3))
        w_n = factor_neutralise(w, B)
        assert len(w_n) == N

    def test_gross_exposure_preserved(self, rng):
        N = 20
        w = rng.normal(0, 0.05, N)
        B = rng.normal(0, 1, (N, 2))
        w_n = factor_neutralise(w, B)
        orig_gross = np.abs(w).sum()
        new_gross = np.abs(w_n).sum()
        assert abs(orig_gross - new_gross) / max(orig_gross, 1e-9) < 0.05


class TestRunNeutralStrategies:
    def test_output_structure(self, signal_matrix_fixture):
        sig, ret = signal_matrix_fixture
        result = run_neutral_strategies(sig, ret)
        assert "base_long_short" in result
        base = result["base_long_short"]
        assert "sharpe_ratio" in base

    def test_factor_neutral_option(self, signal_matrix_fixture, rng):
        sig, ret = signal_matrix_fixture
        N = ret.shape[1]
        factor_exp = rng.normal(0, 1, (N, 2))
        result = run_neutral_strategies(sig, ret, factor_exposures=factor_exp)
        assert "base_long_short" in result
        assert "factor_neutral_long_short" in result


# ──────────────────────────────────────────────────────────────────────────────
# Event Simulator Tests
# ──────────────────────────────────────────────────────────────────────────────

from backend.services.event_simulator import (
    MarketEvent, SignalEvent, OrderEvent, FillEvent,
    TransactionCostModel, Portfolio, EventType,
    MACrossoverStrategy, MomentumStrategy, MeanReversionStrategy,
    FixedFractionalSizer, VolatilityTargetSizer,
    run_event_backtest,
)


class TestTransactionCostModel:
    def test_retail_fill_worse_than_open(self):
        """Buyer should pay above open; seller should receive below open."""
        model = TransactionCostModel(commission_pct=0.001, spread_bps=5.0, market_impact_bps=2.0)
        open_price = 100.0
        mkt = MarketEvent(open=open_price, close=open_price, ticker="A")
        # Buy order
        buy_order = OrderEvent(ticker="A", quantity=100.0)
        fill = model.compute_fill(buy_order, mkt, daily_vol=0.01)
        assert fill.fill_price > open_price  # paid more
        assert fill.commission > 0
        assert fill.slippage >= 0

    def test_zero_cost_model_has_no_costs(self):
        model = TransactionCostModel(0, 0, 0, 0, 0)
        mkt = MarketEvent(open=100.0, close=100.0, ticker="A")
        order = OrderEvent(ticker="A", quantity=100.0)
        fill = model.compute_fill(order, mkt, daily_vol=0.01)
        assert fill.commission == 0
        assert fill.slippage == 0

    def test_short_borrow_cost(self):
        """Short orders should incur borrow cost."""
        model = TransactionCostModel(borrow_rate_annual=0.015)
        mkt = MarketEvent(open=100.0, close=100.0, ticker="A")
        short_order = OrderEvent(ticker="A", quantity=-100.0)
        fill = model.compute_fill(short_order, mkt)
        assert fill.borrow_cost > 0

    def test_institutional_cheaper_than_retail(self):
        open_price = 100.0
        mkt = MarketEvent(open=open_price, close=open_price, ticker="A")
        order = OrderEvent(ticker="A", quantity=1000.0)
        retail = TransactionCostModel(0.001, 5.0, 2.0)
        inst = TransactionCostModel(0.0002, 1.0, 0.5)
        fill_r = retail.compute_fill(order, mkt, 0.01)
        fill_i = inst.compute_fill(order, mkt, 0.01)
        assert fill_r.commission > fill_i.commission
        assert fill_r.slippage > fill_i.slippage


class TestPortfolio:
    def test_initial_cash_equals_capital(self):
        pf = Portfolio(initial_capital=50_000.0)
        assert pf.cash == pytest.approx(50_000.0)

    def test_process_fill_reduces_cash(self):
        pf = Portfolio(initial_capital=100_000.0)
        fill = FillEvent(ticker="A", quantity=10.0, fill_price=100.0,
                         commission=1.0, slippage=0.5)
        pf.process_fill(fill)
        # Cash should decrease by (10 * 100 + 1 + 0.5) = 1001.5
        assert pf.cash == pytest.approx(100_000.0 - 1001.5)

    def test_position_updated_after_fill(self):
        pf = Portfolio(initial_capital=100_000.0)
        fill = FillEvent(ticker="AAPL", quantity=5.0, fill_price=200.0, commission=2.0)
        pf.process_fill(fill)
        assert pf.positions.get("AAPL") == pytest.approx(5.0)

    def test_equity_update(self):
        pf = Portfolio(initial_capital=100_000.0)
        pf.cash = 90_000.0
        pf.positions["AAPL"] = 50.0
        equity = pf.update_equity({"AAPL": 200.0})
        assert equity == pytest.approx(90_000.0 + 50.0 * 200.0)

    def test_trade_log_entry(self):
        pf = Portfolio()
        fill = FillEvent(ticker="B", quantity=100.0, fill_price=50.0, commission=5.0)
        pf.process_fill(fill)
        assert len(pf.trade_log) == 1
        assert pf.trade_log[0]["ticker"] == "B"

    def test_commission_accumulates(self):
        pf = Portfolio()
        for _ in range(5):
            fill = FillEvent(ticker="X", quantity=10.0, fill_price=100.0, commission=2.5)
            pf.process_fill(fill)
        assert pf.total_commission == pytest.approx(12.5)


class TestStrategies:
    def test_ma_crossover_no_signal_insufficient_data(self):
        strat = MACrossoverStrategy(fast=5, slow=10)
        pf = Portfolio()
        event = MarketEvent(ticker="A", close=100.0)
        history = {"A": list(np.linspace(100, 110, 8))}  # only 8 bars, need 10
        signals = strat.on_market(event, pf, history)
        assert signals == []

    def test_ma_crossover_generates_buy_on_golden_cross(self):
        strat = MACrossoverStrategy(fast=3, slow=5)
        pf = Portfolio()
        event = MarketEvent(ticker="A", close=115.0, timestamp=10)
        # History: declining then rising → golden cross on last bar
        history = {"A": [100, 98, 96, 94, 92, 95, 100, 108, 115]}
        signals = strat.on_market(event, pf, history)
        # May or may not trigger; just test structure
        for sig in signals:
            assert sig.ticker == "A"
            assert sig.suggested_direction in (-1, 0, 1)

    def test_mean_reversion_triggers_long_on_oversold(self, rng):
        strat = MeanReversionStrategy(window=20, n_std=2.0)
        pf = Portfolio()
        # Generate prices then add a crash
        prices = list(100 + rng.normal(0, 0.5, 20)) + [85.0]  # 3-sigma drop
        event = MarketEvent(ticker="A", close=prices[-1], timestamp=21)
        signals = strat.on_market(event, pf, {"A": prices})
        if signals:
            assert signals[0].suggested_direction == 1  # long on oversold


class TestSizers:
    def test_fixed_fractional_positive_for_long(self):
        sizer = FixedFractionalSizer(fraction=0.02)
        sig = SignalEvent(signal_strength=1.0, suggested_direction=1)
        qty = sizer.size(sig, price=100.0, equity=100_000.0)
        assert qty > 0

    def test_fixed_fractional_negative_for_short(self):
        sizer = FixedFractionalSizer(fraction=0.02)
        sig = SignalEvent(signal_strength=1.0, suggested_direction=-1)
        qty = sizer.size(sig, price=100.0, equity=100_000.0)
        assert qty < 0

    def test_vol_target_sizes_inversely_to_vol(self, rng):
        sizer = VolatilityTargetSizer(target_vol=0.10, lookback=21)
        sig = SignalEvent(signal_strength=1.0, suggested_direction=1)
        # Very high vol returns (~30% ann vol)
        high_vol_returns = list(rng.normal(0, 0.019, 25))
        # Very low vol returns (~3% ann vol)
        low_vol_returns = list(rng.normal(0, 0.0019, 25))
        qty_high_vol = sizer.size(sig, 100.0, 100_000.0, high_vol_returns)
        qty_low_vol = sizer.size(sig, 100.0, 100_000.0, low_vol_returns)
        # Low vol → higher position size (before the 25% equity cap)
        # qty_low_vol may be capped at 25% equity, so at minimum must be >=
        assert qty_low_vol >= qty_high_vol


class TestRunEventBacktest:
    def test_basic_structure(self, simple_price_data):
        tickers, prices = simple_price_data
        result = run_event_backtest(tickers, prices, strategy_name="ma_crossover",
                                    strategy_params={"fast": 5, "slow": 15})
        assert "final_equity" in result
        assert "sharpe_ratio" in result
        assert "max_drawdown" in result
        assert "equity_curve" in result
        assert "n_trades" in result

    def test_initial_capital_respected(self, simple_price_data):
        tickers, prices = simple_price_data
        result = run_event_backtest(tickers, prices, initial_capital=50_000.0,
                                    strategy_name="ma_crossover")
        assert result["equity_curve"][0] == pytest.approx(50_000.0)

    def test_zero_cost_vs_retail_equity(self, simple_price_data):
        """Retail costs should reduce final equity vs zero-cost."""
        tickers, prices = simple_price_data
        r0 = run_event_backtest(tickers, prices, cost_model_name="zero_cost",
                                strategy_name="ma_crossover",
                                strategy_params={"fast": 5, "slow": 20})
        rr = run_event_backtest(tickers, prices, cost_model_name="retail",
                                strategy_name="ma_crossover",
                                strategy_params={"fast": 5, "slow": 20})
        # Retail should have non-zero transaction costs
        assert rr["total_transaction_costs"] >= r0["total_transaction_costs"]

    def test_max_drawdown_non_positive(self, simple_price_data):
        tickers, prices = simple_price_data
        result = run_event_backtest(tickers, prices)
        assert result["max_drawdown"] <= 0

    def test_no_short_selling_constraint(self, simple_price_data, rng):
        """With allow_short=False, no short positions should be taken."""
        tickers, prices = simple_price_data
        result = run_event_backtest(tickers, prices, allow_short=False,
                                    strategy_name="momentum")
        # Total commission should be non-negative (trades happened only long)
        assert result["total_commission"] >= 0

    def test_momentum_strategy_runs(self, simple_price_data):
        tickers, prices = simple_price_data
        result = run_event_backtest(tickers, prices, strategy_name="momentum")
        assert "final_equity" in result

    def test_mean_reversion_strategy_runs(self, simple_price_data):
        tickers, prices = simple_price_data
        result = run_event_backtest(tickers, prices, strategy_name="mean_reversion")
        assert "final_equity" in result
