"""Tests for the Walk-Forward Validation Engine (backend/services/walk_forward.py).

These exercise the offline, non-network code paths directly (WalkForwardEngine.run()
requires a live yfinance download and is out of scope for unit tests). They cover
_evaluate_period() and _quick_eval(), which previously crashed with a numpy shape
mismatch (21 vs 20 elements) inside the per-bar volatility calculation whenever a
buy or sell signal fired — a bug that had no test coverage because it only lives on
the network-dependent execution path.
"""
import numpy as np
import pytest

from backend.services.walk_forward import (
    WalkForwardEngine, WalkForwardConfig, _windowed_vol,
)
from backend.services.backtest_service import StrategyConfig
from backend.services.execution_model import retail_config


@pytest.fixture
def trending_close():
    """A price series with clear up/down swings so MA crossovers actually fire."""
    rng = np.random.default_rng(3)
    n = 300
    trend = np.concatenate([
        np.linspace(0, 20, n // 3),
        np.linspace(20, -10, n // 3),
        np.linspace(-10, 15, n - 2 * (n // 3)),
    ])
    noise = rng.normal(0, 0.5, n)
    return 100.0 + trend + noise


class TestWindowedVol:
    def test_matches_length_at_every_offset(self, trending_close):
        # Must never raise, for any bar index into the series.
        for i in range(0, len(trending_close)):
            v = _windowed_vol(trending_close, i)
            assert v >= 0.0
            assert np.isfinite(v)

    def test_small_i_returns_default(self, trending_close):
        assert _windowed_vol(trending_close, 1) == 0.02
        assert _windowed_vol(trending_close, 2) == 0.02

    def test_reasonable_magnitude(self, trending_close):
        v = _windowed_vol(trending_close, 100)
        # Daily vol on a ~100-level price series with small noise should be small.
        assert 0 < v < 0.5


class TestEvaluatePeriod:
    def test_runs_without_shape_error(self, trending_close):
        cfg = WalkForwardConfig(
            assets=["TEST"], strategy=StrategyConfig(fast_window=10, slow_window=40),
            execution=retail_config(),
        )
        engine = WalkForwardEngine(cfg)
        close_data = {"TEST": trending_close}
        result = engine._evaluate_period(
            close_data, ["TEST"], start=0, end=len(trending_close),
            fast_w=10, slow_w=40, cfg=cfg,
        )
        assert "sharpe" in result
        assert "equity" in result
        assert len(result["equity"]) > 1
        # A trending series with a 10/40 crossover should trigger at least one trade.
        assert result["n_trades"] >= 1

    def test_quick_eval_runs_without_shape_error(self, trending_close):
        cfg = WalkForwardConfig(assets=["TEST"])
        engine = WalkForwardEngine(cfg)
        close_data = {"TEST": trending_close}
        sharpe = engine._quick_eval(close_data, ["TEST"], 0, len(trending_close), 10, 40)
        assert np.isfinite(sharpe)
