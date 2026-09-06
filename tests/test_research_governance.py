"""Regression coverage for reproducibility and implementation-cost controls."""
import numpy as np
import pytest

from backend.services.event_simulator import FillEvent, Portfolio
from backend.services.portfolio_optimization import (
    PortfolioConstraints, equal_weight, minimum_variance, turnover_aware_mean_variance,
)
from backend.services.research_metadata import DatasetMetadata, ExperimentRecord, data_hash


def test_fill_implementation_shortfall_is_not_charged_twice():
    portfolio = Portfolio(initial_capital=10_000)
    # $1/share implementation shortfall is already reflected in the $101 fill.
    portfolio.process_fill(FillEvent(ticker="ABC", quantity=10, fill_price=101,
                                     commission=2, slippage=10))
    assert portfolio.cash == pytest.approx(8_988)
    assert portfolio.total_slippage == pytest.approx(10)


def test_turnover_penalty_keeps_portfolio_nearer_prior_weights():
    mu = np.array([0.002, 0.0001])
    cov = np.array([[0.0004, 0.0], [0.0, 0.0004]])
    previous = np.array([0.5, 0.5])
    no_penalty = turnover_aware_mean_variance(mu, cov, previous,
                                               turnover_penalty=0.0)
    penalised = turnover_aware_mean_variance(mu, cov, previous,
                                             turnover_penalty=0.01)
    assert np.abs(penalised - previous).sum() < np.abs(no_penalty - previous).sum()
    assert penalised.sum() == pytest.approx(1.0)


def test_portfolio_box_constraints_are_preserved_after_projection():
    cov = np.array([[0.02, 0.019, 0.0], [0.019, 0.02, 0.0], [0.0, 0.0, 0.05]])
    constraints = PortfolioConstraints(long_only=True, min_weight=0.1, max_weight=0.6)
    weights = minimum_variance(cov, constraints)
    equal = equal_weight(3, constraints)
    for candidate in (weights, equal):
        assert candidate.sum() == pytest.approx(1.0)
        assert np.all(candidate >= 0.1 - 1e-9)
        assert np.all(candidate <= 0.6 + 1e-9)


def test_experiment_record_includes_lineage_hash_and_bias_warnings():
    metadata = DatasetMetadata.yahoo_daily()
    record = ExperimentRecord.create(
        dataset=metadata, assets=["AAPL", "MSFT"], date_range="2024-01-01:2024-12-31",
        strategy="momentum_v1", strategy_parameters={"lookback": 20},
        execution_model="square_root_impact", random_seed=42,
    )
    payload = record.to_dict()
    assert payload["experiment_id"].startswith("QS-")
    assert len(payload["data_hash"]) == 64
    assert payload["dataset"]["warnings"]
    assert data_hash({"a": 1, "b": 2}) == data_hash({"b": 2, "a": 1})
