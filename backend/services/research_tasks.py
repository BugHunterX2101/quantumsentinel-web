"""QuantumSentinel — research computations run by the job worker.

Each task takes the validated request parameters and returns a
``TaskOutput``: the result shown to the user, the metadata for its audit
event, and the research trials it evaluated. Tasks never touch the database;
the worker process records trials, persists rows and writes the audit event
after a task succeeds (see research_jobs.complete).

A task reports a problem with its input or data by raising ``TaskError``
with the HTTP-style status the synchronous endpoint used to return; any other
exception becomes a generic failure, so internal details never reach users.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .. import schemas
from . import research_trials

log = logging.getLogger(__name__)


class TaskError(Exception):
    """A user-facing failure: ``status_code`` and a safe ``detail`` message."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass
class TrialRecord:
    family: str
    configs: list[dict]
    source: str


@dataclass
class TaskOutput:
    result: dict
    audit: dict
    trials: TrialRecord | None = None


# --------------------------------------------------------------------------
# Market data
# --------------------------------------------------------------------------

def fetch_return_matrix(assets: list[str], period: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Fetch and align multi-asset return and price matrices.

    Returns (return_matrix T×N, price_matrix T×N, valid_asset_names).
    """
    import yfinance as yf
    import pandas as pd

    data = yf.download(assets, period=period, interval="1d",
                       progress=False, auto_adjust=True)
    if data is None or data.empty:
        raise ValueError("Failed to download market data")

    close_frames = {}
    for ticker in assets:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                s = data["Close"][ticker].dropna()
            else:
                s = data["Close"].dropna()
            if len(s) > 50:
                close_frames[ticker] = s
        except (KeyError, TypeError):
            continue

    if len(close_frames) < 4:
        raise ValueError(f"Only {len(close_frames)} assets had sufficient data")

    # Align on common index
    df = pd.DataFrame(close_frames).dropna()
    if len(df) < 60:
        raise ValueError(f"Only {len(df)} common trading days — need ≥ 60")

    price_matrix = df.to_numpy(dtype=float)
    return_matrix = np.diff(price_matrix, axis=0) / np.maximum(price_matrix[:-1], 1e-9)
    valid_names = list(df.columns)

    return return_matrix, price_matrix[1:], valid_names


def fetch_single_asset(asset: str, period: str) -> tuple[np.ndarray, np.ndarray]:
    """Fetch returns and prices for a single asset."""
    import yfinance as yf
    data = yf.download(asset, period=period, interval="1d",
                       progress=False, auto_adjust=True)
    if data is None or data.empty:
        raise ValueError(f"Could not fetch data for {asset}")
    close = data["Close"].dropna()
    if len(close) < 60:
        raise ValueError(f"Only {len(close)} trading days for {asset}")
    # For a single-ticker yf.download() call, data["Close"] is a 1-column
    # DataFrame in this yfinance version, not a Series — to_numpy() on it
    # yields shape (T, 1). np.diff() on that diffs along the trivial size-1
    # axis instead of along time, producing a shape that can never broadcast
    # against prices[:-1]. reshape(-1) is a no-op for an already-1-D Series
    # and flattens a (T, 1) DataFrame correctly.
    prices = close.to_numpy(dtype=float).reshape(-1)
    returns = np.diff(prices) / np.maximum(prices[:-1], 1e-9)
    return returns, prices


# --------------------------------------------------------------------------
# Backtests and validation
# --------------------------------------------------------------------------

def ma_backtest(params: dict) -> TaskOutput:
    """The dashboard's moving-average backtest (persisted as a Backtest row)."""
    from . import backtest_service
    req = schemas.BacktestRequest.model_validate(params)
    try:
        result = backtest_service.run_moving_average_backtest(
            req.asset, req.fast_window, req.slow_window, req.period
        )
    except ValueError as exc:
        raise TaskError(422, str(exc)) from exc
    trials = TrialRecord(
        research_trials.family_hash("ma_crossover", [req.asset]),
        [{"fast_window": req.fast_window, "slow_window": req.slow_window,
          "period": req.period, "execution_preset": "retail"}],
        "backtest")
    audit = {"asset": req.asset, "period": req.period, "total_return": result["total_return"]}
    return TaskOutput(result, audit, trials)


def advanced_backtest(params: dict) -> TaskOutput:
    """Advanced backtester with realistic execution costs, multiple strategies,
    and comprehensive risk metrics."""
    from .backtest_service import (
        BacktestConfig, BacktestEngine, StrategyConfig, StrategyType,
    )
    from .execution_model import (
        zero_cost_config, retail_config, institutional_config,
        PositionSizer, SizingMethod,
    )
    req = schemas.AdvancedBacktestRequest.model_validate(params)

    exec_map = {"zero_cost": zero_cost_config, "retail": retail_config,
                "institutional": institutional_config}
    exec_config = exec_map.get(req.execution_preset, retail_config)()
    exec_config.allow_short_selling = req.allow_short_selling
    exec_config.leverage_limit = req.max_leverage

    sizing_map = {
        "fixed_fractional": SizingMethod.FIXED_FRACTIONAL,
        "volatility_target": SizingMethod.VOLATILITY_TARGET,
        "kelly": SizingMethod.KELLY,
        "equal_weight": SizingMethod.EQUAL_WEIGHT,
    }
    exec_config.sizer = PositionSizer(
        method=sizing_map.get(req.sizing_method, SizingMethod.FIXED_FRACTIONAL),
        risk_per_trade=req.risk_per_trade,
        max_position_pct=req.max_position_pct,
        max_leverage=req.max_leverage,
    )

    strategy_map = {
        "ma_crossover": StrategyType.MA_CROSSOVER,
        "sba_signal": StrategyType.SBA_SIGNAL,
        "momentum": StrategyType.MOMENTUM,
        "mean_reversion": StrategyType.MEAN_REVERSION,
    }

    config = BacktestConfig(
        assets=req.assets,
        period=req.period,
        initial_capital=req.initial_capital,
        strategy=StrategyConfig(
            strategy_type=strategy_map.get(req.strategy_type, StrategyType.MA_CROSSOVER),
            fast_window=req.fast_window,
            slow_window=req.slow_window,
        ),
        execution=exec_config,
        benchmark=req.benchmark,
    )

    try:
        engine = BacktestEngine(config)
        result = engine.run()
    except ValueError as exc:
        raise TaskError(422, str(exc)) from exc
    except Exception as exc:
        log.exception("Advanced backtest failed")
        raise TaskError(500, "Backtest failed") from exc

    trials = TrialRecord(research_trials.family_hash(config.strategy.strategy_type, req.assets),
                         [req.model_dump()], "backtest")
    audit = {"assets": req.assets, "strategy": req.strategy_type, "period": req.period}
    return TaskOutput(result, audit, trials)


def walk_forward(params: dict) -> TaskOutput:
    """Walk-forward validation with rolling/expanding windows and
    out-of-sample performance aggregation."""
    from .walk_forward import WalkForwardConfig, WalkForwardEngine
    from .backtest_service import StrategyConfig
    from .execution_model import retail_config, institutional_config, zero_cost_config
    req = schemas.WalkForwardRequest.model_validate(params)

    exec_map = {"zero_cost": zero_cost_config, "retail": retail_config,
                "institutional": institutional_config}
    exec_cfg = exec_map.get(req.execution_preset, retail_config)()

    config = WalkForwardConfig(
        assets=req.assets,
        window_type=req.window_type,
        train_years=req.train_years,
        test_years=req.test_years,
        total_years=req.total_years,
        strategy=StrategyConfig(fast_window=req.fast_window,
                                slow_window=req.slow_window),
        execution=exec_cfg,
        optimize_parameters=req.optimize_parameters,
    )

    try:
        engine = WalkForwardEngine(config)
        result = engine.run()
    except ValueError as exc:
        raise TaskError(422, str(exc)) from exc
    except Exception as exc:
        log.exception("Walk-forward validation failed")
        raise TaskError(500, "Walk-forward failed") from exc

    # Every parameter set the in-sample search evaluated is a trial.
    common = {"window_type": req.window_type, "train_years": req.train_years,
              "test_years": req.test_years, "total_years": req.total_years,
              "execution_preset": req.execution_preset}
    trials = TrialRecord(research_trials.family_hash("ma_crossover", req.assets),
                         [{**common, **g} for g in result["parameter_grid"]], "walk_forward")
    audit = {"assets": req.assets, "window_type": req.window_type,
             "n_folds": result.get("n_folds", 0)}
    return TaskOutput(result, audit, trials)


def event_backtest(params: dict) -> TaskOutput:
    """Event-driven backtest with 1-bar execution delay, realistic commissions,
    bid/ask spread, slippage, and borrow costs."""
    import yfinance as yf
    from . import historical_data
    from .event_simulator import run_event_backtest
    req = schemas.EventBacktestRequest.model_validate(params)

    try:
        raw = yf.download(req.assets, period=req.period, interval="1d",
                          progress=False, auto_adjust=True)
        if raw is None or raw.empty:
            raise TaskError(422, "Failed to download market data")

        # Bar i is the same date for every ticker (not merely the i-th bar
        # each happened to have).
        panel = historical_data.aligned_panel(raw, req.assets, min_rows=60)
        if not panel.tickers or len(panel) < 60:
            raise TaskError(422, "No tickers had sufficient data")
        price_data: dict[str, np.ndarray] = {
            t: panel.close[t].to_numpy(dtype=float) for t in panel.tickers
        }
    except TaskError:
        raise
    except Exception as exc:
        raise TaskError(422, str(exc)) from exc

    strategy_params = {
        "fast": req.fast_window, "slow": req.slow_window,
        "lookback": 60, "n_long": 3, "n_short": 3,
    }

    try:
        result = run_event_backtest(
            tickers=list(price_data.keys()),
            price_data=price_data,
            strategy_name=req.strategy,
            strategy_params=strategy_params,
            initial_capital=req.initial_capital,
            cost_model_name=req.cost_model,
            allow_short=req.allow_short,
            sizing_method=req.sizing_method,
        )
    except Exception as exc:
        log.exception("Event backtest failed")
        raise TaskError(500, "Event backtest failed") from exc

    trials = TrialRecord(research_trials.family_hash(f"event:{req.strategy}", req.assets),
                         [req.model_dump()], "event_backtest")
    audit = {"n_assets": len(price_data), "strategy": req.strategy, "cost_model": req.cost_model}
    return TaskOutput(result, audit, trials)


# --------------------------------------------------------------------------
# Cross-sectional research
# --------------------------------------------------------------------------

def alpha_research(params: dict) -> TaskOutput:
    """Alpha research: IC, Rank IC, IC decay, hit rate, quintile analysis,
    factor turnover — measures signal predictive quality before backtesting."""
    from .alpha_research import run_alpha_research
    from .factor_model import compute_factors
    req = schemas.AlphaResearchRequest.model_validate(params)

    try:
        return_matrix, price_matrix, names = fetch_return_matrix(req.assets, req.period)
    except ValueError as exc:
        raise TaskError(422, str(exc)) from exc

    T, N = return_matrix.shape

    factor_mats = compute_factors(return_matrix, price_matrix)
    sig_key_map = {
        "momentum": "momentum", "reversal": "reversal",
        "volatility": "volatility", "quality": "quality",
        "sba": "momentum",  # fallback for SBA to momentum in cross-section
    }
    sig_key = sig_key_map.get(req.signal_type, "momentum")
    signal_matrix = factor_mats.get(sig_key, factor_mats.get("momentum"))
    if signal_matrix is None:
        raise TaskError(422, "Could not compute signal matrix")

    try:
        result = run_alpha_research(signal_matrix, return_matrix,
                                    max_horizon=req.max_horizon)
    except Exception as exc:
        log.exception("Alpha research failed")
        raise TaskError(500, "Alpha research failed") from exc

    result["asset_names"] = names
    result["signal_type"] = req.signal_type
    audit = {"n_assets": N, "signal_type": req.signal_type, "period": req.period}
    return TaskOutput(result, audit)


def factor_model(params: dict) -> TaskOutput:
    """Fama-MacBeth cross-sectional factor model with Newey-West inference."""
    from .factor_model import compute_factors, fama_macbeth, barra_risk_decomposition
    req = schemas.FactorModelRequest.model_validate(params)

    try:
        return_matrix, price_matrix, names = fetch_return_matrix(req.assets, req.period)
    except ValueError as exc:
        raise TaskError(422, str(exc)) from exc

    all_factors = compute_factors(return_matrix, price_matrix)
    requested = {k: v for k, v in all_factors.items() if k in req.factors}
    if not requested:
        requested = all_factors  # use all if none match

    try:
        fm_result = fama_macbeth(return_matrix, requested,
                                 newey_west_lags=req.newey_west_lags)
        risk_result = barra_risk_decomposition(return_matrix, requested)
    except Exception as exc:
        log.exception("Factor model failed")
        raise TaskError(500, "Factor model failed") from exc

    result = {
        "fama_macbeth": fm_result,
        "risk_decomposition": risk_result,
        "asset_names": names,
        "factors_computed": list(requested.keys()),
    }
    audit = {"n_assets": len(names), "factors": list(requested.keys())}
    return TaskOutput(result, audit)


def correlation(params: dict) -> TaskOutput:
    """Multi-method correlation analysis: Pearson, Spearman, EWMA,
    Ledoit-Wolf shrinkage, OAS, PCA factor decomposition with diagnostics."""
    from .correlation_engine import run_correlation_engine
    req = schemas.CorrelationRequest.model_validate(params)

    try:
        return_matrix, _, names = fetch_return_matrix(req.assets, req.period)
    except ValueError as exc:
        raise TaskError(422, str(exc)) from exc

    try:
        result = run_correlation_engine(
            return_matrix, names,
            ewma_halflife=req.ewma_halflife,
            pca_components=req.pca_components,
        )
    except Exception as exc:
        log.exception("Correlation engine failed")
        raise TaskError(500, "Correlation engine failed") from exc

    audit = {"n_assets": len(names), "period": req.period}
    return TaskOutput(result, audit)


def portfolio_optimization(params: dict) -> TaskOutput:
    """Min-Variance, Max-Sharpe, Risk Parity, Max-Diversification,
    Equal-Weight and SBA signal-weighted portfolios with efficient frontier."""
    from .portfolio_optimization import run_portfolio_optimization, PortfolioConstraints
    req = schemas.PortfolioOptRequest.model_validate(params)

    try:
        return_matrix, _, names = fetch_return_matrix(req.assets, req.period)
    except ValueError as exc:
        raise TaskError(422, str(exc)) from exc

    con = PortfolioConstraints(
        long_only=req.long_only,
        min_weight=req.min_weight,
        max_weight=req.max_weight,
    )

    # SBA signals: use momentum factor as proxy
    sba_signals = None
    if req.include_sba:
        from .factor_model import compute_factors
        factors = compute_factors(return_matrix)
        mom = factors.get("momentum")
        if mom is not None:
            last_valid = mom[-1, :]
            valid = np.isfinite(last_valid)
            if valid.sum() > 0:
                sba_signals = np.where(valid, np.maximum(last_valid, 0), 0)

    try:
        result = run_portfolio_optimization(
            return_matrix, names,
            corr_method=req.covariance_method,
            rf_rate=req.risk_free_rate / 252,  # convert annual to daily
            constraints=con,
            sba_signals=sba_signals,
        )
    except Exception as exc:
        log.exception("Portfolio optimisation failed")
        raise TaskError(500, "Optimisation failed") from exc

    audit = {"n_assets": len(names), "method": req.covariance_method}
    return TaskOutput(result, audit)


# --------------------------------------------------------------------------
# Regimes, market-neutral strategies, benchmarks and reports
# --------------------------------------------------------------------------

def regime_detection(params: dict) -> TaskOutput:
    """Gaussian HMM, volatility-percentile and SMA-trend regimes."""
    from .regime_detection import run_regime_detection
    req = schemas.RegimeDetectionRequest.model_validate(params)

    try:
        returns, prices = fetch_single_asset(req.asset, req.period)
    except ValueError as exc:
        raise TaskError(422, str(exc)) from exc

    try:
        result = run_regime_detection(returns, prices=prices,
                                      hmm_iters=req.hmm_iters)
    except Exception as exc:
        log.exception("Regime detection failed")
        raise TaskError(500, "Regime detection failed") from exc

    result["asset"] = req.asset
    result["period"] = req.period
    audit = {"asset": req.asset, "period": req.period}
    return TaskOutput(result, audit)


def neutral_strategy(params: dict) -> TaskOutput:
    """Cross-sectional dollar-neutral long/short strategy with optional
    factor neutralisation."""
    from .neutral_strategies import run_neutral_strategies
    from .factor_model import compute_factors
    req = schemas.NeutralStrategyRequest.model_validate(params)

    try:
        return_matrix, price_matrix, names = fetch_return_matrix(req.assets, req.period)
    except ValueError as exc:
        raise TaskError(422, str(exc)) from exc

    factor_mats = compute_factors(return_matrix, price_matrix)
    sig_key_map = {
        "momentum": "momentum", "reversal": "reversal",
        "volatility": "low_volatility", "quality": "quality",
    }
    sig_key = sig_key_map.get(req.signal_type, "momentum")
    signal_matrix = factor_mats.get(sig_key, factor_mats.get("momentum"))
    if signal_matrix is None:
        raise TaskError(422, "Could not compute signal matrix")

    # Optional factor exposures for neutralisation
    factor_exposures = None
    if req.factor_neutral:
        # Use momentum and low-vol as neutralisation factors
        mom = factor_mats.get("momentum")
        lvol = factor_mats.get("low_volatility")
        if mom is not None and lvol is not None:
            T = return_matrix.shape[0]
            # Use time-averaged exposures (cross-sectional mean per asset)
            valid_t = np.where(np.all(np.isfinite(mom[-min(252, T):, :]), axis=1))[0]
            if len(valid_t) >= 5:
                fe_mom = np.nanmean(mom[valid_t, :], axis=0)
                fe_lvol = np.nanmean(lvol[valid_t, :], axis=0)
                factor_exposures = np.column_stack([fe_mom, fe_lvol])
                factor_exposures = np.where(np.isfinite(factor_exposures),
                                            factor_exposures, 0.0)

    try:
        result = run_neutral_strategies(
            signal_matrix=signal_matrix,
            return_matrix=return_matrix,
            asset_names=names,
            factor_exposures=factor_exposures,
        )
    except Exception as exc:
        log.exception("Neutral strategy failed")
        raise TaskError(500, "Neutral strategy failed") from exc

    result["asset_names"] = names
    result["signal_type"] = req.signal_type
    audit = {"n_assets": len(names), "signal_type": req.signal_type}
    return TaskOutput(result, audit)


def pairs_trading(params: dict) -> TaskOutput:
    """Engle-Granger cointegration, Kalman hedge ratio and Z-score signals."""
    import yfinance as yf
    from . import historical_data
    from .neutral_strategies import pairs_trading_signals
    req = schemas.PairsTradingRequest.model_validate(params)

    if req.asset_y == req.asset_x:
        raise TaskError(422, "asset_y and asset_x must be different")

    try:
        raw = yf.download([req.asset_y, req.asset_x], period=req.period,
                          interval="1d", progress=False, auto_adjust=True)
        if raw is None or raw.empty:
            raise TaskError(422, "Failed to download pair data")

        # Cointegration compares the two legs date by date, so they must be
        # aligned on the shared calendar.
        panel = historical_data.aligned_panel(raw, [req.asset_y, req.asset_x])
        for ticker in (req.asset_y, req.asset_x):
            if ticker not in panel.tickers:
                raise ValueError(f"No price data for {ticker}")
        if len(panel) < 60:
            raise ValueError(f"Only {len(panel)} common trading days for {req.asset_y}/{req.asset_x}")
        prices_y = panel.close[req.asset_y].to_numpy(dtype=float)
        prices_x = panel.close[req.asset_x].to_numpy(dtype=float)
        min_len = len(panel)
    except TaskError:
        raise
    except Exception as exc:
        raise TaskError(422, str(exc)) from exc

    try:
        result = pairs_trading_signals(
            y=prices_y, x=prices_x,
            entry_z=req.entry_z,
            exit_z=req.exit_z,
            use_kalman=req.use_kalman,
        )
    except Exception as exc:
        log.exception("Pairs trading failed")
        raise TaskError(500, "Pairs trading failed") from exc

    result["asset_y"] = req.asset_y
    result["asset_x"] = req.asset_x
    result["n_bars"] = min_len
    audit = {"pair": f"{req.asset_y}/{req.asset_x}"}
    return TaskOutput(result, audit)


def latency_benchmark(params: dict) -> TaskOutput:
    """End-to-end research pipeline latency, optionally as percentiles and
    C++-versus-NumPy comparisons."""
    from .latency_bench import run_full_benchmark, run_percentile_benchmark, bench_cpp_vs_python
    req = schemas.LatencyBenchmarkRequest.model_validate(params)

    try:
        return_matrix, price_matrix, names = fetch_return_matrix(req.assets, req.period)
    except ValueError as exc:
        raise TaskError(422, str(exc)) from exc

    results: dict = {}
    try:
        if req.percentile_mode:
            results["percentile_profile"] = run_percentile_benchmark(
                return_matrix, price_matrix,
                tickers=names,
                n_runs=req.n_runs,
            )
        else:
            results = run_full_benchmark(return_matrix, price_matrix, tickers=names)
    except Exception as exc:
        log.exception("Latency benchmark failed")
        raise TaskError(500, "Benchmark failed") from exc

    if req.cpp_vs_python:
        try:
            T, N = return_matrix.shape
            results["cpp_vs_python"] = bench_cpp_vs_python(
                T=min(T, 500), N=min(N, 10), n_runs=min(req.n_runs, 20)
            )
        except Exception as exc:
            results["cpp_vs_python"] = {"error": str(exc)}

    audit = {"n_assets": len(names), "percentile_mode": req.percentile_mode, "n_runs": req.n_runs}
    return TaskOutput(results, audit)


def research_report(params: dict) -> TaskOutput:
    """The 7-section quant research report pipeline."""
    from .report_generator import run_full_report_pipeline
    req = schemas.ReportRequest.model_validate(params)

    try:
        report = run_full_report_pipeline(
            tickers=req.assets,
            period=req.period,
            strategy_type=req.strategy_type,
            run_wf=req.include_walk_forward,
            run_factor=req.include_factor_model,
            run_regime=req.include_regime,
        )
    except Exception as exc:
        log.exception("Research report generation failed")
        raise TaskError(500, "Report generation failed") from exc

    if "error" in report:
        raise TaskError(422, report["error"])

    audit = {"n_assets": len(req.assets), "period": req.period, "strategy": req.strategy_type}
    return TaskOutput(report, audit)
