"""QuantumSentinel — Research Report Generator (Phase 4).

Runs the full quant research pipeline for a given asset universe and
produces a structured, self-contained JSON report with seven sections:

  1. executive_summary    — Sharpe, alpha, IR, max drawdown, turnover
  2. walk_forward_table   — per-fold OOS Sharpe + degradation flag
  3. factor_premia_table  — factor, premium, t-stat, significant
  4. regime_statistics    — bull/bear pct_time, ann_return, ann_vol, Sharpe
  5. statistical_validation — Newey-West t, permutation p, DSR result
  6. risk_decomposition   — CVaR, Sortino, Calmar, Omega ratio
  7. efficient_frontier   — (vol, return) pairs for charting

All computations delegate to the existing service modules.
The report is fully serialisable to JSON (no numpy types).
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean(v):
    """Recursively convert numpy scalars/arrays to Python primitives."""
    if isinstance(v, dict):
        return {k: _clean(vv) for k, vv in v.items()}
    if isinstance(v, list):
        return [_clean(x) for x in v]
    if isinstance(v, np.ndarray):
        return [_clean(x) for x in v.tolist()]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _safe_sharpe(returns: np.ndarray) -> float:
    std = float(np.std(returns, ddof=1))
    if std < 1e-9:
        return 0.0
    return float(np.mean(returns) / std * math.sqrt(252))


def _cvar(returns: np.ndarray, pct: float = 5.0) -> float:
    if len(returns) < 10:
        return 0.0
    cutoff = np.percentile(returns, pct)
    tail = returns[returns <= cutoff]
    return float(tail.mean()) if len(tail) > 0 else float(cutoff)


def _sortino(returns: np.ndarray) -> float:
    downside = returns[returns < 0]
    if len(downside) < 2:
        return 0.0
    dsd = float(np.std(downside, ddof=1) * math.sqrt(252))
    if dsd < 1e-9:
        return 0.0
    return float(np.mean(returns) * 252 / dsd)


def _calmar(returns: np.ndarray) -> float:
    cum = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / np.maximum(peak, 1e-9)
    max_dd = float(dd.min())
    ann_ret = float(np.mean(returns) * 252)
    if max_dd >= 0:
        return ann_ret
    return ann_ret / abs(max_dd)


def _omega(returns: np.ndarray, threshold: float = 0.0) -> float:
    """Omega ratio: E[max(R-L,0)] / E[max(L-R,0)]."""
    gains = np.maximum(returns - threshold, 0.0)
    losses = np.maximum(threshold - returns, 0.0)
    g = float(gains.mean())
    l_ = float(losses.mean())
    return g / max(l_, 1e-9)


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _build_executive_summary(returns: np.ndarray,
                              benchmark_returns: np.ndarray | None = None
                              ) -> dict:
    valid = returns[np.isfinite(returns)]
    if len(valid) < 5:
        return {"error": "insufficient returns"}

    ann_ret = float(valid.mean() * 252)
    ann_vol = float(valid.std(ddof=1) * math.sqrt(252))
    sharpe  = ann_ret / max(ann_vol, 1e-9)

    cum = np.cumprod(1 + valid)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / np.maximum(peak, 1e-9)
    max_dd = float(dd.min())

    alpha = ann_ret
    ir = 0.0
    if benchmark_returns is not None and len(benchmark_returns) == len(valid):
        alpha = ann_ret - float(benchmark_returns.mean() * 252)
        te = float((valid - benchmark_returns).std(ddof=1) * math.sqrt(252))
        ir = alpha / max(te, 1e-9)

    # Turnover proxy: fraction of days with |return| > 0 (signal days)
    active_days = int(np.sum(np.abs(valid) > 1e-6))
    turnover_proxy = round(active_days / max(len(valid), 1), 4)

    return {
        "annual_return":     round(ann_ret, 4),
        "annual_volatility": round(ann_vol, 4),
        "sharpe_ratio":      round(sharpe,  4),
        "alpha_vs_benchmark": round(alpha, 4),
        "information_ratio": round(ir,     4),
        "max_drawdown":      round(max_dd, 4),
        "active_days_pct":   turnover_proxy,
        "n_observations":    len(valid),
    }


def _build_risk_decomposition(returns: np.ndarray) -> dict:
    valid = returns[np.isfinite(returns)]
    if len(valid) < 10:
        return {"error": "insufficient returns"}

    return {
        "cvar_5pct":   round(_cvar(valid, 5.0),  6),
        "cvar_1pct":   round(_cvar(valid, 1.0),  6),
        "sortino_ratio": round(_sortino(valid),   4),
        "calmar_ratio":  round(_calmar(valid),    4),
        "omega_ratio":   round(_omega(valid),     4),
        "skewness":      round(float(
            np.mean(((valid - valid.mean()) / max(valid.std(ddof=1), 1e-9)) ** 3)
        ), 4),
        "excess_kurtosis": round(float(
            np.mean(((valid - valid.mean()) / max(valid.std(ddof=1), 1e-9)) ** 4) - 3
        ), 4),
    }


def _build_wf_table(wf_result: dict) -> list[dict]:
    """Convert walk-forward result into a clean per-fold table."""
    folds = wf_result.get("folds", [])
    if not folds:
        return []
    rows = []
    for f in folds:
        oos = f.get("oos_metrics", {})
        rows.append({
            "fold":       f.get("fold_id", 0),
            "train_start": f.get("train_start", ""),
            "train_end":   f.get("train_end",   ""),
            "test_start":  f.get("test_start",  ""),
            "test_end":    f.get("test_end",    ""),
            "oos_sharpe":  round(oos.get("sharpe_ratio", 0.0), 4),
            "oos_return":  round(oos.get("annual_return", 0.0), 4),
            "oos_max_dd":  round(oos.get("max_drawdown",  0.0), 4),
            "best_fast":   f.get("best_fast_window", None),
            "best_slow":   f.get("best_slow_window", None),
            "degraded":    oos.get("sharpe_ratio", 0.0) < 0,
        })
    return rows


def _build_factor_table(fm_result: dict) -> list[dict]:
    """Convert Fama-MacBeth result into a clean factor premia table."""
    premia = fm_result.get("factor_premia", {})
    rows = []
    for factor, stats in premia.items():
        t_stat = stats.get("t_stat", 0.0)
        rows.append({
            "factor":    factor,
            "premium_ann": round(stats.get("premium_ann", 0.0), 4),
            "t_stat":    round(t_stat, 4),
            "p_value":   round(stats.get("p_value", 1.0), 6),
            "significant_5pct": abs(t_stat) > 1.96,
            "newey_west_se": round(stats.get("nw_se", 0.0), 6),
        })
    # Sort by |t_stat| descending
    rows.sort(key=lambda x: abs(x["t_stat"]), reverse=True)
    return rows


def _build_regime_table(regime_result: dict) -> dict:
    """Extract regime statistics into a report-friendly structure."""
    hmm = regime_result.get("hmm_regime", {})
    vol = regime_result.get("volatility_regime", {})
    current = regime_result.get("current_summary", {})

    return {
        "current_hmm":    current.get("hmm",   "unknown"),
        "current_vol":    current.get("vol",   "unknown"),
        "bull_prob_pct":  current.get("bull_prob_pct", 50.0),
        "hmm_regimes":    hmm.get("regime_stats",   {}),
        "vol_regimes":    vol.get("regime_stats",   {}),
        "transition_matrix": hmm.get("transition_matrix", {}),
        "n_regime_switches": hmm.get("n_regime_switches", 0),
    }


def _build_stat_validation(stat_result: dict) -> dict:
    """Extract the key statistical test results."""
    nw   = stat_result.get("ttest_nw",          {})
    bs   = stat_result.get("bootstrap_sharpe",  {})
    perm = stat_result.get("permutation_test",  {})
    dsr  = stat_result.get("deflated_sharpe",   {})
    lb   = stat_result.get("ljung_box",         {})
    summ = stat_result.get("summary",           {})

    return {
        "newey_west_t_stat":    round(nw.get("t_stat", 0.0), 4),
        "newey_west_p_value":   round(nw.get("p_value", 1.0), 6),
        "mean_significant_5pct": nw.get("significant_5pct", False),
        "bootstrap_sharpe":     round(bs.get("sharpe",   0.0), 4),
        "sharpe_ci_lower":      round(bs.get("ci_lower", 0.0), 4),
        "sharpe_ci_upper":      round(bs.get("ci_upper", 0.0), 4),
        "ci_excludes_zero":     bs.get("ci_lower", 0.0) > 0,
        "permutation_p_value":  round(perm.get("p_value", 1.0), 6),
        "permutation_significant": perm.get("significant_5pct", False),
        "deflated_sharpe_p":    round(dsr.get("dsr_p_value", 1.0), 6),
        "survives_deflation":   dsr.get("significant_5pct", False),
        "has_autocorrelation":  lb.get("has_significant_autocorrelation", False),
        "overall_credible":     summ.get("overall_credible", False),
        "dsr_interpretation":   dsr.get("interpretation", ""),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_report(
    returns: np.ndarray,
    benchmark_returns: np.ndarray | None = None,
    wf_result:     dict | None = None,
    factor_result: dict | None = None,
    regime_result: dict | None = None,
    stat_result:   dict | None = None,
    frontier:      list | None = None,
    metadata:      dict | None = None,
) -> dict:
    """Assemble a full quant research report from pre-computed results.

    All arguments except `returns` are optional — missing sections are
    marked with {"status": "not_run"}.

    Parameters
    ----------
    returns          : (T,) daily strategy return series
    benchmark_returns: (T,) benchmark return series (optional)
    wf_result        : output of WalkForwardEngine.run()
    factor_result    : output of fama_macbeth()
    regime_result    : output of run_regime_detection()
    stat_result      : output of run_full_stat_tests()
    frontier         : efficient frontier points from portfolio_optimization
    metadata         : arbitrary dict (assets, period, strategy name, etc.)

    Returns
    -------
    dict — fully JSON-serialisable report
    """
    rng = np.asarray(returns, dtype=np.float64)
    rng = rng[np.isfinite(rng)]

    report: dict = {
        "version": "4.0.0",
        "metadata": _clean(metadata or {}),
    }

    # 1. Executive summary
    bench_arr = (np.asarray(benchmark_returns, dtype=np.float64)
                 if benchmark_returns is not None else None)
    report["executive_summary"] = _clean(
        _build_executive_summary(rng, bench_arr)
    )

    # 2. Walk-forward table
    if wf_result is not None:
        report["walk_forward_table"] = _clean(_build_wf_table(wf_result))
        summary_oos = wf_result.get("aggregated_oos", {})
        report["walk_forward_summary"] = _clean({
            "n_folds":          wf_result.get("n_folds", 0),
            "pct_profitable_folds": wf_result.get("pct_profitable_folds", 0),
            "mean_oos_sharpe":  summary_oos.get("mean_sharpe", 0.0),
            "oos_degradation":  wf_result.get("sharpe_degradation_pct", 0.0),
            "overfitting_flag": wf_result.get("overfitting_detected", False),
        })
    else:
        report["walk_forward_table"] = {"status": "not_run"}
        report["walk_forward_summary"] = {"status": "not_run"}

    # 3. Factor premia table
    if factor_result is not None:
        report["factor_premia_table"] = _clean(_build_factor_table(factor_result))
        report["factor_model_summary"] = _clean({
            "n_factors":     len(factor_result.get("factor_premia", {})),
            "n_significant": sum(
                1 for s in factor_result.get("factor_premia", {}).values()
                if abs(s.get("t_stat", 0.0)) > 1.96
            ),
            "avg_r_squared": factor_result.get("avg_r_squared", None),
        })
    else:
        report["factor_premia_table"] = {"status": "not_run"}
        report["factor_model_summary"] = {"status": "not_run"}

    # 4. Regime statistics
    if regime_result is not None:
        report["regime_statistics"] = _clean(_build_regime_table(regime_result))
    else:
        report["regime_statistics"] = {"status": "not_run"}

    # 5. Statistical validation
    if stat_result is not None:
        report["statistical_validation"] = _clean(
            _build_stat_validation(stat_result)
        )
    else:
        report["statistical_validation"] = {"status": "not_run"}

    # 6. Risk decomposition
    report["risk_decomposition"] = _clean(_build_risk_decomposition(rng))

    # 7. Efficient frontier
    if frontier is not None and len(frontier) > 0:
        report["efficient_frontier"] = _clean(frontier)
    else:
        report["efficient_frontier"] = {"status": "not_run"}

    return report


# ---------------------------------------------------------------------------
# Convenience: run the full pipeline from raw prices
# ---------------------------------------------------------------------------

def run_full_report_pipeline(
    tickers: list[str],
    period: str = "2y",
    strategy_type: str = "momentum",
    run_wf: bool = True,
    run_factor: bool = True,
    run_regime: bool = True,
) -> dict:
    """Fetch data and run the complete research pipeline, returning a report.

    This is the high-level entry point used by the API endpoint.
    Each sub-pipeline catches exceptions independently so partial failures
    do not abort the whole report.
    """
    import yfinance as yf
    from .alpha_research import compute_signal_matrix
    from .factor_model import compute_factors, fama_macbeth
    from .regime_detection import run_regime_detection
    from .walk_forward import WalkForwardConfig, WalkForwardEngine
    from .backtest_service import StrategyConfig
    from .execution_model import retail_config
    from .stat_tests import run_full_stat_tests

    metadata = {
        "tickers": tickers,
        "period": period,
        "strategy": strategy_type,
        "n_assets": len(tickers),
    }

    # ── Fetch data ──
    try:
        raw = yf.download(tickers, period=period, auto_adjust=True,
                          progress=False, group_by="ticker")
        if len(tickers) == 1:
            prices = raw["Close"].values
            price_matrix = prices.reshape(-1, 1)
        else:
            close_df = raw.xs("Close", axis=1, level=0)[tickers]
            close_df = close_df.dropna(how="all").ffill().bfill()
            price_matrix = close_df.values
            prices = price_matrix[:, 0]

        return_matrix = np.diff(price_matrix, axis=0) / np.maximum(
            price_matrix[:-1], 1e-9
        )
        returns_1d = return_matrix[:, 0]
        T, N = return_matrix.shape
        metadata["n_bars"] = T
    except Exception as exc:
        return {"error": f"Data fetch failed: {exc}", "metadata": metadata}

    # ── Walk-forward ──
    wf_result = None
    if run_wf and len(tickers) >= 1:
        try:
            cfg = WalkForwardConfig(
                assets=[tickers[0]],
                window_type="rolling",
                train_years=2,
                test_years=1,
                total_years=5,
                strategy=StrategyConfig(fast_window=20, slow_window=50),
                execution=retail_config(),
                optimize_parameters=True,
            )
            wf_result = WalkForwardEngine(cfg).run()
        except Exception as exc:
            log.warning("Walk-forward failed: %s", exc)

    # ── Factor model ──
    factor_result = None
    if run_factor and N >= 4:
        try:
            factors = compute_factors(return_matrix)
            factor_result = fama_macbeth(return_matrix, factors)
        except Exception as exc:
            log.warning("Factor model failed: %s", exc)

    # ── Regime detection ──
    regime_result = None
    if run_regime:
        try:
            regime_result = run_regime_detection(
                returns_1d, prices=prices[:len(returns_1d)]
            )
        except Exception as exc:
            log.warning("Regime detection failed: %s", exc)

    # ── Statistical tests (on momentum signal returns) ──
    stat_result = None
    try:
        stat_result = run_full_stat_tests(returns_1d, n_strategies_tested=1)
    except Exception as exc:
        log.warning("Stat tests failed: %s", exc)

    # ── Efficient frontier ──
    frontier = None
    if N >= 2:
        try:
            from .portfolio_optimization import run_portfolio_optimization
            opt = run_portfolio_optimization(return_matrix, asset_names=tickers)
            frontier = opt.get("efficient_frontier", None)
        except Exception as exc:
            log.warning("Portfolio optimisation failed: %s", exc)

    return generate_report(
        returns=returns_1d,
        wf_result=wf_result,
        factor_result=factor_result,
        regime_result=regime_result,
        stat_result=stat_result,
        frontier=frontier,
        metadata=metadata,
    )
