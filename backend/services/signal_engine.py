"""QuantumSentinel — Signal Engine (Simulated Bifurcation Algorithm).

Runs genuine SBA bifurcation dynamics over REAL live market data pulled from
Yahoo Finance via `yfinance`. RSI/MACD/momentum/Bollinger features and the
resulting BUY/SELL/HOLD signals are computed from actual recent price action.

H = -1/2 * sum_ij J_ij * s_i * s_j - sum_i h_i * s_i   (Ising Hamiltonian)
dx_i/dt = y_i
dy_i/dt = (a(t)-1)*x_i - x_i^3 + c * sum_j J_ij*x_j + h_i
"""
import logging
import time
import threading

import numpy as np
import yfinance as yf

log = logging.getLogger(__name__)

# 12-asset basket: 8 large-caps + 4 sector diversifiers for stronger signals
TRACKED_ASSETS = [
    "AAPL", "MSFT", "TSLA", "NVDA", "GOOGL",
    "AMZN", "META", "SPY", "AMD", "NFLX", "INTC", "QQQ",
]

N_STEPS = 200     # More iterations → better convergence
DT = 0.05         # Smaller step → more stable dynamics
COUPLING = 0.5    # coupling constant c

_cache: dict = {"signals": {}, "generated_at": 0.0, "generating": False,
                "lock": threading.Lock()}
CACHE_TTL_SECONDS = 60  # 1-min live data freshness window


# ---------------------------------------------------------------------------
# Feature extraction — all using industry-standard formulations
# ---------------------------------------------------------------------------

def _rsi_wilder(close: np.ndarray, period: int = 14) -> float:
    """Wilder's Smoothed RSI (industry standard).

    Uses a proper EMA warm-up period (first value = simple average of initial
    `period` gains/losses) then applies Wilder's smoothing (alpha = 1/period)
    for subsequent bars. This matches TradingView, Bloomberg, and Reuters.

    Simple-average RSI (Cutler's) is NOT used here — it diverges materially
    from Wilder's on shorter lookback windows and gives inaccurate overbought/
    oversold signals.
    """
    n_required = period * 3  # warm-up: 3× period for accurate smoothing
    if len(close) < n_required:
        return 50.0
    deltas = np.diff(close.astype(float))
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    # Seed: simple average of first `period` bars
    avg_gain = float(gains[:period].mean())
    avg_loss = float(losses[:period].mean())
    # Wilder's EMA smoothing for remaining bars
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss < 1e-10:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - 100.0 / (1.0 + rs))


def _ema(values: np.ndarray, span: int) -> np.ndarray:
    """Standard exponential moving average (alpha = 2/(span+1))."""
    alpha = 2.0 / (span + 1)
    out = np.empty(len(values), dtype=float)
    out[0] = float(values[0])
    for i in range(1, len(values)):
        out[i] = alpha * float(values[i]) + (1.0 - alpha) * out[i - 1]
    return out


def _macd_histogram(close: np.ndarray) -> float:
    """MACD histogram (EMA12 - EMA26 - signal9).  Requires at least 35 bars."""
    if len(close) < 35:
        return 0.0
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd_line = ema12 - ema26
    signal_line = _ema(macd_line, 9)
    return float((macd_line - signal_line)[-1])


def _momentum(close: np.ndarray, lookback: int = 20) -> float:
    """Rate-of-change momentum over `lookback` bars (20-day default)."""
    if len(close) <= lookback:
        return 0.0
    return float(close[-1] / close[-1 - lookback] - 1.0)


def _bollinger_width(close: np.ndarray, window: int = 20) -> float:
    """Bollinger Band width = (upper - lower) / mid.  Normalised volatility proxy."""
    if len(close) < window:
        return 0.0
    w = close[-window:].astype(float)
    mid = w.mean()
    if mid < 1e-10:
        return 0.0
    std = w.std(ddof=1)  # sample std
    return float(4.0 * std / mid)  # (upper-lower)/mid = 4σ/mid


def extract_features(close: np.ndarray) -> dict:
    return {
        "rsi": _rsi_wilder(close),
        "macd_histogram": _macd_histogram(close),
        "momentum": _momentum(close),
        "bb_width": _bollinger_width(close),
    }


# ---------------------------------------------------------------------------
# Simulated Bifurcation Algorithm
# ---------------------------------------------------------------------------

def build_coupling_matrix(returns_matrix: np.ndarray) -> np.ndarray:
    """J_ij = Pearson correlation of 20-bar daily returns.

    Mean-field normalised by n so coupling strength stays comparable
    regardless of basket size (otherwise a large correlated basket saturates
    every spin toward the same sign).  NaN entries (zero-variance assets) are
    zeroed rather than propagated.
    """
    n = returns_matrix.shape[0]
    if n < 2:
        return np.zeros((n, n))
    J = np.corrcoef(returns_matrix)
    np.fill_diagonal(J, 0.0)
    J = np.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0)
    return J / n


def run_sba(coupling_matrix: np.ndarray, local_fields: np.ndarray,
            n_steps: int = N_STEPS, dt: float = DT,
            coupling: float = COUPLING) -> np.ndarray:
    """Vectorised NumPy SBA bifurcation loop.

    The initial state is seeded deterministically from local_fields so
    results are reproducible for audit/backtest parity.
    """
    n = len(local_fields)
    x = np.clip(np.asarray(local_fields, dtype=float) * 0.05, -0.05, 0.05)
    y = np.zeros(n, dtype=float)
    for step in range(n_steps):
        a = step / n_steps                          # pressure ramp 0→1
        coupling_sum = coupling_matrix @ x
        dy = (a - 1.0) * x - x ** 3 + coupling * coupling_sum + local_fields
        y = y + dy * dt
        x = np.clip(x + y * dt, -1.0, 1.0)
    return x


def score_signal(spin: float, rsi: float) -> tuple[str, float]:
    """Convert a spin value + RSI into a directional signal and confidence.

    RSI confirmation logic:
      - If spin says BUY but RSI > 70 (overbought), penalise confidence
      - If spin says SELL but RSI < 30 (oversold), penalise confidence
    """
    raw_confidence = abs(spin)
    rsi_penalty = 0.25 if ((spin > 0 and rsi > 70) or (spin < 0 and rsi < 30)) else 0.0
    confidence = max(0.0, min(1.0, raw_confidence - rsi_penalty))
    if spin > 0.15:
        signal_type = "BUY"
    elif spin < -0.15:
        signal_type = "SELL"
    else:
        signal_type = "HOLD"
    return signal_type, confidence


# ---------------------------------------------------------------------------
# Data download — robust multi-ticker column access for all yfinance versions
# ---------------------------------------------------------------------------

def _extract_close(data, asset: str) -> np.ndarray | None:
    """Return the Close price series for `asset` from a yfinance download result.

    yfinance changed its MultiIndex column layout across versions:
    - v0.1.x : data[asset]["Close"]
    - v0.2.x : data["Close"][asset]  (field-first MultiIndex)
    Both patterns are tried in order.
    """
    import pandas as pd  # local import — already installed as yfinance dep
    try:
        # v0.2.x layout: top-level = field name
        if isinstance(data.columns, pd.MultiIndex):
            series = data["Close"][asset].dropna()
        else:
            # v0.1.x layout: top-level = ticker
            series = data[asset]["Close"].dropna()
        arr = series.to_numpy(dtype=float)
        return arr if len(arr) > 0 else None
    except (KeyError, TypeError):
        # Final fallback: try flat column named (Close, asset)
        try:
            arr = data[("Close", asset)].dropna().to_numpy(dtype=float)
            return arr if len(arr) > 0 else None
        except Exception:
            return None


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

def generate_signals(assets: list[str] | None = None) -> dict:
    """Download live data and run the full SBA signal pipeline.

    Returns a structured dict that is safe to serialise directly as a JSON API
    response.  All data is live from Yahoo Finance — no mocked prices.
    """
    assets = assets or TRACKED_ASSETS
    t0 = time.perf_counter()

    try:
        data = yf.download(
            assets, period="3mo", interval="1d",
            progress=False, auto_adjust=True,
        )
    except Exception as exc:
        log.warning("yfinance download failed: %s", exc)
        return {
            "signals": [], "generated_at": time.time(),
            "pipeline_ms": 0, "sba_ms": 0, "n_assets": 0,
            "error": "market data unavailable — using stale cache if present",
        }

    if data.empty:
        log.warning("yfinance returned empty DataFrame for assets: %s", assets)
        return {
            "signals": [], "generated_at": time.time(),
            "pipeline_ms": 0, "sba_ms": 0, "n_assets": 0,
            "error": "empty market data response",
        }

    feature_rows, returns_rows, closes_last, rsis = [], [], {}, {}
    valid_assets: list[str] = []
    MIN_BARS = 50  # need enough history for Wilder's RSI (14*3) + Bollinger (20)

    for asset in assets:
        close = _extract_close(data, asset)
        if close is None or len(close) < MIN_BARS:
            log.debug("Skipping %s: insufficient bars (%s)", asset,
                      len(close) if close is not None else 0)
            continue
        feats = extract_features(close)
        feature_rows.append([
            feats["rsi"], feats["macd_histogram"],
            feats["momentum"], feats["bb_width"],
        ])
        # 20-bar daily-return series for coupling matrix
        rets = np.diff(close[-21:]) / np.where(close[-21:-1] != 0, close[-21:-1], 1.0)
        returns_rows.append(rets)
        closes_last[asset] = float(close[-1])
        rsis[asset] = feats["rsi"]
        valid_assets.append(asset)

    if not valid_assets:
        return {
            "signals": [], "generated_at": time.time(),
            "pipeline_ms": round((time.perf_counter() - t0) * 1000, 2),
            "sba_ms": 0, "n_assets": 0,
            "error": "no assets had sufficient price history",
        }

    returns_matrix = np.array(returns_rows, dtype=float)
    J = build_coupling_matrix(returns_matrix)

    # h_i: cross-sectional z-score of momentum so each asset's own trend can
    # outweigh herd correlation effects.
    momentum_vals = np.array([row[2] for row in feature_rows], dtype=float)
    sigma = momentum_vals.std(ddof=1) if len(momentum_vals) > 1 else 1e-9
    mu = momentum_vals.mean()
    h = np.clip((momentum_vals - mu) / max(sigma, 1e-9), -3.0, 3.0) * 0.35

    sba_t0 = time.perf_counter()
    spins = run_sba(J, h)
    sba_ms = (time.perf_counter() - sba_t0) * 1000.0

    signals = []
    for i, asset in enumerate(valid_assets):
        signal_type, confidence = score_signal(float(spins[i]), rsis[asset])
        signals.append({
            "asset": asset,
            "signal_type": signal_type,
            "confidence": round(confidence, 4),
            "spin": round(float(spins[i]), 4),
            "last_price": round(closes_last[asset], 2),
            "features": {
                k: round(v, 4) for k, v in zip(
                    ["rsi", "macd_histogram", "momentum", "bb_width"],
                    feature_rows[i],
                )
            },
            "sba_iterations": N_STEPS,
            "engine_version": "1.1.0-python-sba",
        })

    total_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "signals": signals,
        "generated_at": time.time(),
        "pipeline_ms": round(total_ms, 2),
        "sba_ms": round(sba_ms, 2),
        "n_assets": len(valid_assets),
    }


# ---------------------------------------------------------------------------
# Thread-safe cache with double-check locking to prevent concurrent generation
# ---------------------------------------------------------------------------

def get_cached_signals(assets: list[str] | None = None) -> dict:
    """Return cached signals if fresh; regenerate otherwise.

    Uses a `_generating` flag inside the lock so only ONE thread runs the
    expensive yfinance download + SBA pipeline at a time.  Concurrent callers
    receive the stale cache while generation is in progress.
    """
    with _cache["lock"]:
        now = time.time()
        if (now - _cache["generated_at"] < CACHE_TTL_SECONDS
                and _cache["signals"]):
            return _cache["signals"]
        if _cache["generating"]:
            # Another thread is generating — return stale data to avoid pile-up
            if _cache["signals"]:
                return _cache["signals"]
        _cache["generating"] = True

    try:
        fresh = generate_signals(assets)
    except Exception:
        log.exception("signal generation failed — returning stale cache")
        with _cache["lock"]:
            _cache["generating"] = False
            return _cache["signals"] or {
                "signals": [], "generated_at": time.time(),
                "pipeline_ms": 0, "sba_ms": 0, "n_assets": 0,
                "error": "signal generation failed",
            }

    with _cache["lock"]:
        _cache["signals"] = fresh
        _cache["generated_at"] = time.time()
        _cache["generating"] = False
    return fresh


def invalidate_cache() -> None:
    """Thread-safe cache invalidation used by POST /api/signals/refresh."""
    with _cache["lock"]:
        _cache["generated_at"] = 0.0
