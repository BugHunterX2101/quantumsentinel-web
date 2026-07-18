"""QuantumSentinel — Quantum-Inspired Signal Engine (Simulated Bifurcation Algorithm).

Ported to Python/NumPy from the Rust reference design in the architecture doc
(sba.rs). Runs the same bifurcation dynamics over REAL market data pulled
from Yahoo Finance via `yfinance` — this is not a toy random-number demo,
the RSI/MACD/momentum/Bollinger features and the resulting BUY/SELL/HOLD
signals are computed from actual recent price action.

H = -1/2 * sum_ij J_ij * s_i * s_j - sum_i h_i * s_i   (Ising Hamiltonian)
dx_i/dt = y_i
dy_i/dt = (a(t)-1)*x_i - x_i^3 + c * sum_j J_ij*x_j + h_i
"""
import time
import threading
import numpy as np
import yfinance as yf

TRACKED_ASSETS = ["AAPL", "MSFT", "TSLA", "NVDA", "GOOGL", "AMZN", "META", "SPY"]

N_STEPS = 100     # T
DT = 0.1          # delta_t
COUPLING = 0.5    # c

_cache = {"signals": {}, "generated_at": 0.0, "lock": threading.Lock()}
CACHE_TTL_SECONDS = 45


# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------
def _rsi(close: np.ndarray, period: int = 14) -> float:
    if len(close) < period + 1:
        return 50.0
    deltas = np.diff(close[-(period + 1):])
    gains = deltas[deltas > 0].sum()
    losses = -deltas[deltas < 0].sum()
    avg_gain = gains / period
    avg_loss = losses / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - 100 / (1 + rs))


def _ema(values: np.ndarray, span: int) -> np.ndarray:
    alpha = 2 / (span + 1)
    out = np.empty_like(values, dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def _macd_histogram(close: np.ndarray) -> float:
    if len(close) < 30:
        return 0.0
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd_line = ema12 - ema26
    signal_line = _ema(macd_line, 9)
    return float((macd_line - signal_line)[-1])


def _momentum(close: np.ndarray, lookback: int = 10) -> float:
    if len(close) <= lookback:
        return 0.0
    return float(close[-1] / close[-1 - lookback] - 1)


def _bollinger_width(close: np.ndarray, window: int = 20) -> float:
    if len(close) < window:
        return 0.0
    window_vals = close[-window:]
    mid = window_vals.mean()
    std = window_vals.std()
    upper, lower = mid + 2 * std, mid - 2 * std
    if mid == 0:
        return 0.0
    return float((upper - lower) / mid)


def extract_features(close: np.ndarray) -> dict:
    return {
        "rsi": _rsi(close),
        "macd_histogram": _macd_histogram(close),
        "momentum": _momentum(close),
        "bb_width": _bollinger_width(close),
    }


# --------------------------------------------------------------------------
# Simulated Bifurcation Algorithm
# --------------------------------------------------------------------------
def build_coupling_matrix(returns_matrix: np.ndarray) -> np.ndarray:
    """J_ij = Pearson correlation of daily-return time series over a rolling
    20-bar window (returns_matrix shape: n_assets x n_bars), per the
    architecture doc. Mean-field normalized by n so coupling strength stays
    comparable regardless of basket size (otherwise a large correlated
    basket saturates every spin toward the same sign)."""
    n = returns_matrix.shape[0]
    if n < 2:
        return np.zeros((n, n))
    J = np.corrcoef(returns_matrix)
    np.fill_diagonal(J, 0.0)
    J = np.nan_to_num(J, nan=0.0)
    return J / n


def run_sba(coupling_matrix: np.ndarray, local_fields: np.ndarray,
            n_steps: int = N_STEPS, dt: float = DT, coupling: float = COUPLING) -> np.ndarray:
    """Vectorized NumPy port of the Rust rayon-parallel bifurcation loop."""
    n = len(local_fields)
    rng = np.random.default_rng()
    x = rng.uniform(-0.05, 0.05, size=n)
    y = np.zeros(n)
    for step in range(n_steps):
        a = step / n_steps
        coupling_sum = coupling_matrix @ x
        dy = (a - 1.0) * x - x ** 3 + coupling * coupling_sum + local_fields
        y_new = y + dy * dt
        x_new = np.clip(x + y_new * dt, -1.0, 1.0)
        x, y = x_new, y_new
    return x


def score_signal(spin: float, rsi: float) -> tuple[str, float]:
    raw_confidence = abs(spin)
    rsi_penalty = 0.3 if ((spin > 0 and rsi > 70) or (spin < 0 and rsi < 30)) else 0.0
    confidence = max(0.0, min(1.0, raw_confidence - rsi_penalty))
    if spin > 0.1:
        signal_type = "BUY"
    elif spin < -0.1:
        signal_type = "SELL"
    else:
        signal_type = "HOLD"
    return signal_type, confidence


# --------------------------------------------------------------------------
# End-to-end pipeline
# --------------------------------------------------------------------------
def generate_signals(assets: list[str] | None = None) -> dict:
    assets = assets or TRACKED_ASSETS
    t0 = time.perf_counter()

    data = yf.download(assets, period="3mo", interval="1d", progress=False, group_by="ticker")

    feature_rows, returns_rows, closes_last, rsis = [], [], {}, {}
    valid_assets = []
    for asset in assets:
        try:
            close = data[asset]["Close"].dropna().to_numpy()
        except Exception:
            continue
        if len(close) < 25:
            continue
        feats = extract_features(close)
        feature_rows.append([feats["rsi"], feats["macd_histogram"], feats["momentum"], feats["bb_width"]])
        # rolling 20-bar daily-return series feeds the coupling matrix (asset correlation)
        returns = np.diff(close[-21:]) / close[-21:-1]
        returns_rows.append(returns)
        closes_last[asset] = float(close[-1])
        rsis[asset] = feats["rsi"]
        valid_assets.append(asset)

    if not valid_assets:
        return {"signals": [], "generated_at": time.time(), "pipeline_ms": 0}

    returns_matrix = np.array(returns_rows)
    J = build_coupling_matrix(returns_matrix)

    # h_i: idiosyncratic momentum relative to the basket (cross-sectional
    # z-score) so a stock's OWN trend can outweigh herd correlation.
    momentum_vals = np.array([row[2] for row in feature_rows])
    mu, sigma = momentum_vals.mean(), momentum_vals.std() or 1e-6
    h = np.clip((momentum_vals - mu) / sigma, -3, 3) * 0.35

    sba_t0 = time.perf_counter()
    spins = run_sba(J, h)
    sba_ms = (time.perf_counter() - sba_t0) * 1000

    signals = []
    for i, asset in enumerate(valid_assets):
        signal_type, confidence = score_signal(float(spins[i]), rsis[asset])
        signals.append({
            "asset": asset,
            "signal_type": signal_type,
            "confidence": round(confidence, 4),
            "spin": round(float(spins[i]), 4),
            "last_price": round(closes_last[asset], 2),
            "features": {k: round(v, 4) for k, v in zip(
                ["rsi", "macd_histogram", "momentum", "bb_width"], feature_rows[i])},
            "sba_iterations": N_STEPS,
            "engine_version": "1.0.0-python-sba",
        })

    total_ms = (time.perf_counter() - t0) * 1000
    result = {
        "signals": signals,
        "generated_at": time.time(),
        "pipeline_ms": round(total_ms, 2),
        "sba_ms": round(sba_ms, 2),
        "n_assets": len(valid_assets),
    }
    return result


def get_cached_signals(assets: list[str] | None = None) -> dict:
    with _cache["lock"]:
        now = time.time()
        if now - _cache["generated_at"] < CACHE_TTL_SECONDS and _cache["signals"]:
            return _cache["signals"]
    fresh = generate_signals(assets)
    with _cache["lock"]:
        _cache["signals"] = fresh
        _cache["generated_at"] = time.time()
    return fresh
