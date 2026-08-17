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

# ---------------------------------------------------------------------------
# Default preloaded assets — only these 20 are fetched on server startup and
# kept warm in the rolling cache. All other assets are fetched on-demand via
# compute_single_asset() when the user searches for them.
# ---------------------------------------------------------------------------
PRELOADED_ASSETS = [
    "AAPL","MSFT","NVDA","GOOGL","META",
    "TSLA","AMZN","JPM","V","JNJ",
    "XOM","SPY","QQQ","GLD","COIN",
    "NFLX","AMD","BKNG","LLY","TSM",
]

# ---------------------------------------------------------------------------
# Full universe catalogue — used for search autocomplete & exchange mapping.
# Signals are NOT pre-computed for this entire list; they are fetched on-demand.
# ---------------------------------------------------------------------------
TRACKED_ASSETS = [
    # ══ NYSE / NASDAQ (US) ════════════════════════════════════════════════════
    # Technology
    "AAPL","MSFT","NVDA","GOOGL","GOOG","META","TSLA","AVGO","ORCL","ADBE",
    "CRM","INTC","AMD","QCOM","TXN","MU","AMAT","KLAC","LRCX","SNPS",
    "CDNS","MRVL","PANW","CRWD","ZS","NET","FTNT","OKTA","DDOG","SNOW",
    "PLTR","UBER","LYFT","ABNB","BKNG","EXPE","PYPL","AFRM","SHOP",
    # Communication & Media
    "NFLX","DIS","CMCSA","T","VZ","TMUS","CHTR","WBD","PARA","FOXA",
    # Consumer Discretionary
    "AMZN","HD","LOW","TGT","WMT","COST","SBUX","MCD","YUM",
    "NKE","LULU","DECK","TPR","RL","PVH","ANF",
    # Consumer Staples
    "PG","KO","PEP","PM","MO","MDLZ","GIS","KHC","CPB","CAG",
    # Financials
    "JPM","BAC","WFC","GS","MS","C","BLK","SCHW","AXP","V",
    "MA","COF","ALLY","SYF","USB","PNC","TFC","RF","KEY",
    # Healthcare & Pharma
    "JNJ","UNH","LLY","PFE","ABBV","MRK","TMO","ABT","DHR","BMY",
    "AMGN","GILD","BIIB","REGN","VRTX","MRNA","BNTX","IQV","SYK","EW",
    # Energy
    "XOM","CVX","COP","SLB","EOG","DVN","MPC","PSX","VLO","HAL",
    # Industrials
    "BA","CAT","GE","HON","LMT","RTX","NOC","GD","MMM","UPS",
    "FDX","DE","EMR","ETN","PH","ROK","IR","CARR","OTIS",
    # Materials
    "LIN","APD","SHW","FCX","NEM","AA","ALB","MP","VALE",
    # Real Estate (REITs)
    "AMT","PLD","CCI","EQIX","PSA","SPG","O","VICI","AVB","EQR",
    # Utilities
    "NEE","DUK","SO","D","AEP","XEL","PCG","EXC","ED","FE",
    # Broad Market ETFs
    "SPY","QQQ","IWM","DIA","VTI","VOO","VEA","VWO","EFA","EEM",
    "XLK","XLF","XLV","XLE","XLI","XLY","XLP","XLB","XLRE","XLU",
    "GLD","SLV","USO","TLT","IEF","HYG","LQD","BND","AGG","TIPS",
    # US-listed ADRs
    "TSM","ASML","SAP","NVO","BABA","JD","PDD","BIDU","SE","GRAB",
    "SONY","TM","HMC","NTDOY","LI","NIO",
    # Crypto-equity proxies
    "COIN","MSTR","MARA","RIOT","CLSK","HUT","WULF",

    # ══ NSE / BSE (India) ═════════════════════════════════════════════════════
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","HINDUNILVR.NS",
    "ICICIBANK.NS","BHARTIARTL.NS","ITC.NS","KOTAKBANK.NS","LT.NS",
    "WIPRO.NS","BAJFINANCE.NS","HCLTECH.NS","SBIN.NS","AXISBANK.NS",
    "MARUTI.NS","SUNPHARMA.NS","TATAMOTORS.NS","ONGC.NS","NTPC.NS",

    # ══ LSE (UK) ══════════════════════════════════════════════════════════════
    "HSBA.L","BP.L","SHEL.L","AZN.L","GSK.L",
    "RIO.L","VOD.L","ULVR.L","LLOY.L","BARC.L",

    # ══ TSE (Japan) ═══════════════════════════════════════════════════════════
    "7203.T","6758.T","9984.T","8306.T","6861.T",
    "7974.T","4063.T","8035.T","4502.T","9432.T",

    # ══ HKEX (Hong Kong) ══════════════════════════════════════════════════════
    "9988.HK","0700.HK","9999.HK","1810.HK","3690.HK",
    "2318.HK","0941.HK","1299.HK","0005.HK","0939.HK",

    # ══ ASX (Australia) ═══════════════════════════════════════════════════════
    "CBA.AX","BHP.AX","NAB.AX","WBC.AX","ANZ.AX",
    "CSL.AX","WES.AX","MQG.AX","TLS.AX","FMG.AX",

    # ══ TSX (Canada) ══════════════════════════════════════════════════════════
    "RY.TO","TD.TO","BNS.TO","BMO.TO","CNR.TO",
    "CP.TO","SU.TO","ENB.TO","CNQ.TO","SHOP.TO",

    # ══ Xetra (Germany / EU) ══════════════════════════════════════════════════
    "SAP.DE","SIE.DE","BAYER.DE","ALV.DE","BMW.DE",
    "MBG.DE","VOW3.DE","MUV2.DE","BAS.DE","ADS.DE",

    # ══ Crypto Spot (24/7) ════════════════════════════════════════════════════
    "BTC-USD","ETH-USD","SOL-USD","BNB-USD","ADA-USD",
    "XRP-USD","DOGE-USD","AVAX-USD","DOT-USD","MATIC-USD",
]

# Asset → exchange mapping for market hours
ASSET_EXCHANGE_MAP = {}
def _build_exchange_map():
    rules = [
        ("US",  lambda t: not any(t.endswith(x) for x in [".NS",".L",".T",".HK",".AX",".TO",".DE","-USD"])),
        ("NSE", lambda t: t.endswith(".NS")),
        ("LSE", lambda t: t.endswith(".L")),
        ("TSE", lambda t: t.endswith(".T")),
        ("HKEX",lambda t: t.endswith(".HK")),
        ("ASX", lambda t: t.endswith(".AX")),
        ("TSX", lambda t: t.endswith(".TO")),
        ("XETRA",lambda t: t.endswith(".DE")),
        ("CRYPTO",lambda t: t.endswith("-USD")),
    ]
    for ticker in TRACKED_ASSETS:
        for exch, test in rules:
            if test(ticker):
                ASSET_EXCHANGE_MAP[ticker] = exch
                break

_build_exchange_map()

# Alias: for cache generation use PRELOADED_ASSETS; TRACKED_ASSETS = full catalogue
def get_preloaded_assets() -> list[str]:
    return PRELOADED_ASSETS[:]

N_STEPS = 200     # More iterations → better convergence
DT = 0.05         # Smaller step → more stable dynamics
COUPLING = 0.5    # coupling constant c

_cache: dict = {"signals": None, "generated_at": 0.0, "generating": False,
                "lock": threading.Lock()}
CACHE_TTL_SECONDS = 30  # 1-min live data freshness window


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

    By default only runs over PRELOADED_ASSETS (20 stocks). For on-demand
    single-asset signals use compute_single_asset() instead.
    """
    assets = assets or PRELOADED_ASSETS
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
        # Safe division: use np.where to avoid fake 100% returns when a price bar is 0.
        # np.nan_to_num converts any remaining NaN/Inf (e.g. consecutive 0-price bars) to 0.
        raw_close = close[-21:]
        prev_close = raw_close[:-1]
        rets = np.nan_to_num(
            np.diff(raw_close) / np.where(prev_close != 0, prev_close, np.nan),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
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
        # Check both TTL and that we actually have signal data (not just an empty dict)
        has_data = bool(_cache.get("signals") and _cache["signals"].get("signals"))
        if has_data and (now - _cache["generated_at"] < CACHE_TTL_SECONDS):
            return _cache["signals"]
        if _cache["generating"]:
            # Another thread is generating — return stale data to avoid pile-up
            if _cache.get("signals"):
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


# ---------------------------------------------------------------------------
# On-demand single-asset signal (real-time Yahoo Finance fetch)
# ---------------------------------------------------------------------------

_ondemand_cache: dict = {}   # ticker -> {signal, fetched_at}
_ONDEMAND_TTL = 30           # seconds before on-demand result expires


def compute_single_asset(ticker: str) -> dict | None:
    """Fetch a single asset from Yahoo Finance and compute its SBA signal.

    Results are cached for _ONDEMAND_TTL seconds so rapid user typing
    doesn't hammer the Yahoo Finance API.  Returns None if the ticker
    is invalid or has insufficient price history.
    """
    ticker = ticker.strip().upper()
    now = time.time()

    # Serve from on-demand cache if fresh
    cached = _ondemand_cache.get(ticker)
    if cached and (now - cached["fetched_at"]) < _ONDEMAND_TTL:
        return cached["signal"]

    # Also check the main preloaded cache
    with _cache["lock"]:
        main_snapshot = _cache.get("signals") or {}
    # _cache["signals"] stores the full signals response dict (not the list)
    for s in (main_snapshot.get("signals") or []):
        if s["asset"] == ticker:
            _ondemand_cache[ticker] = {"signal": s, "fetched_at": now}
            return s

    # Real-time Yahoo Finance fetch
    try:
        data = yf.download(
            [ticker], period="3mo", interval="1d",
            progress=False, auto_adjust=True,
        )
    except Exception as exc:
        log.warning("on-demand fetch failed for %s: %s", ticker, exc)
        return None

    if data is None or data.empty:
        return None

    close = _extract_close(data, ticker)
    if close is None or len(close) < 50:
        log.debug("Insufficient bars for on-demand ticker %s", ticker)
        return None

    feats = extract_features(close)
    # Single-asset: spin = tanh(momentum z-score) as simplified SBA proxy
    mom = feats["momentum"]
    spin = float(np.tanh(mom * 5.0))  # scale momentum → spin range [-1, 1]
    signal_type, confidence = score_signal(spin, feats["rsi"])

    result = {
        "asset": ticker,
        "signal_type": signal_type,
        "confidence": round(confidence, 4),
        "spin": round(spin, 4),
        "last_price": round(float(close[-1]), 2),
        "features": {k: round(v, 4) for k, v in feats.items()},
        "sba_iterations": 1,  # simplified single-asset mode
        "engine_version": "1.1.0-python-sba",
        "on_demand": True,
        "exchange": ASSET_EXCHANGE_MAP.get(ticker, _infer_exchange(ticker)),
    }

    _ondemand_cache[ticker] = {"signal": result, "fetched_at": now}
    return result


def _infer_exchange(ticker: str) -> str:
    """Infer exchange from ticker suffix for unknown tickers."""
    if ticker.endswith(".NS"): return "NSE"
    if ticker.endswith(".L"):  return "LSE"
    if ticker.endswith(".T"):  return "TSE"
    if ticker.endswith(".HK"): return "HKEX"
    if ticker.endswith(".AX"): return "ASX"
    if ticker.endswith(".TO"): return "TSX"
    if ticker.endswith(".DE"): return "XETRA"
    if ticker.endswith("-USD"): return "CRYPTO"
    return "US"
