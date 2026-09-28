"""QuantumSentinel — market prices and the internal paper broker's fill rules.

Execution is paper-only by construction: this codebase contains no external
broker client, so there is no configuration that can route an order to a
real-money venue. Orders fill against live Yahoo Finance prices, and the cash
ledger is owned by ``paper_broker``.

An executable price is a real trade no older than MAX_QUOTE_AGE_SECONDS by
the exchange's own trade timestamp. A closed market's last close is still a
valid valuation mark, but never an execution price.
"""
import math
import time

import numpy as np
import yfinance as yf

from ..config import MAX_QUOTE_AGE_SECONDS

# asset -> (price, fetched_at, market_time): market_time is the exchange
# timestamp (epoch seconds) of the trade, or None when the source gives none.
_price_cache: dict[str, tuple[float, float, float | None]] = {}
PRICE_CACHE_TTL = 20


class MarketDataUnavailable(Exception):
    """No trustworthy price could be obtained. Callers must reject, never guess."""


def _human_age(seconds: float) -> str:
    if seconds < 3600:
        return f"{seconds / 60:.0f} minutes"
    if seconds < 172_800:
        return f"{seconds / 3600:.1f} hours"
    return f"{seconds / 86_400:.1f} days"


class StaleMarketData(MarketDataUnavailable):
    """A real price exists but its last trade is too old, or of unknown time,
    to execute against (typically: the market is closed). ``price`` is still
    a valid valuation mark."""

    def __init__(self, asset: str, price: float, market_time: float | None, max_age: float):
        self.asset = asset
        self.price = price
        self.market_time = market_time
        self.age_seconds = None if market_time is None else max(0.0, time.time() - market_time)
        detail = ("the time of its last trade is unknown" if self.age_seconds is None
                  else f"its last trade was {_human_age(self.age_seconds)} ago")
        super().__init__(f"{asset} is not trading now: {detail}; orders execute only against "
                         f"a quote less than {_human_age(max_age)} old")


def _valid_price(value) -> float | None:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(price) and price > 0):
        return None
    # Yahoo delivers many prices as float32 widened to float64, e.g. 341.07
    # arrives as 341.0700073242. When a value is exactly float32-representable,
    # its shortest float32 repr is the decimal the provider meant; genuine
    # float64 prices are left untouched.
    as_f32 = np.float32(price)
    if float(as_f32) == price:
        price = float(np.format_float_positional(as_f32, unique=True))
    return price


def _fast_info_price(asset: str) -> float | None:
    try:
        fast = yf.Ticker(asset).fast_info
    except Exception:
        return None
    # fast_info is an object, not a dict; individual attributes can raise.
    for attr in ("last_price", "regularMarketPrice"):
        try:
            price = _valid_price(getattr(fast, attr, None))
        except Exception:
            price = None
        if price is not None:
            return price
    return None


def _last_close(asset: str) -> float | None:
    try:
        hist = yf.Ticker(asset).history(period="5d")
    except Exception:
        return None
    if not len(hist):
        return None
    return _valid_price(hist["Close"].iloc[-1])


def _live_quote(asset: str) -> tuple[float, float] | None:
    """Latest regular-session trade: (price, exchange trade time in epoch s)."""
    try:
        ticker = yf.Ticker(asset)
        bars = ticker.history(period="1d", interval="1m")
        meta = getattr(ticker, "history_metadata", None) or {}
    except Exception:
        return None
    price = _valid_price(meta.get("regularMarketPrice"))
    market_time = meta.get("regularMarketTime")
    if price is not None and isinstance(market_time, (int, float)) and market_time > 0:
        return price, float(market_time)
    # No metadata: the last one-minute bar carries both price and time.
    try:
        closes = bars["Close"].dropna()
    except Exception:
        return None
    if len(closes):
        price = _valid_price(closes.iloc[-1])
        stamp = closes.index[-1]
        if price is not None and hasattr(stamp, "timestamp"):
            return price, float(stamp.timestamp())
    return None


def _quote(asset: str) -> tuple[float, float | None]:
    """(price, market_time), fetched at most every PRICE_CACHE_TTL seconds.

    Raises MarketDataUnavailable when no finite positive price exists.
    """
    now = time.time()
    cached = _price_cache.get(asset)
    if cached and now - cached[1] < PRICE_CACHE_TTL:
        return cached[0], cached[2]
    live = _live_quote(asset)
    if live is not None:
        price, market_time = live
    else:
        # A price without a trade time: good for valuation, never executable.
        price, market_time = _fast_info_price(asset) or _last_close(asset), None
        if price is None:
            raise MarketDataUnavailable(f"no market price available for {asset}")
    _price_cache[asset] = (price, now, market_time)
    return price, market_time


def get_last_price(asset: str) -> float:
    """Latest executable price: a real trade at most MAX_QUOTE_AGE_SECONDS old.

    Raises StaleMarketData when the last trade is older than that or its time
    is unknown, and MarketDataUnavailable when no finite positive price exists.
    """
    price, market_time = _quote(asset)
    if market_time is None or time.time() - market_time > MAX_QUOTE_AGE_SECONDS:
        raise StaleMarketData(asset, price, market_time, MAX_QUOTE_AGE_SECONDS)
    return price


def get_mark_price(asset: str) -> tuple[float | None, bool]:
    """Best available price for valuation only — never for execution.

    Returns (price, is_stale). A closed market's last trade is a stale but
    valid mark; after a failed fetch it falls back to the last cached
    observation of any age, and to (None, True) when nothing was ever
    observed.
    """
    try:
        return get_last_price(asset), False
    except StaleMarketData as exc:
        return exc.price, True
    except MarketDataUnavailable:
        cached = _price_cache.get(asset)
        return (cached[0], True) if cached else (None, True)


def simulate_fill(asset: str, side: str, qty: float, order_type: str,
                   limit_price: float | None, stop_price: float | None = None,
                   last_price: float | None = None) -> dict:
    """Paper-broker fill rule against the latest real market price."""
    if last_price is None:
        last_price = get_last_price(asset)
    if order_type == "market":
        return {"status": "FILLED", "filled_price": last_price}
    if order_type in ("stop", "stop_limit"):
        triggered = (side == "buy" and last_price >= (stop_price or float("inf"))) or \
                    (side == "sell" and last_price <= (stop_price or 0))
        if not triggered:
            return {"status": "ACCEPTED", "filled_price": None}
        if order_type == "stop":
            return {"status": "FILLED", "filled_price": last_price}
    if limit_price is None:
        return {"status": "FILLED", "filled_price": last_price}
    marketable = (side == "buy" and last_price <= limit_price) or \
                 (side == "sell" and last_price >= limit_price)
    if marketable:
        # Price improvement: a buy never pays above its limit, a sell never
        # receives below it, and both get the better market price when available.
        fill_px = min(last_price, limit_price) if side == "buy" else max(last_price, limit_price)
        return {"status": "FILLED", "filled_price": round(fill_px, 6 if fill_px < 1 else 2)}
    return {"status": "ACCEPTED", "filled_price": None}


def check_pending_limit_fill(asset: str, side: str, limit_price: float,
                             last_price: float | None = None) -> float | None:
    """Fill price for a resting limit order if it is now marketable, else None."""
    if last_price is None:
        last_price = get_last_price(asset)
    if side == "buy" and last_price <= limit_price:
        return round(min(last_price, limit_price), 6 if last_price < 1 else 2)
    if side == "sell" and last_price >= limit_price:
        return round(max(last_price, limit_price), 6 if last_price < 1 else 2)
    return None
