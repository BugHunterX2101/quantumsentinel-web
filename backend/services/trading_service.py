"""QuantumSentinel — market prices and the internal paper broker's fill rules.

Execution is paper-only by construction: this codebase contains no external
broker client, so there is no configuration that can route an order to a
real-money venue. Orders fill against live Yahoo Finance prices, and the cash
ledger is owned by ``paper_broker``.
"""
import math
import time

import numpy as np
import yfinance as yf

_price_cache: dict[str, tuple[float, float]] = {}  # asset -> (price, fetched_at)
PRICE_CACHE_TTL = 20


class MarketDataUnavailable(Exception):
    """No trustworthy price could be obtained. Callers must reject, never guess."""


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


def get_last_price(asset: str) -> float:
    """Latest executable price, at most PRICE_CACHE_TTL seconds old.

    Raises MarketDataUnavailable when no finite positive price exists.
    """
    now = time.time()
    cached = _price_cache.get(asset)
    if cached and now - cached[1] < PRICE_CACHE_TTL:
        return cached[0]
    price = _fast_info_price(asset) or _last_close(asset)
    if price is None:
        raise MarketDataUnavailable(f"no market price available for {asset}")
    _price_cache[asset] = (price, now)
    return price


def get_mark_price(asset: str) -> tuple[float | None, bool]:
    """Best available price for valuation only — never for execution.

    Returns (price, is_stale). Falls back to the last cached observation of
    any age when a fresh fetch fails, and to (None, True) when nothing was
    ever observed.
    """
    try:
        return get_last_price(asset), False
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
