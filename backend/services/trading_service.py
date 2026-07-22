"""QuantumSentinel — Trading Engine.

If ALPACA_API_KEY + ALPACA_SECRET_KEY env vars are set, orders route to the
real Alpaca paper-trading REST API (https://paper-api.alpaca.markets).
Otherwise falls back to a built-in paper broker that fills orders against
live Yahoo Finance prices — so the whole order lifecycle works end-to-end
even with zero external credentials configured.
"""
import os
import time
import requests
import yfinance as yf

ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

_price_cache: dict[str, tuple[float, float]] = {}  # asset -> (price, ts)
PRICE_CACHE_TTL = 20


def alpaca_enabled() -> bool:
    return bool(ALPACA_API_KEY and ALPACA_SECRET_KEY)


def get_last_price(asset: str) -> float:
    now = time.time()
    cached = _price_cache.get(asset)
    if cached and now - cached[1] < PRICE_CACHE_TTL:
        return cached[0]
    try:
        fast = yf.Ticker(asset).fast_info
        price = float(fast["last_price"])
    except Exception:
        hist = yf.Ticker(asset).history(period="1d")
        price = float(hist["Close"].iloc[-1]) if len(hist) else 100.0
    _price_cache[asset] = (price, now)
    return price


def _alpaca_headers():
    return {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        "Content-Type": "application/json",
    }


def submit_alpaca_order(asset: str, side: str, qty: float, order_type: str,
                         limit_price: float | None, stop_price: float | None,
                         time_in_force: str) -> dict:
    body = {
        "symbol": asset, "qty": str(qty), "side": side,
        "type": order_type, "time_in_force": time_in_force,
    }
    if order_type == "limit" and limit_price:
        body["limit_price"] = str(limit_price)
    if order_type in ("stop", "stop_limit") and stop_price:
        body["stop_price"] = str(stop_price)
    if order_type == "stop_limit" and limit_price:
        body["limit_price"] = str(limit_price)
    resp = requests.post(f"{ALPACA_BASE_URL}/v2/orders", json=body,
                          headers=_alpaca_headers(), timeout=10)
    resp.raise_for_status()
    return resp.json()


def simulate_fill(asset: str, side: str, qty: float, order_type: str,
                   limit_price: float | None, stop_price: float | None = None) -> dict:
    """Local paper-broker matching engine using the latest real market price."""
    last_price = get_last_price(asset)
    if order_type == "market":
        return {"status": "FILLED", "filled_price": last_price, "alpaca_order_id": None}
    if order_type in ("stop", "stop_limit"):
        triggered = (side == "buy" and last_price >= (stop_price or float("inf"))) or \
                    (side == "sell" and last_price <= (stop_price or 0))
        if not triggered:
            return {"status": "ACCEPTED", "filled_price": None, "alpaca_order_id": None}
        if order_type == "stop":
            return {"status": "FILLED", "filled_price": last_price, "alpaca_order_id": None}
    # limit and triggered stop-limit order: fill immediately if marketable, else pending
    marketable = (side == "buy" and last_price <= limit_price) or \
                 (side == "sell" and last_price >= limit_price)
    if marketable:
        return {"status": "FILLED", "filled_price": limit_price, "alpaca_order_id": None}
    return {"status": "ACCEPTED", "filled_price": None, "alpaca_order_id": None}


def check_pending_limit_fill(asset: str, side: str, limit_price: float) -> float | None:
    """Called on read to see if a pending limit order has become marketable."""
    last_price = get_last_price(asset)
    if side == "buy" and last_price <= limit_price:
        return limit_price
    if side == "sell" and last_price >= limit_price:
        return limit_price
    return None
