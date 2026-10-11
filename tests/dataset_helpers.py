"""Yahoo-shaped history frames and metadata, so provider tests need no network.

The shapes copy what yfinance 1.5.2 returns from ``Ticker.history(...,
auto_adjust=False, actions=True)``: a DatetimeIndex at local midnight in the
exchange's timezone, the columns below, and ``get_history_metadata()`` keys.
"""
import datetime as dt

import numpy as np
import pandas as pd

from backend.services.dataset_providers import SnapshotRequest, YahooProvider

META = {
    "AAPL": {"instrumentType": "EQUITY", "exchangeName": "NMS", "exchangeTimezoneName": "America/New_York",
             "currency": "USD"},
    "VOD.L": {"instrumentType": "EQUITY", "exchangeName": "LSE", "exchangeTimezoneName": "Europe/London",
              "currency": "GBp"},
    "SPY": {"instrumentType": "ETF", "exchangeName": "PCX", "exchangeTimezoneName": "America/New_York",
            "currency": "USD"},
    "BTC-USD": {"instrumentType": "CRYPTOCURRENCY", "exchangeName": "CCC", "exchangeTimezoneName": "UTC",
                "currency": "USD"},
    "EURUSD=X": {"instrumentType": "CURRENCY", "exchangeName": "CCY",
                 "exchangeTimezoneName": "Europe/London", "currency": "USD"},
}
# Fixed "now" for snapshots: 2024-01-10 15:00 UTC, mid-session in New York.
NOW = dt.datetime(2024, 1, 10, 15, 0, tzinfo=dt.timezone.utc)


def frame(symbol, days, closes=None, tz=None, **columns):
    """A history frame for ``days`` (ISO dates) at local midnight."""
    tz = tz or META[symbol]["exchangeTimezoneName"]
    index = pd.DatetimeIndex([pd.Timestamp(d).tz_localize(tz) for d in days], name="Date")
    closes = list(closes if closes is not None else [100.0 + i for i in range(len(days))])
    data = {"Open": [c - 0.5 for c in closes], "High": [c + 1.0 for c in closes],
            "Low": [c - 1.0 for c in closes], "Close": closes, "Adj Close": closes,
            "Volume": [1_000_000.0] * len(days), "Dividends": [0.0] * len(days),
            "Stock Splits": [0.0] * len(days)}
    data.update(columns)
    return pd.DataFrame(data, index=index)


WEEK = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10"]


def default_frames():
    return {
        "AAPL": frame("AAPL", WEEK, Dividends=[0, 0, 0.24, 0, 0, 0, 0],
                      **{"Stock Splits": [0, 0, 0, 0, 4.0, 0, 0]}),
        "VOD.L": frame("VOD.L", WEEK, closes=[70.1, 70.2, np.nan, 70.4, 70.5, 70.6, 70.7]),
        "BTC-USD": frame("BTC-USD", WEEK, closes=[45000.0 + i for i in range(7)]),
        "EURUSD=X": frame("EURUSD=X", WEEK, closes=[1.09 + i / 1000 for i in range(7)], Volume=[0.0] * 7),
    }


class FakeYahoo:
    """A fetch function serving fixed frames, recording each call."""

    def __init__(self, frames=None, meta=None):
        self.frames = default_frames() if frames is None else frames
        self.meta = META if meta is None else meta
        self.calls = []

    def __call__(self, symbol, start, end):
        self.calls.append((symbol, start, end))
        return self.frames.get(symbol, pd.DataFrame()), self.meta.get(symbol, {})


def provider(frames=None, meta=None, now=NOW):
    return YahooProvider(fetch=FakeYahoo(frames, meta), client="yfinance test", clock=lambda: now)


def snapshot(symbols=("AAPL", "BTC-USD"), **kwargs):
    request = SnapshotRequest(tuple(symbols), dt.date(2024, 1, 1), dt.date(2024, 1, 11))
    return provider(**kwargs).snapshot(request)
