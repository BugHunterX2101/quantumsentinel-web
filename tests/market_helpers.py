"""Research market data from fixed frames instead of the network.

``fake_market`` answers the provider's fetch (dataset_providers._yahoo_fetch)
from a yf.download-shaped frame — what these tests used to return from a
faked ``yf.download`` — and gives the test a dataset root of its own. A task
then snapshots, stores and (as a job) registers that data exactly as in
production (services/market_data.py).
"""
import datetime as dt

import numpy as np
import pandas as pd

from backend import config
from backend.services import dataset_providers, market_data
from dataset_helpers import FakeYahoo

EQUITY = {"instrumentType": "EQUITY", "exchangeName": "NMS", "exchangeTimezoneName": "America/New_York",
          "currency": "USD"}
CRYPTO = {"instrumentType": "CRYPTOCURRENCY", "exchangeName": "CCC", "exchangeTimezoneName": "UTC",
          "currency": "USD"}


def history_frames(data: pd.DataFrame, crypto=()) -> dict[str, pd.DataFrame]:
    """Each ticker of a yf.download-shaped frame as Ticker.history(...,
    auto_adjust=False, actions=True) would return it, with no dividends (so
    its adjusted prices are the frame's)."""
    if data is None or data.empty:
        return {}
    fields = set(data.columns.get_level_values(0))
    frames = {}
    for ticker in dict.fromkeys(data.columns.get_level_values(1)):
        def field(name):
            column = data[name if name in fields else "Close"][ticker]
            return pd.to_numeric(column, errors="coerce").to_numpy(dtype=float)
        tz = (CRYPTO if ticker in crypto else EQUITY)["exchangeTimezoneName"]
        close = field("Close")
        volume = field("Volume") if "Volume" in fields else np.full(len(close), 1e6)
        frames[ticker] = pd.DataFrame({
            "Open": field("Open"), "High": field("High"), "Low": field("Low"), "Close": close,
            "Adj Close": close, "Volume": volume, "Dividends": 0.0, "Stock Splits": 0.0},
            index=pd.DatetimeIndex(data.index).tz_localize(tz))
    return frames


def fake_market(monkeypatch, tmp_path, data: pd.DataFrame, crypto=(), now: dt.datetime | None = None):
    """Serve ``data`` as the provider's data; returns the fake fetch. "Now"
    defaults to two days after the frame's last date, so every row is a
    completed session."""
    frames = history_frames(data, crypto)
    fake = FakeYahoo(frames, {t: CRYPTO if t in crypto else EQUITY for t in frames})
    monkeypatch.setattr(dataset_providers, "_yahoo_fetch", fake)
    monkeypatch.setattr(dataset_providers, "_yahoo_client", lambda: "yfinance test")
    monkeypatch.setattr(config, "DATASET_ROOT", tmp_path / "datasets")
    if now is None:
        last = pd.DatetimeIndex(data.index).max() if frames else pd.Timestamp("2024-01-01")
        now = dt.datetime.combine(last.date() + dt.timedelta(days=2), dt.time(12), dt.timezone.utc)
    monkeypatch.setattr(market_data, "utcnow", lambda: now)
    # The production provider, now fetching from ``fake`` (conftest blocks the network).
    monkeypatch.setattr(market_data, "default_provider",
                        lambda: dataset_providers.YahooProvider(clock=lambda: market_data.utcnow()))
    return fake
