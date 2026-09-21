"""Regression test for backend/main.py:_fetch_single_asset.

Found via live testing (Playwright + direct API smoke test) of
POST /api/research/regime, which crashed on every real request with
"operands could not be broadcast together with shapes (753,0) (752,1)".

Root cause: for a single-ticker yfinance download (yf.download(<str>, ...)),
data["Close"] is a 1-column DataFrame in this yfinance version, not a
Series — unlike every other yfinance call site in the codebase, which
always passes a list (even for one ticker) and yfinance always returns
MultiIndex columns for a list, collapsing cleanly to a Series per ticker.
close.to_numpy(dtype=float) on the 1-column DataFrame produced shape
(T, 1) instead of (T,), so np.diff(prices) (default axis=-1, size 1)
diffed the trivial column axis instead of time, yielding shape (T, 0)
instead of (T-1,) — which could never broadcast against prices[:-1]
(shape (T-1, 1)).

backend/services/regime_detection.py itself is correctly implemented and
already has thorough coverage in tests/test_phase3.py (all of it exercised
with proper 1-D arrays) — the bug lived entirely in this data-fetching glue
function, which had no coverage anywhere since every other test constructs
its return/price arrays directly rather than going through this endpoint
boundary.
"""
import numpy as np
import pandas as pd
import pytest


def _fake_yf_download_single_ticker(n=120, ticker="SPY"):
    """Reproduces the real yfinance shape: a MultiIndex-columned frame where
    selecting the "Close" field yields a 1-column DataFrame, not a Series."""
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    prices = 100.0 + np.cumsum(np.random.default_rng(1).normal(0, 1, n))
    columns = pd.MultiIndex.from_tuples(
        [("Close", ticker), ("High", ticker), ("Low", ticker),
         ("Open", ticker), ("Volume", ticker)],
        names=["Price", "Ticker"],
    )
    data = pd.DataFrame(
        {("Close", ticker): prices, ("High", ticker): prices * 1.01,
         ("Low", ticker): prices * 0.99, ("Open", ticker): prices,
         ("Volume", ticker): 1_000_000.0},
        index=idx, columns=columns,
    )
    return data


class TestFetchSingleAsset:
    def test_returns_and_prices_are_1d(self, monkeypatch):
        import yfinance
        from backend import main

        fake_data = _fake_yf_download_single_ticker(n=120)
        assert fake_data["Close"].shape == (120, 1), (
            "fixture must reproduce the real 1-column-DataFrame shape "
            "yfinance returns for a single-ticker download in this version"
        )
        monkeypatch.setattr(yfinance, "download", lambda *a, **kw: fake_data)

        returns, prices = main._fetch_single_asset("SPY", "6mo")

        assert prices.ndim == 1
        assert returns.ndim == 1
        assert prices.shape == (120,)
        assert returns.shape == (119,)
        assert np.all(np.isfinite(returns))

    def test_does_not_raise_broadcast_error(self, monkeypatch):
        """The exact failure mode observed live: a shape-(T,0) vs (T-1,1)
        broadcast error inside np.diff(prices) / prices[:-1]."""
        import yfinance
        from backend import main

        fake_data = _fake_yf_download_single_ticker(n=753)
        monkeypatch.setattr(yfinance, "download", lambda *a, **kw: fake_data)

        try:
            returns, prices = main._fetch_single_asset("SPY", "3y")
        except ValueError as exc:
            pytest.fail(f"_fetch_single_asset raised unexpectedly: {exc}")

        assert len(returns) == len(prices) - 1
