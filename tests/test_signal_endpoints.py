"""Regression tests for GET /api/signals/latest (backend/main.py:latest_signals).

Two bugs found via live browser testing (Playwright), neither with prior
unit coverage since the whole request/response cycle is skipped by every
other test in this suite:

1. Watchlist gap-filling: get_cached_signals() only ever computes the 20
   preloaded TRACKED_ASSETS, so a ticker a user watchlisted via search (the
   entire point of the watchlist feature) was silently absent from every
   response — it never appeared on reload, poll, or the next WebSocket push.
2. Stale ETag: the ETag was hashed from the shared cache's generation
   timestamp and raw asset count only, never from the caller's resolved
   watchlist/exchange-filter set. A browser's fetch() automatically resends
   a previously-seen ETag as If-None-Match, so after a user edited their
   watchlist or exchange preferences the server kept recognising the old
   (unchanged) tag and answered 304, silently serving the pre-edit result
   until the shared cache happened to regenerate on its own ~30-60s cycle.

Matches the established pattern in tests/test_order_security.py: import
backend.main and call the endpoint function directly against an in-memory
SQLite user, monkeypatching the network-touching signal_engine calls
instead of hitting yfinance.
"""
import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend import models
from backend.database import Base
from backend.services import signal_engine


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


class _FakeRequest:
    """Duck-typed stand-in for starlette.Request — latest_signals() only
    ever calls request.headers.get(...), so a full ASGI Request is
    unnecessary."""
    def __init__(self, headers=None):
        self.headers = headers or {}


def _preloaded_cache():
    return {
        "signals": [
            {"asset": "AAPL", "signal_type": "BUY", "confidence": 0.8, "last_price": 190.0},
            {"asset": "MSFT", "signal_type": "HOLD", "confidence": 0.5, "last_price": 410.0},
        ],
        "generated_at": 1_700_000_000.0,
        "pipeline_ms": 10, "sba_ms": 5, "n_assets": 2,
    }


def _make_user(db, watchlist, preferred_exchanges):
    user = models.User(email=f"wl-{id(watchlist)}@example.com", password_hash="x",
                        watchlist=watchlist, preferred_exchanges=preferred_exchanges)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


class TestWatchlistGapFilling:
    def test_watchlisted_ticker_outside_preloaded_cache_is_included(self, db, monkeypatch):
        from backend import main
        monkeypatch.setattr(signal_engine, "get_cached_signals", lambda: _preloaded_cache())
        monkeypatch.setattr(
            signal_engine, "compute_single_asset",
            lambda t: {"asset": t, "signal_type": "BUY", "confidence": 0.6, "last_price": 50.0}
            if t == "RELIANCE.NS" else None,
        )
        user = _make_user(db, ["AAPL", "RELIANCE.NS"], ["US", "NSE"])

        result = main.latest_signals(_FakeRequest(), assets=None, user=user)

        body = json.loads(result.body)
        assets = [s["asset"] for s in body["signals"]]
        assert "RELIANCE.NS" in assets, "watchlisted ticker outside the preloaded cache must still appear"
        assert body["n_assets"] == 2

    def test_non_watchlisted_asset_not_pulled_in(self, db, monkeypatch):
        """compute_single_asset() must only be consulted for tickers actually
        in `wanted` — it should not silently expand the response."""
        from backend import main
        monkeypatch.setattr(signal_engine, "get_cached_signals", lambda: _preloaded_cache())
        calls = []

        def fake_compute(t):
            calls.append(t)
            return None

        monkeypatch.setattr(signal_engine, "compute_single_asset", fake_compute)
        user = _make_user(db, ["AAPL"], ["US"])

        main.latest_signals(_FakeRequest(), assets=None, user=user)

        assert calls == [], "no gap-filling should occur when the watchlist is a subset of the cache"


class TestEtagReflectsPersonalizedFilter:
    def test_etag_rejects_stale_tag_after_watchlist_edit(self, db, monkeypatch):
        from backend import main
        monkeypatch.setattr(signal_engine, "get_cached_signals", lambda: _preloaded_cache())
        monkeypatch.setattr(signal_engine, "compute_single_asset", lambda t: None)
        user = _make_user(db, ["AAPL"], ["US"])

        r1 = main.latest_signals(_FakeRequest(), assets=None, user=user)
        etag1 = r1.headers["ETag"]

        # User adds MSFT to their watchlist — the shared cache's
        # generated_at is unchanged, only the caller's filter set changed.
        user.watchlist = ["AAPL", "MSFT"]
        db.commit()

        # A browser's fetch() would resend the old ETag automatically.
        r2 = main.latest_signals(_FakeRequest(headers={"if-none-match": etag1}), assets=None, user=user)

        assert r2.status_code != 304, "a watchlist edit must invalidate the previous ETag"
        body2 = json.loads(r2.body)
        assert body2["n_assets"] == 2
        assert {s["asset"] for s in body2["signals"]} == {"AAPL", "MSFT"}

    def test_matching_etag_with_unchanged_watchlist_still_returns_304(self, db, monkeypatch):
        """The bandwidth-saving fast path must keep working for the common
        case — repeated polling with no watchlist/exchange changes."""
        from backend import main
        monkeypatch.setattr(signal_engine, "get_cached_signals", lambda: _preloaded_cache())
        monkeypatch.setattr(signal_engine, "compute_single_asset", lambda t: None)
        user = _make_user(db, ["AAPL"], ["US"])

        r1 = main.latest_signals(_FakeRequest(), assets=None, user=user)
        etag1 = r1.headers["ETag"]

        r2 = main.latest_signals(_FakeRequest(headers={"if-none-match": etag1}), assets=None, user=user)

        assert r2.status_code == 304
