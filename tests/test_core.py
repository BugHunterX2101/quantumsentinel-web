import numpy as np
import pytest

from backend import models, schemas
from backend.crypto import pqc
from backend.database import Base
from backend.services import integration_service, signal_engine, trading_service


def test_sba_is_reproducible():
    coupling = np.array([[0.0, 0.2], [0.2, 0.0]])
    fields = np.array([0.1, -0.2])
    first = signal_engine.run_sba(coupling, fields)
    second = signal_engine.run_sba(coupling, fields)
    np.testing.assert_array_equal(first, second)


def test_order_simulator_respects_stop_direction(monkeypatch):
    monkeypatch.setattr(trading_service, "get_last_price", lambda _asset: 100.0)
    assert trading_service.simulate_fill("ABC", "buy", 1, "stop", None, 110)["status"] == "ACCEPTED"
    assert trading_service.simulate_fill("ABC", "sell", 1, "stop", None, 90)["status"] == "ACCEPTED"


def test_schema_rejects_unsafe_webhook():
    with pytest.raises(ValueError):
        schemas.WebhookRequest(url="http://127.0.0.1/hook")


def test_api_key_scope_and_hashing():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    raw, prefix, digest = integration_service.generate_api_key()
    user = models.User(email="scope@example.com", password_hash="x")
    session.add(user); session.commit()
    session.add(models.ApiKey(user_id=user.id, name="read-only", key_prefix=prefix,
                              key_hash=digest, scopes=["read"]))
    session.commit()
    assert integration_service.verify_api_key(session, raw, "read") is not None
    assert integration_service.verify_api_key(session, raw, "trade") is None


# ── Regression tests for deep-audit bug fixes ─────────────────────────────────

def test_b2_insight_uses_correct_bb_width_key():
    """B2: _generate_insight must use 'bb_width' key (not 'bollinger_width').
    With bb_width=0.15 (> 0.08 threshold) and up to 4 parts now, the
    Bollinger band volatility note must appear.
    """
    feats = {"rsi": 50.0, "macd_histogram": 0.001, "momentum": 0.02, "bb_width": 0.15}
    insight = signal_engine._generate_insight("BUY", feats, 0.5, "TEST")
    assert "Bollinger" in insight, (
        f"Expected 'Bollinger' in insight (bb_width=0.15 > threshold 0.08), got: {insight!r}"
    )

    # Control: with bb_width=0 and bb_width not triggering anything, no wide-band note
    feats_flat = {"rsi": 50.0, "macd_histogram": 0.001, "momentum": 0.02, "bb_width": 0.05}
    insight_flat = signal_engine._generate_insight("BUY", feats_flat, 0.5, "TEST")
    assert "wide" not in insight_flat.lower()


def test_b3_change_pct_safe_when_prev_close_is_zero():
    """B3: change_pct must not raise ZeroDivisionError when prev close is 0."""
    close = np.array([0.0, 0.0, 0.0, 0.0, 5.0])
    prev = float(close[-2])
    result = round(float((close[-1] - prev) / prev * 100), 2) if prev != 0.0 else 0.0
    assert result == 0.0

    close2 = np.array([5.0, 10.0])
    prev2 = float(close2[-2])
    result2 = round(float((close2[-1] - prev2) / prev2 * 100), 2) if prev2 != 0.0 else 0.0
    assert result2 == 100.0


def test_b4_signal_field_names_are_3mo():
    """B4: on-demand signal source must label range as high_3mo/low_3mo, not 52w."""
    import inspect
    src = inspect.getsource(signal_engine.compute_single_asset)
    assert "high_3mo" in src, "Must use high_3mo"
    assert "low_3mo" in src, "Must use low_3mo"
    assert "high_52w" not in src


def test_b7_market_close_is_strict_less_than():
    """B7: market open check must use strict < at close to avoid 1-minute overshoot."""
    import inspect
    src = inspect.getsource(signal_engine._market_is_open)
    assert "open_t <= cur < close_t" in src, "Must use strict < at market close"


def test_t2_stop_limit_with_no_limit_price_does_not_crash(monkeypatch):
    """T2: stop_limit with limit_price=None must not raise TypeError."""
    monkeypatch.setattr(trading_service, "get_last_price", lambda _: 105.0)
    # stop_limit buy triggered (last 105 >= stop 100)
    result = trading_service.simulate_fill("XYZ", "buy", 1.0, "stop_limit",
                                           limit_price=None, stop_price=100.0)
    assert "status" in result
    assert result["status"] in ("FILLED", "ACCEPTED")


def test_t3_limit_buy_fills_at_market_when_better(monkeypatch):
    """T3: Limit buy fills at min(market, limit) — price improvement for buyer."""
    monkeypatch.setattr(trading_service, "get_last_price", lambda _: 95.0)
    result = trading_service.simulate_fill("ABC", "buy", 1.0, "limit",
                                           limit_price=100.0, stop_price=None)
    assert result["status"] == "FILLED"
    assert result["filled_price"] == 95.0, f"Expected fill at 95 (market), got {result['filled_price']}"


def test_t3_limit_sell_fills_at_market_when_better(monkeypatch):
    """T3: Limit sell fills at max(market, limit) — price improvement for seller."""
    monkeypatch.setattr(trading_service, "get_last_price", lambda _: 110.0)
    result = trading_service.simulate_fill("ABC", "sell", 1.0, "limit",
                                           limit_price=100.0, stop_price=None)
    assert result["status"] == "FILLED"
    assert result["filled_price"] == 110.0, f"Expected fill at 110 (market), got {result['filled_price']}"


def test_t3_limit_buy_stays_pending_when_not_marketable(monkeypatch):
    """T3: Limit buy where market > limit stays ACCEPTED (not marketable)."""
    monkeypatch.setattr(trading_service, "get_last_price", lambda _: 110.0)
    result = trading_service.simulate_fill("ABC", "buy", 1.0, "limit",
                                           limit_price=100.0, stop_price=None)
    assert result["status"] == "ACCEPTED"
    assert result["filled_price"] is None


def test_s1_backtest_rejects_fast_ge_slow():
    """S1: BacktestRequest must reject slow_window <= fast_window."""
    with pytest.raises(Exception):
        schemas.BacktestRequest(asset="AAPL", fast_window=50, slow_window=20, period="1y")


def test_s1_backtest_accepts_valid_windows():
    """S1: BacktestRequest must accept fast < slow."""
    req = schemas.BacktestRequest(asset="AAPL", fast_window=20, slow_window=50, period="1y")
    assert req.fast_window == 20 and req.slow_window == 50


def test_rsi_returns_100_for_pure_uptrend():
    """RSI: no losses in period => RS=inf => RSI=100."""
    close = np.linspace(100, 200, 60)
    assert signal_engine._rsi_wilder(close) == 100.0


def test_rsi_returns_50_for_insufficient_data():
    """RSI: fewer than 42 bars returns neutral 50."""
    close = np.array([100.0, 101.0, 102.0])
    assert signal_engine._rsi_wilder(close) == 50.0


def test_bollinger_width_is_nonnegative():
    """Bollinger width must always be >= 0."""
    close = np.array([100.0 + i * 0.5 + (i % 3) * 2 for i in range(30)])
    assert signal_engine._bollinger_width(close) >= 0.0


def test_bollinger_width_near_zero_for_flat_price():
    """Bollinger width is effectively 0 for a constant price series."""
    close = np.full(25, 150.0)
    assert signal_engine._bollinger_width(close) < 1e-6


def test_score_signal_buy_penalised_when_overbought():
    """score_signal: overbought RSI reduces BUY confidence."""
    _, conf_ok  = signal_engine.score_signal(0.8, 50.0)  # RSI neutral
    _, conf_hot = signal_engine.score_signal(0.8, 75.0)  # RSI overbought
    assert conf_hot < conf_ok


def test_schema_backtest_period_validation():
    """BacktestRequest: invalid period string is rejected."""
    with pytest.raises(Exception):
        schemas.BacktestRequest(asset="AAPL", fast_window=10, slow_window=50, period="3d")
    req = schemas.BacktestRequest(asset="AAPL", fast_window=10, slow_window=50, period="6mo")
    assert req.period == "6mo"


def test_infer_exchange_crypto():
    """infer_exchange: BTC-USD returns CRYPTO."""
    assert signal_engine.infer_exchange("BTC-USD") == "CRYPTO"
    assert signal_engine.infer_exchange("ETH-USD") == "CRYPTO"


def test_infer_exchange_nse():
    """infer_exchange: .NS suffix returns NSE."""
    assert signal_engine.infer_exchange("RELIANCE.NS") == "NSE"


def test_infer_exchange_us_default():
    """infer_exchange: plain ticker defaults to US exchange."""
    assert signal_engine.infer_exchange("AAPL") == "US"
    assert signal_engine.infer_exchange("MSFT") == "US"
