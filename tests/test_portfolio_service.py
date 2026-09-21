"""Tests for backend/services/portfolio_service.py.

Covers recompute_positions() and equity_curve_from_trades(), including the
oversell edge case: the HTTP-layer guard in main.py (`quantity > held` ->
400) is a check-then-act race — held qty is checked, then the trade is
recorded — so a stale/concurrent fill can still slip an oversell into the
trade history. Both functions must handle that the same way (clamp to the
actually-held quantity) so the position book and the equity-curve/risk-metric
view of the world never disagree.
"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend import models
from backend.database import Base
from backend.services import portfolio_service


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


@pytest.fixture
def user(db):
    u = models.User(email="pnl@example.com", password_hash="x")
    db.add(u)
    db.commit()
    return u


def _add_trade(db, user_id, asset, side, qty, price):
    t = models.Trade(
        user_id=user_id, asset=asset, side=side, quantity=qty,
        filled_price=price, status="FILLED",
    )
    db.add(t)
    db.commit()
    return t


class TestRecomputePositions:
    def test_simple_buy_creates_position(self, db, user):
        _add_trade(db, user.id, "AAPL", "buy", 10, 100.0)
        portfolio_service.recompute_positions(db, user.id)
        rows = db.query(models.Position).filter_by(user_id=user.id).all()
        assert len(rows) == 1
        assert float(rows[0].quantity) == 10.0
        assert float(rows[0].avg_entry_price) == 100.0

    def test_buy_then_partial_sell_realizes_pnl(self, db, user):
        _add_trade(db, user.id, "AAPL", "buy", 10, 100.0)
        _add_trade(db, user.id, "AAPL", "sell", 4, 120.0)
        portfolio_service.recompute_positions(db, user.id)
        rows = db.query(models.Position).filter_by(user_id=user.id).all()
        assert len(rows) == 1
        assert float(rows[0].quantity) == 6.0
        assert float(rows[0].realized_pnl) == pytest.approx((120.0 - 100.0) * 4)

    def test_oversell_is_clamped_not_shorted(self, db, user):
        # A sell larger than the held quantity (simulating a race that
        # bypassed the HTTP-layer guard) must clamp to the held amount,
        # never produce a negative position.
        _add_trade(db, user.id, "AAPL", "buy", 5, 100.0)
        _add_trade(db, user.id, "AAPL", "sell", 20, 110.0)
        portfolio_service.recompute_positions(db, user.id)
        rows = db.query(models.Position).filter_by(user_id=user.id).all()
        # Flat position (5 - 5 = 0) — no row should exist.
        assert len(rows) == 0


class TestEquityCurveFromTrades:
    def test_buy_reduces_cash_then_marks_to_market(self, db, user, monkeypatch):
        monkeypatch.setattr(portfolio_service, "get_last_price", lambda asset: 150.0)
        _add_trade(db, user.id, "AAPL", "buy", 10, 100.0)
        curve = portfolio_service.equity_curve_from_trades(db, user.id, starting_capital=100_000.0)
        # cash = 100000 - 1000 = 99000; open value = 10*150 = 1500 -> 100500
        assert curve[-1] == pytest.approx(100_500.0)

    def test_oversell_does_not_create_phantom_short(self, db, user, monkeypatch):
        # Consistency with recompute_positions: an oversell trade must be
        # clamped to the held quantity here too, not credited in full and
        # left to push the book negative (a phantom short position that
        # would silently corrupt the equity curve and every risk metric
        # derived from it).
        monkeypatch.setattr(portfolio_service, "get_last_price", lambda asset: 100.0)
        _add_trade(db, user.id, "AAPL", "buy", 5, 100.0)
        _add_trade(db, user.id, "AAPL", "sell", 20, 110.0)
        curve = portfolio_service.equity_curve_from_trades(db, user.id, starting_capital=100_000.0)
        # Only 5 shares should ever be considered sold: cash = 100000 - 500 + 5*110 = 100050
        # Open value = 0 (flat). Final equity must NOT reflect the extra 15
        # "phantom" shares sold (which would give 100000 - 500 + 20*110 = 101700).
        assert curve[-1] == pytest.approx(100_050.0)
