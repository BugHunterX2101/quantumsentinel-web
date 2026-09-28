"""Server-side paper ledger: strict pricing, atomic cash reservation, fills.

These drive the real order route (backend.main.place_order) against an
in-memory database with a controllable market price.
"""
import threading

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend import main, models, schemas
from backend.database import Base
from backend.services import order_security, paper_broker, portfolio_service, trading_service


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


@pytest.fixture
def user(db):
    u = models.User(email="ledger@example.com", password_hash="x")
    db.add(u)
    db.commit()
    return u


@pytest.fixture
def market(monkeypatch):
    """Controllable last price per asset; missing asset = no market data."""
    prices: dict[str, float] = {}

    def last_price(asset):
        if asset not in prices:
            raise trading_service.MarketDataUnavailable(asset)
        return prices[asset]

    monkeypatch.setattr(trading_service, "get_last_price", last_price)
    trading_service._price_cache.clear()
    return prices


@pytest.fixture
def uncapped(monkeypatch):
    """Lift the per-asset concentration cap to test pure cash limits."""
    monkeypatch.setattr(main, "PAPER_MAX_POSITION_FRACTION", 1.0)
    monkeypatch.setattr(order_security, "PAPER_MAX_POSITION_FRACTION", 1.0)


def order(db, user, key=None, **fields):
    req = schemas.OrderRequest(**{"asset": "AAPL", "side": "buy", "quantity": 1,
                                  "order_type": "market", **fields})
    return main.place_order(req, user, db, key)


def account(db, user):
    return main.portfolio_account(user, db)


class TestStrictMarketData:
    def test_no_price_rejects_order_and_creates_nothing(self, db, user, market):
        with pytest.raises(HTTPException) as exc:
            order(db, user)
        assert exc.value.status_code == 503
        assert db.query(models.Trade).count() == 0

    def test_get_last_price_never_fabricates_a_price(self, monkeypatch):
        import pandas as pd

        class NoData:
            def __init__(self, _sym):
                self.fast_info = type("Info", (), {"last_price": None, "regularMarketPrice": None})()

            def history(self, **_kw):
                return pd.DataFrame()

        monkeypatch.setattr(trading_service.yf, "Ticker", NoData)
        trading_service._price_cache.clear()
        with pytest.raises(trading_service.MarketDataUnavailable):
            trading_service.get_last_price("NODATA")

    def test_non_finite_or_non_positive_prices_are_rejected(self, monkeypatch):
        import pandas as pd

        class Bad:
            def __init__(self, _sym):
                self.fast_info = type("Info", (), {"last_price": float("nan"), "regularMarketPrice": -3})()

            def history(self, **_kw):
                return pd.DataFrame({"Close": [0.0]})

        monkeypatch.setattr(trading_service.yf, "Ticker", Bad)
        trading_service._price_cache.clear()
        with pytest.raises(trading_service.MarketDataUnavailable):
            trading_service.get_last_price("BAD")


def test_float32_widened_prices_are_normalised_to_their_decimal_value():
    import numpy as np
    assert trading_service._valid_price(float(np.float32(341.07))) == 341.07
    assert trading_service._valid_price(float(np.float32(65432.12))) == 65432.12
    assert trading_service._valid_price(123.456789012) == 123.456789012   # genuine float64 kept


class TestLedger:
    def test_account_is_server_side_and_fills_settle_exact_cash(self, db, user, market):
        market["AAPL"] = 123.45
        assert account(db, user)["cash"] == 100_000.0
        order(db, user, quantity=3)
        acct = account(db, user)
        assert acct["cash"] == pytest.approx(100_000 - 3 * 123.45, abs=1e-9)
        assert acct["equity"] == pytest.approx(100_000.0, abs=1e-6)
        order(db, user, side="sell", quantity=3)
        assert account(db, user)["cash"] == pytest.approx(100_000.0, abs=1e-9)

    def test_equity_includes_position_value_not_just_cash(self, db, user, market):
        market["AAPL"] = 100.0
        order(db, user, quantity=40)          # $4,000 of stock
        market["AAPL"] = 110.0
        acct = account(db, user)
        assert acct["cash"] == pytest.approx(96_000.0)
        assert acct["positions_value"] == pytest.approx(4_400.0)
        assert acct["equity"] == pytest.approx(100_400.0)
        assert acct["total_pnl"] == pytest.approx(400.0)

    def test_gross_exposure_is_zero_after_a_closed_round_trip(self, db, user, market):
        market["AAPL"] = 100.0
        order(db, user, quantity=40)
        order(db, user, side="sell", quantity=40)
        acct = account(db, user)
        assert acct["gross_exposure"] == 0.0
        assert acct["net_exposure"] == 0.0

    def test_legacy_trades_are_migrated_into_the_ledger(self, db, user, market):
        db.add(models.Trade(user_id=user.id, asset="AAPL", side="buy", quantity=10,
                            order_type="market", status="FILLED", filled_price=50))
        db.commit()
        portfolio_service.recompute_positions(db, user.id)
        market["AAPL"] = 50.0
        assert account(db, user)["cash"] == pytest.approx(99_500.0)


class TestReservations:
    def test_resting_buy_reserves_cash_and_cancel_releases_it(self, db, user, market):
        market["AAPL"] = 150.0
        placed = order(db, user, order_type="limit", limit_price=100.0, quantity=20)
        assert placed["status"] == "ACCEPTED"
        acct = account(db, user)
        assert acct["reserved_cash"] == pytest.approx(2_000.0)
        assert acct["available_cash"] == pytest.approx(98_000.0)
        main.cancel_order(placed["order_id"], user, db)
        assert account(db, user)["reserved_cash"] == 0.0

    def test_open_orders_cannot_oversubscribe_cash(self, db, user, market, uncapped):
        market["AAPL"] = 150.0
        order(db, user, order_type="limit", limit_price=100.0, quantity=400)   # $40k
        order(db, user, order_type="limit", limit_price=100.0, quantity=400)   # $80k
        with pytest.raises(HTTPException) as exc:
            order(db, user, order_type="limit", limit_price=100.0, quantity=400)
        assert exc.value.status_code == 400
        assert account(db, user)["reserved_cash"] == pytest.approx(80_000.0)

    def test_ioc_that_does_not_fill_expires_and_releases_cash(self, db, user, market):
        market["AAPL"] = 150.0
        placed = order(db, user, order_type="limit", limit_price=100.0, quantity=10, time_in_force="ioc")
        assert placed["status"] == "EXPIRED"
        assert account(db, user)["reserved_cash"] == 0.0

    def test_per_asset_concentration_counts_existing_position_and_open_orders(self, db, user, market):
        market["AAPL"] = 100.0
        order(db, user, quantity=30)                                            # $3,000 held
        order(db, user, order_type="limit", limit_price=90.0, quantity=10)      # $900 reserved
        with pytest.raises(HTTPException, match="per-asset limit"):
            order(db, user, quantity=12)                                        # +$1,200 > $5,000

    def test_rejected_order_releases_its_idempotency_key(self, db, user, market, uncapped):
        market["AAPL"] = 150.0
        order(db, user, order_type="limit", limit_price=100.0, quantity=999)   # $99,900 reserved
        with pytest.raises(HTTPException):
            order(db, user, key="retry_key_0001", order_type="limit", limit_price=100.0, quantity=10)
        assert db.query(models.IdempotencyRecord).filter_by(idempotency_key="retry_key_0001").count() == 0

    def test_reservation_is_atomic_under_concurrency(self, tmp_path):
        engine = create_engine(f"sqlite:///{tmp_path / 'ledger.db'}",
                               connect_args={"check_same_thread": False, "timeout": 30})
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        setup = Session()
        u = models.User(email="race@example.com", password_hash="x")
        setup.add(u)
        setup.commit()
        paper_broker.get_or_create_account(setup, u.id)
        user_id = u.id
        setup.close()

        wins = []
        barrier = threading.Barrier(10)

        def attempt():
            s = Session()
            try:
                barrier.wait()
                ok = paper_broker.try_reserve(s, user_id, paper_broker.to_micros(15_000))
                s.commit()
                wins.append(ok)
            finally:
                s.close()

        threads = [threading.Thread(target=attempt) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        check = Session()
        acct = check.get(models.PaperAccount, user_id)
        assert wins.count(True) == 6                       # 6 x $15k <= $100k < 7 x $15k
        assert acct.reserved_micros == paper_broker.to_micros(90_000)
        check.close()


class TestSells:
    def test_open_sell_orders_count_against_sellable_quantity(self, db, user, market):
        market["AAPL"] = 100.0
        order(db, user, quantity=10)
        order(db, user, side="sell", order_type="limit", limit_price=200.0, quantity=10)
        with pytest.raises(HTTPException, match="available paper position"):
            order(db, user, side="sell", quantity=5)

    def test_sell_fill_larger_than_holding_is_rejected_not_credited(self, db, user, market):
        market["AAPL"] = 100.0
        order(db, user, quantity=5)
        trade = models.Trade(user_id=user.id, asset="AAPL", side="sell", quantity=50,
                             order_type="limit", limit_price=100, status="ACCEPTED")
        db.add(trade)
        db.commit()
        cash_before = account(db, user)["cash"]
        outcome = paper_broker.fill_order(db, trade, 100.0)
        assert outcome.status == "REJECTED"
        assert trade.status == "REJECTED"
        assert account(db, user)["cash"] == cash_before


class TestLifecycle:
    def test_reading_orders_never_fills_them(self, db, user, market):
        market["AAPL"] = 150.0
        placed = order(db, user, order_type="limit", limit_price=100.0, quantity=10)
        market["AAPL"] = 90.0                                  # now marketable
        listed = main.list_orders(user, db)
        assert {o["order_id"]: o["status"] for o in listed}[placed["order_id"]] == "ACCEPTED"

    def test_sweeper_fills_marketable_resting_orders_with_price_improvement(self, db, user, market):
        market["AAPL"] = 150.0
        placed = order(db, user, order_type="limit", limit_price=100.0, quantity=10)
        market["AAPL"] = 90.0
        outcomes = paper_broker.process_open_orders(db)
        assert [(t.id, o.status) for t, o in outcomes] == [(placed["order_id"], "FILLED")]
        trade = db.get(models.Trade, placed["order_id"])
        assert float(trade.filled_price) == 90.0
        acct = account(db, user)
        assert acct["cash"] == pytest.approx(99_100.0)
        assert acct["reserved_cash"] == 0.0
        # A second sweep is a no-op: the fill is a compare-and-set.
        assert paper_broker.process_open_orders(db) == []

    def test_sweeper_skips_assets_without_market_data(self, db, user, market):
        market["AAPL"] = 150.0
        placed = order(db, user, order_type="limit", limit_price=100.0, quantity=10)
        del market["AAPL"]
        assert paper_broker.process_open_orders(db) == []
        assert db.get(models.Trade, placed["order_id"]).status == "ACCEPTED"

    def test_triggered_stop_limit_rests_as_a_limit_order(self, db, user, market):
        market["AAPL"] = 100.0
        placed = order(db, user, order_type="stop_limit", stop_price=105.0, limit_price=104.0, quantity=5)
        market["AAPL"] = 106.0                                 # triggers, above the limit
        assert paper_broker.process_open_orders(db) == []
        trade = db.get(models.Trade, placed["order_id"])
        assert trade.order_type == "limit"
        market["AAPL"] = 103.0                                 # below stop, marketable limit
        assert [o.status for _t, o in paper_broker.process_open_orders(db)] == ["FILLED"]

    def test_cancel_loses_cleanly_to_a_fill(self, db, user, market):
        market["AAPL"] = 150.0
        placed = order(db, user, order_type="limit", limit_price=100.0, quantity=10)
        market["AAPL"] = 90.0
        paper_broker.process_open_orders(db)
        with pytest.raises(HTTPException) as exc:
            main.cancel_order(placed["order_id"], user, db)
        assert exc.value.status_code == 400
        assert account(db, user)["reserved_cash"] == 0.0

    def test_sweep_writes_audit_entries_for_fills(self, db, user, market, monkeypatch):
        monkeypatch.setattr(main, "SessionLocal", lambda: db)
        monkeypatch.setattr(db, "close", lambda: None)
        market["AAPL"] = 150.0
        placed = order(db, user, order_type="limit", limit_price=100.0, quantity=10)
        market["AAPL"] = 95.0
        assert main.sweep_open_orders_once() == 1
        actions = [a.action for a in db.query(models.AuditLog).filter_by(resource_id=placed["order_id"])]
        assert "ORDER_FILLED" in actions
