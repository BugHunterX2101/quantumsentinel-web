"""Server-side paper ledger: strict pricing, atomic cash reservation, fills.

These drive the real order route (backend.main.place_order) against an
in-memory database with a controllable market price.
"""
import threading

import pytest
from fastapi import HTTPException
from sqlalchemy.orm import sessionmaker

from backend import main, models, schemas
from backend.services import order_security, paper_broker, portfolio_service, trading_service


@pytest.fixture
def db(make_engine):
    session = sessionmaker(bind=make_engine())()
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

    def test_reservation_is_atomic_under_concurrency(self, make_engine):
        Session = sessionmaker(bind=make_engine())
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

    def test_a_replayed_order_nonce_is_refused_and_nothing_else_is_reported_as_one(self, db, user, market,
                                                                                   monkeypatch):
        import time
        market["AAPL"] = 100.0
        now = int(time.time())
        envelope = {"timestamp": now, "expires_at": now + 60, "nonce": "replayed-nonce-0001"}
        order(db, user, order_id="order-0000001", **envelope)
        with pytest.raises(HTTPException) as exc:
            order(db, user, order_id="order-0000002", **envelope)
        assert exc.value.status_code == 409 and "nonce" in exc.value.detail
        # Any other failure while committing an order is not a nonce replay.
        monkeypatch.setattr(db, "commit", lambda: (_ for _ in ()).throw(RuntimeError("database down")))
        with pytest.raises(RuntimeError, match="database down"):
            order(db, user, order_id="order-0000003", nonce="fresh-nonce-000001",
                  timestamp=now, expires_at=now + 60)

    def test_concurrent_sells_of_one_holding_are_credited_once(self, make_engine, market):
        """Each concurrent sell used to check the holding against the
        positions projection, which is rebuilt only after a fill commits, so
        several sells of the same 5 shares were all filled and all credited."""
        Session = sessionmaker(bind=make_engine())
        market["AAPL"] = 100.0
        with Session() as setup:
            u = models.User(email="race-sell@example.com", password_hash="x")
            setup.add(u)
            setup.commit()
            order(setup, u, quantity=5)                                    # holds 5 AAPL
            sells = [models.Trade(user_id=u.id, asset="AAPL", side="sell", quantity=5,
                                  order_type="limit", limit_price=100, status="ACCEPTED") for _ in range(6)]
            setup.add_all(sells)
            setup.commit()
            user_id, sell_ids = u.id, [t.id for t in sells]
            cash_before = paper_broker.get_or_create_account(setup, user_id).cash_micros
        outcomes, barrier = [], threading.Barrier(len(sell_ids))

        def fill(trade_id):
            with Session() as s:
                trade = s.get(models.Trade, trade_id)
                barrier.wait()
                outcomes.append(paper_broker.fill_order(s, trade, 100.0).status)

        threads = [threading.Thread(target=fill, args=(tid,)) for tid in sell_ids]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sorted(outcomes) == ["FILLED"] + ["REJECTED"] * 5
        with Session() as check:
            cash_after = check.get(models.PaperAccount, user_id).cash_micros
            assert cash_after - cash_before == paper_broker.to_micros(500)   # one sale of 5 x $100
            assert paper_broker.held_quantity(check, user_id, "AAPL") == 0

    def test_concurrent_fills_of_one_user_leave_one_position_row_per_asset(self, make_engine):
        """Each fill used to rebuild positions after committing, unlocked: two
        overlapping rebuilds each deleted the rows they could see and inserted
        their own, leaving duplicate rows that overstated holdings and equity."""
        Session = sessionmaker(bind=make_engine())
        with Session() as setup:
            u = models.User(email="race-positions@example.com", password_hash="x")
            setup.add(u)
            setup.commit()
            user_id = u.id
            paper_broker.get_or_create_account(setup, user_id)
        for round_number in range(1, 4):
            with Session() as setup:
                buys = [models.Trade(user_id=user_id, asset=asset, side="buy", quantity=1, order_type="limit",
                                     limit_price=100, status="ACCEPTED") for asset in ["AAPL", "MSFT"] * 4]
                setup.add_all(buys)
                setup.commit()
                buy_ids = [t.id for t in buys]
            outcomes, barrier = [], threading.Barrier(len(buy_ids))

            def fill(trade_id):
                with Session() as s:
                    trade = s.get(models.Trade, trade_id)
                    barrier.wait()
                    outcomes.append(paper_broker.fill_order(s, trade, 100.0).status)

            threads = [threading.Thread(target=fill, args=(tid,)) for tid in buy_ids]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            assert outcomes == ["FILLED"] * len(buy_ids)
            with Session() as check:
                rows = check.query(models.Position).filter_by(user_id=user_id).all()
                assert sorted((p.asset, float(p.quantity)) for p in rows) == [
                    ("AAPL", 4.0 * round_number), ("MSFT", 4.0 * round_number)]

    def test_a_fill_whose_positions_rebuild_fails_is_rolled_back(self, db, user, monkeypatch):
        """The fill and its positions commit together or not at all."""
        trade = models.Trade(user_id=user.id, asset="AAPL", side="buy", quantity=1, order_type="limit",
                             limit_price=100, status="ACCEPTED")
        db.add(trade)
        db.commit()
        cash_before = paper_broker.get_or_create_account(db, user.id).cash_micros

        def broken_rebuild(_db, _user_id):
            raise RuntimeError("positions rebuild failed")

        monkeypatch.setattr(portfolio_service, "rebuild_positions", broken_rebuild)
        with pytest.raises(RuntimeError, match="positions rebuild failed"):
            paper_broker.fill_order(db, trade, 100.0)
        db.rollback()
        db.expire_all()
        assert db.get(models.Trade, trade.id).status == "ACCEPTED"
        assert db.get(models.PaperAccount, user.id).cash_micros == cash_before

    def test_repair_positions_rebuilds_only_users_with_duplicate_rows(self, db, user, capsys):
        from backend import manage

        other = models.User(email="clean-positions@example.com", password_hash="x")
        db.add(other)
        db.commit()
        for owner, quantity in ((user, 1), (user, 3), (other, 2)):
            db.add(models.Trade(user_id=owner.id, asset="AAPL", side="buy", quantity=quantity,
                                order_type="market", status="FILLED", filled_price=100))
        db.commit()
        portfolio_service.recompute_positions(db, other.id)
        clean_row_id = db.query(models.Position).filter_by(user_id=other.id).one().id
        # What the old race left behind: one row per overlapping rebuild.
        db.add_all([models.Position(user_id=user.id, asset="AAPL", quantity=q, avg_entry_price=100,
                                    realized_pnl=0) for q in (1, 4)])
        db.commit()

        assert manage._repair_positions(sessionmaker(bind=db.get_bind())) == 0
        assert "1 user(s)" in capsys.readouterr().out
        db.expire_all()
        assert [float(p.quantity) for p in db.query(models.Position).filter_by(user_id=user.id)] == [4.0]
        assert db.query(models.Position).filter_by(user_id=other.id).one().id == clean_row_id

    def test_the_fill_time_holding_agrees_with_the_positions_shown_to_the_user(self, db, user):
        """An account damaged by the old double-sell race has two filled sells
        of one 5-share buy. Positions clamp each sell at zero; the holding the
        sell check uses must count the same way, or a later legitimate sell of
        newly bought shares would be refused."""
        import datetime as dt

        start = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
        for minutes, side in enumerate(["buy", "sell", "sell", "buy"]):
            db.add(models.Trade(user_id=user.id, asset="AAPL", side=side, quantity=5, order_type="market",
                                status="FILLED", filled_price=100, filled_at=start + dt.timedelta(minutes=minutes)))
        db.commit()
        portfolio_service.recompute_positions(db, user.id)
        shown = db.query(models.Position).filter_by(user_id=user.id, asset="AAPL").one().quantity
        assert paper_broker.held_quantity(db, user.id, "AAPL") == shown == 5

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

    def test_the_sweep_loads_exactly_the_orders_the_price_can_act_on(self, db, user):
        """The SQL pre-filter must select precisely the resting orders the
        sweep loop would fill or convert: none left behind, none loaded in vain."""
        import random

        from sqlalchemy import select

        rng = random.Random(20261006)
        last = 100.0
        prices = [None, 99.99, 100.0, 100.01, 50.0, 150.0]
        for _ in range(400):
            db.add(models.Trade(user_id=user.id, asset="AAPL", status="ACCEPTED", quantity=1,
                                side=rng.choice(["buy", "sell"]),
                                order_type=rng.choice(["limit", "stop", "stop_limit", "market"]),
                                limit_price=rng.choice(prices), stop_price=rng.choice(prices)))
        db.commit()

        def acts_on(t):  # the sweep loop's own decision, order by order
            limit = float(t.limit_price) if t.limit_price is not None else None
            stop = float(t.stop_price) if t.stop_price is not None else None
            if t.order_type == "limit" and limit is not None:
                return trading_service.check_pending_limit_fill("AAPL", t.side, limit, last_price=last) is not None
            if t.order_type in ("stop", "stop_limit") and stop is not None:
                fill = trading_service.simulate_fill("AAPL", t.side, 1, t.order_type, limit, stop, last_price=last)
                return fill["status"] == "FILLED" or (t.order_type == "stop_limit"
                                                      and paper_broker.stop_triggered(t.side, stop, last))
            return True

        every = db.execute(select(models.Trade).where(models.Trade.status == "ACCEPTED")).scalars().all()
        expected = {t.id for t in every if acts_on(t)}
        selected = set(db.execute(select(models.Trade.id).where(
            models.Trade.status == "ACCEPTED", paper_broker._actionable(last))).scalars())
        assert selected == expected
        assert 0 < len(expected) < len(every)

    def test_only_one_process_sweeps_at_a_time(self, db, user, market, monkeypatch):
        from sqlalchemy import func, select

        from backend.database import engine
        monkeypatch.setattr(main, "SessionLocal", lambda: db)
        monkeypatch.setattr(db, "close", lambda: None)
        market["AAPL"] = 150.0
        order(db, user, order_type="limit", limit_price=100.0, quantity=10)
        market["AAPL"] = 95.0
        with engine.connect() as other_process:
            assert other_process.execute(select(func.pg_try_advisory_lock(main._SWEEP_LOCK_KEY))).scalar()
            assert main.sweep_open_orders_once() == 0          # another sweeper holds the tick
            other_process.execute(select(func.pg_advisory_unlock(main._SWEEP_LOCK_KEY)))
            other_process.commit()
        assert main.sweep_open_orders_once() == 1

    def test_sweep_writes_audit_entries_for_fills(self, db, user, market, monkeypatch):
        monkeypatch.setattr(main, "SessionLocal", lambda: db)
        monkeypatch.setattr(db, "close", lambda: None)
        market["AAPL"] = 150.0
        placed = order(db, user, order_type="limit", limit_price=100.0, quantity=10)
        market["AAPL"] = 95.0
        assert main.sweep_open_orders_once() == 1
        actions = [a.action for a in db.query(models.AuditLog).filter_by(resource_id=placed["order_id"])]
        assert "ORDER_FILLED" in actions
