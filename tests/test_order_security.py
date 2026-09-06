import datetime as dt

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend import models
from backend.database import Base
from backend.services import order_security, security_service


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


def test_canonical_order_is_deterministic_and_decimal_safe():
    common = dict(order_id="o-12345678", user_id="u-12345678", asset="aapl", side="buy",
                  quantity=100.0, order_type="limit", limit_price=225.50,
                  stop_price=None, time_in_force="day", timestamp=1_700_000_000,
                  expires_at=1_700_000_100, nonce="a" * 24)
    first = order_security.canonical_order(**common)
    second = order_security.canonical_order(**common)
    assert first == second
    assert '"quantity":"100"' in first
    assert '"limit_price":"225.5"' in first
    assert '"protocol":"QS-ORDER-V1"' in first


def test_idempotency_returns_original_response_and_rejects_payload_change(db):
    assert order_security.reserve_idempotency(db, "u1", "key_12345678", "a" * 64) is None
    order_security.complete_idempotency(db, "u1", "key_12345678", {"order_id": "one"})
    assert order_security.reserve_idempotency(db, "u1", "key_12345678", "a" * 64) == {"order_id": "one"}
    with pytest.raises(HTTPException, match="different payload"):
        order_security.reserve_idempotency(db, "u1", "key_12345678", "b" * 64)


def test_risk_gate_honours_asset_kill_switch():
    order_security.set_kill_switch("asset", "AAPL", True)
    try:
        with pytest.raises(HTTPException) as exc:
            order_security.assert_risk_gate(user_id="u1", asset="AAPL", side="buy", quantity=1,
                                            price=100, held_quantity=0, account_equity=10_000,
                                            current_gross_exposure=0)
        assert exc.value.status_code == 423
    finally:
        order_security.set_kill_switch("asset", "AAPL", False)


def test_audit_history_is_hash_chained_and_tampering_is_detected(db):
    security_service.write_audit_log(db, "u1", "ORDER_CREATED", "trade", "t1", {"asset": "AAPL"})
    security_service.write_audit_log(db, "u1", "ORDER_FILLED", "trade", "t1", {"price": 100})
    assert security_service.verify_audit_chain(db)
    link = db.query(models.AuditChainLink).filter_by(sequence=2).one()
    link.entry_hash = "f" * 64
    db.commit()
    assert not security_service.verify_audit_chain(db)


def test_order_endpoint_returns_original_response_for_idempotent_retry(db, monkeypatch):
    """Exercise risk → canonical envelope → paper execution → audit end to end."""
    from backend import main
    from backend import schemas
    user = models.User(email="orders@example.com", password_hash="x")
    db.add(user)
    db.commit()

    monkeypatch.setattr(main.trading_service, "get_last_price", lambda _: 100.0)
    monkeypatch.setattr(main.trading_service, "alpaca_enabled", lambda: False)
    request = schemas.OrderRequest(asset="AAPL", side="buy", quantity=10, order_type="market")
    first = main.place_order(request, user, db, "retry_key_12345")
    second = main.place_order(request, user, db, "retry_key_12345")
    assert first["order_id"] == second["order_id"]
    assert db.query(models.Trade).count() == 1
    assert db.query(models.OrderSecurityRecord).count() == 1


def test_sdk_order_forwards_idempotency_key(db, monkeypatch):
    from backend import main, schemas
    user = models.User(email="sdk-orders@example.com", password_hash="x")
    db.add(user)
    db.commit()
    key = models.ApiKey(user_id=user.id, name="trade", key_prefix="qs_test",
                        key_hash="hash", scopes=["trade"])
    db.add(key)
    db.commit()
    monkeypatch.setattr(main.trading_service, "get_last_price", lambda _: 100.0)
    monkeypatch.setattr(main.trading_service, "alpaca_enabled", lambda: False)
    request = schemas.OrderRequest(asset="MSFT", side="buy", quantity=10, order_type="market")
    first = main.sdk_order(request, key, db, "sdk_retry_12345")
    second = main.sdk_order(request, key, db, "sdk_retry_12345")
    assert first["order_id"] == second["order_id"]
