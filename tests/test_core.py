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
