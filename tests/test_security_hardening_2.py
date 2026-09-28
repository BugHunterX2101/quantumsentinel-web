"""Security hardening: roles, JWT claims, HMAC nonce ordering, fail-closed risk
state, server identity pin, WebSocket session controls, CORS/CSP, vendored JS."""
import asyncio
import hashlib
import hmac as hmac_mod
import time
from pathlib import Path

import jwt
import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from backend import main, models, schemas
from backend.database import Base, get_db
from backend.services import auth_service, integration_service, order_security, redis_store, security_service

ORIGIN = "http://localhost:8000"


@pytest.fixture
def Session(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'sec.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    monkeypatch.setattr(main, "SessionLocal", factory)

    def override():
        db = factory()
        try:
            yield db
        finally:
            db.close()

    main.app.dependency_overrides[get_db] = override
    yield factory
    main.app.dependency_overrides.pop(get_db, None)


@pytest.fixture
def client(Session):
    return TestClient(main.app)


def make_user(Session, email="user@example.com", role="user"):
    db = Session()
    user = models.User(email=email, password_hash="x", role=role)
    db.add(user)
    db.commit()
    db.refresh(user)
    db.close()
    return user


class TestJwtClaims:
    def test_tokens_carry_and_require_iss_aud_jti(self):
        payload = auth_service.decode_access_token(auth_service.create_access_token("u1", "free"))
        assert payload["iss"] == "quantumsentinel" and payload["aud"] == "quantumsentinel-api"
        assert len(payload["jti"]) == 32

    @pytest.mark.parametrize("drop,override", [
        ("aud", {}), ("iss", {}), ("jti", {}),
        (None, {"aud": "another-service"}), (None, {"iss": "someone-else"}),
    ])
    def test_tokens_without_valid_claims_are_rejected(self, drop, override):
        now = int(time.time())
        claims = {"iss": "quantumsentinel", "aud": "quantumsentinel-api", "sub": "u1", "jti": "a" * 32,
                  "tier": "free", "iat": now, "exp": now + 60, **override}
        if drop:
            claims.pop(drop)
        from backend.config import JWT_ALGORITHM, JWT_SIGNING_KEY
        token = jwt.encode(claims, JWT_SIGNING_KEY, algorithm=JWT_ALGORITHM)
        assert auth_service.decode_access_token(token) is None


class TestApiKeys:
    def test_users_cannot_mint_admin_scoped_keys(self):
        with pytest.raises(ValueError):
            schemas.ApiKeyRequest(name="ops key", scopes=["admin"])
        assert schemas.ApiKeyRequest(name="bot key", scopes=["trade", "read"]).scopes == ["read", "trade"]

    def _signed(self, key_id, secret, nonce, path="/api/sdk/portfolio", body=b""):
        ts = str(int(time.time()))
        message = f"GET||{path}||{ts}||{nonce}||{hashlib.sha256(body).hexdigest()}"
        return {"X-QS-Key-ID": key_id, "X-QS-Timestamp": ts, "X-QS-Nonce": nonce,
                "X-QS-Signature": hmac_mod.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()}

    def test_forged_request_does_not_burn_the_nonce(self, Session, client):
        user = make_user(Session)
        db = Session()
        raw, prefix, digest, secret = integration_service.generate_api_key()
        key = models.ApiKey(user_id=user.id, name="sdk", key_prefix=prefix, key_hash=digest,
                            hmac_secret_encrypted=integration_service.encrypt_hmac_secret(secret), scopes=["read"])
        db.add(key)
        db.commit()
        key_id = key.id
        db.close()
        nonce = "nonce-" + str(time.time_ns())
        forged = self._signed(key_id, "not-the-secret", nonce)
        assert client.get("/api/sdk/portfolio", headers=forged).status_code == 403
        genuine = self._signed(key_id, secret, nonce)
        assert client.get("/api/sdk/portfolio", headers=genuine).status_code == 200
        assert client.get("/api/sdk/portfolio", headers=genuine).status_code == 409


class _BrokenRedis:
    async def exists(self, *_a):
        raise ConnectionError("redis down")

    async def scan_iter(self, *_a):
        raise ConnectionError("redis down")
        yield  # pragma: no cover


class TestFailClosed:
    def test_order_gate_rejects_when_kill_switch_state_is_unknown_in_production(self, monkeypatch):
        monkeypatch.setattr(order_security, "ENVIRONMENT", "production")
        with pytest.raises(HTTPException) as exc:
            order_security.assert_risk_gate(user_id="u1", asset="AAPL", side="buy", quantity=1, price=10,
                                            held_quantity=0, account_equity=100_000,
                                            current_gross_exposure=0, redis_client=_BrokenRedis())
        assert exc.value.status_code == 503

    def test_development_falls_back_to_process_state(self, monkeypatch):
        monkeypatch.setattr(order_security, "ENVIRONMENT", "development")
        order_security.assert_risk_gate(user_id="u1", asset="AAPL", side="buy", quantity=1, price=10,
                                        held_quantity=0, account_equity=100_000,
                                        current_gross_exposure=0, redis_client=_BrokenRedis())

    def test_kill_switch_reads_raise_in_production(self, monkeypatch):
        import backend.config as config
        monkeypatch.setattr(config, "ENVIRONMENT", "production")
        with pytest.raises(redis_store.RiskStateUnavailable):
            asyncio.run(redis_store.is_kill_switch_active(_BrokenRedis(), "global"))
        with pytest.raises(redis_store.RiskStateUnavailable):
            asyncio.run(redis_store.list_kill_switches(_BrokenRedis()))


class TestServerIdentityPin:
    def test_startup_refuses_a_key_that_does_not_match_the_pin(self, monkeypatch):
        security_service.server_identity.sign(b"ensure-key")
        monkeypatch.setattr(main, "TRUSTED_SERVER_DSA_FINGERPRINT", "0" * 64)
        with pytest.raises(RuntimeError):
            main._enforce_server_identity_pin()
        monkeypatch.setattr(main, "TRUSTED_SERVER_DSA_FINGERPRINT", security_service.server_identity.fingerprint.upper())
        main._enforce_server_identity_pin()


class TestRoles:
    def test_roles_are_provisioned_by_the_cli_only(self, Session, monkeypatch):
        from backend import manage
        monkeypatch.setattr(manage, "SessionLocal", Session)
        monkeypatch.setattr(manage, "init_db", lambda: None)
        user = make_user(Session, "op@example.com")
        assert not main._is_admin(user)
        assert manage.main(["set-role", "OP@example.com", "risk_admin"]) == 0
        db = Session()
        assert main._is_admin(db.get(models.User, user.id))
        assert db.query(models.AuditLog).filter_by(action="ROLE_CHANGED").count() == 1
        db.close()
        assert manage.main(["set-role", "nobody@example.com", "admin"]) == 1


@pytest.fixture
def quiet_signals(monkeypatch):
    monkeypatch.setattr(main.signal_engine, "get_cached_signals", lambda: {"signals": [], "generated_at": "t"})
    monkeypatch.setattr(main.signal_engine, "compute_single_asset", lambda _t: None)
    main._ws_connections.clear()


class TestWebSocket:
    def _connect(self, client, token=None, subprotocols=None):
        if token:
            client.cookies.set("qs_access", token)
        return client.websocket_connect("/api/signals/stream", headers={"origin": ORIGIN},
                                        subprotocols=subprotocols or ["qs"])

    def test_token_in_subprotocol_is_not_accepted(self, Session, client, quiet_signals):
        user = make_user(Session, "ws@example.com")
        token = auth_service.create_access_token(user.id, "free")
        with pytest.raises(WebSocketDisconnect) as exc:
            with self._connect(client, subprotocols=["qs", token]) as ws:
                ws.receive_json()
        assert exc.value.code == 4401

    def test_cookie_session_streams_and_closes_at_token_expiry(self, Session, client, quiet_signals, monkeypatch):
        user = make_user(Session, "ws@example.com")
        # JWT times are whole seconds (exp = int(now) + N), so N=1 can leave
        # well under a second of validity; 3 guarantees at least 2 s.
        monkeypatch.setattr(auth_service, "JWT_EXPIRE_SECONDS", 3)
        with self._connect(client, auth_service.create_access_token(user.id, "free")) as ws:
            assert ws.receive_json()["sequence"] == 0
            with pytest.raises(WebSocketDisconnect) as exc:
                ws.receive_json()
            assert exc.value.code == 4401

    def test_oversized_client_message_closes_the_socket(self, Session, client, quiet_signals):
        user = make_user(Session, "ws@example.com")
        with self._connect(client, auth_service.create_access_token(user.id, "free")) as ws:
            ws.receive_json()
            ws.send_text("x" * 70_000)
            with pytest.raises(WebSocketDisconnect) as exc:
                ws.receive_json()
            assert exc.value.code == 1009

    def test_per_user_connection_limit(self, Session, client, quiet_signals):
        user = make_user(Session, "ws@example.com")
        token = auth_service.create_access_token(user.id, "free")
        sockets = [self._connect(client, token) for _ in range(3)]
        opened = [s.__enter__() for s in sockets]
        for ws in opened:
            ws.receive_json()
        with pytest.raises(WebSocketDisconnect) as exc:
            with self._connect(client, token) as extra:
                extra.receive_json()
        assert exc.value.code == 4429
        for s in sockets:
            s.__exit__(None, None, None)


class TestTransportPolicy:
    def test_cors_preflight_allows_only_listed_headers(self, client):
        ok = client.options("/api/trading/orders", headers={
            "origin": ORIGIN, "access-control-request-method": "POST",
            "access-control-request-headers": "content-type,x-csrf-token,idempotency-key"})
        assert ok.status_code == 200
        bad = client.options("/api/trading/orders", headers={
            "origin": ORIGIN, "access-control-request-method": "POST",
            "access-control-request-headers": "x-anything-goes"})
        assert bad.status_code == 400

    def test_csp_allows_same_origin_scripts_only(self, client):
        csp = client.get("/health/live").headers["content-security-policy"]
        assert "script-src 'self';" in csp and "jsdelivr" not in csp

    def test_vendored_three_js_is_the_verified_release(self):
        vendor = Path(__file__).resolve().parent.parent / "frontend" / "vendor"
        digest = hashlib.sha256((vendor / "three.module.min.js").read_bytes()).hexdigest()
        # sha256 of package/build/three.module.min.js from three-0.161.0.tgz,
        # whose npm sha512 integrity was verified when it was vendored.
        assert digest == "8da856fd9ddfe38fdb286da04bc1d85f3bf108bf083e0eac71cd276ec6674030"
        assert "cdn.jsdelivr" not in (vendor.parent / "index.html").read_text(encoding="utf-8")
