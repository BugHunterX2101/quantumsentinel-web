"""Tests for security hardening (Items 1–8).

Covers:
- Item 1: Cookie-based auth, refresh-token rotation, CSRF
- Item 2: PQC handshake nonce replay protection
- Item 3: Full-transcript handshake signing
- Item 4: Server identity fingerprinting
- Item 5: Redis-backed kill switches
- Item 6: Audit chain with signing_key_id
- Item 7: HMAC-signed API requests
- Item 8: Server signing key history
"""
import hashlib
import hmac
import json
import os
import secrets
import time

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
os.environ.setdefault("ENVIRONMENT", "development")

from backend.services import auth_service, security_service, order_security, integration_service
from backend.crypto import pqc
from backend.database import SessionLocal, init_db, engine, Base
from backend import models


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    """Create all tables once for the module."""
    init_db()
    yield
    # Tables are left for other test modules


@pytest.fixture
def db():
    """Provide a fresh DB session per test."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def test_user(db):
    """Create a test user for auth tests."""
    user = db.query(models.User).filter_by(email="security_test@qs.dev").first()
    if not user:
        user = models.User(
            email="security_test@qs.dev",
            password_hash=auth_service.hash_password("TestPass123!"),
            tier="pro",
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    return user


# ===========================================================================
# Item 1: Cookie-based auth + refresh token rotation
# ===========================================================================

class TestRefreshTokenRotation:
    """Item 1: Refresh-token rotation with family-based reuse detection."""

    def test_create_refresh_token(self, db, test_user):
        raw, family_id = auth_service.create_refresh_token(db, test_user.id)
        assert raw and len(raw) > 32
        assert family_id

        # Verify stored in DB
        token_hash = hashlib.sha256(raw.encode()).hexdigest()
        record = db.query(models.RefreshToken).filter_by(token_hash=token_hash).first()
        assert record is not None
        assert record.user_id == test_user.id
        assert record.family_id == family_id
        assert not record.is_used
        assert not record.is_revoked

    def test_rotate_refresh_token(self, db, test_user):
        raw, family_id = auth_service.create_refresh_token(db, test_user.id)
        result = auth_service.rotate_refresh_token(db, raw)
        assert result is not None
        new_raw, user_id, new_family = result
        assert user_id == test_user.id
        assert new_family == family_id  # same family
        assert new_raw != raw

        # Old token is now marked as used
        old_hash = hashlib.sha256(raw.encode()).hexdigest()
        old_record = db.query(models.RefreshToken).filter_by(token_hash=old_hash).first()
        assert old_record.is_used

    def test_reuse_detection_revokes_family(self, db, test_user):
        raw, family_id = auth_service.create_refresh_token(db, test_user.id)

        # First rotation succeeds
        result1 = auth_service.rotate_refresh_token(db, raw)
        assert result1 is not None

        # Second rotation of the SAME token triggers reuse detection
        result2 = auth_service.rotate_refresh_token(db, raw)
        assert result2 is None  # rejected

        # All tokens in the family should be revoked
        family_tokens = db.query(models.RefreshToken).filter_by(family_id=family_id).all()
        assert all(t.is_revoked for t in family_tokens)

    def test_revoke_refresh_token(self, db, test_user):
        raw, _ = auth_service.create_refresh_token(db, test_user.id)
        assert auth_service.revoke_refresh_token(db, raw)

        # Cannot rotate a revoked token
        result = auth_service.rotate_refresh_token(db, raw)
        assert result is None

    def test_revoke_all_user_tokens(self, db, test_user):
        auth_service.create_refresh_token(db, test_user.id)
        auth_service.create_refresh_token(db, test_user.id)
        count = auth_service.revoke_user_refresh_tokens(db, test_user.id)
        assert count >= 2


class TestCSRFTokens:
    """Item 1: CSRF double-submit pattern."""

    def test_generate_and_verify_csrf(self, test_user):
        token = auth_service.generate_csrf_token(test_user.id)
        assert token
        assert auth_service.verify_csrf_token(token)

    def test_invalid_csrf_rejected(self):
        assert not auth_service.verify_csrf_token("invalid-token")
        assert not auth_service.verify_csrf_token("")

    def test_expired_csrf_rejected(self, test_user):
        token = auth_service.generate_csrf_token(test_user.id)
        # Verify with max_age=0 should fail (token is at least 0 seconds old)
        assert not auth_service.verify_csrf_token(token, max_age=0)


class TestAccessToken:
    """Item 1: JWT access token creation and decoding."""

    def test_create_and_decode(self, test_user):
        token = auth_service.create_access_token(test_user.id, test_user.tier)
        payload = auth_service.decode_access_token(token)
        assert payload is not None
        assert payload["sub"] == test_user.id
        assert payload["tier"] == test_user.tier

    def test_invalid_token_returns_none(self):
        assert auth_service.decode_access_token("garbage.token.here") is None


# ===========================================================================
# Item 2: PQC handshake nonce replay protection
# ===========================================================================

class TestNonceReplayProtection:
    """Item 2: Atomic nonce consumption (in-memory fallback)."""

    def test_nonce_consumed_once(self):
        nonce = os.urandom(32)
        assert auth_service._consume_nonce_local(nonce) is True
        assert auth_service._consume_nonce_local(nonce) is False  # replay

    def test_different_nonces_accepted(self):
        assert auth_service._consume_nonce_local(os.urandom(32)) is True
        assert auth_service._consume_nonce_local(os.urandom(32)) is True


# ===========================================================================
# Item 3 + 4: Full-transcript handshake + server identity
# ===========================================================================

class TestHandshake:
    """Items 3-4: Handshake transcript signing and server identity."""

    def test_handshake_returns_v2_fields(self, db, test_user):
        client_pub, _ = pqc.x25519_keygen()
        nonce = os.urandom(32)
        result = auth_service.perform_handshake(
            db, test_user.id,
            pqc.b64(client_pub), None, pqc.b64(nonce),
        )
        # V2 fields
        assert result["protocol_version"] == "QS-HANDSHAKE-V2"
        assert result["transcript_hash"]
        assert result["server_dsa_fingerprint"]
        assert result["server_dsa_public_key"]
        assert result["ml_dsa_signature"]

    def test_handshake_nonce_replay_rejected(self, db, test_user):
        from fastapi import HTTPException
        client_pub, _ = pqc.x25519_keygen()
        nonce = os.urandom(32)
        nonce_b64 = pqc.b64(nonce)

        # First handshake succeeds
        auth_service.perform_handshake(db, test_user.id, pqc.b64(client_pub), None, nonce_b64)

        # Second handshake with same nonce should be rejected
        with pytest.raises(HTTPException) as exc_info:
            auth_service.perform_handshake(db, test_user.id, pqc.b64(client_pub), None, nonce_b64)
        assert exc_info.value.status_code == 409
        assert "HANDSHAKE_REPLAY" in str(exc_info.value.detail)

    def test_server_fingerprint_is_sha256_of_pk(self):
        identity = security_service.server_identity
        if identity.dsa_pk:
            expected = hashlib.sha256(identity.dsa_pk).hexdigest()
            assert identity.fingerprint == expected


class TestSessionExpiryConcurrency:
    """perform_handshake() is a sync FastAPI route, so concurrent requests run
    _expire_sessions() on separate threadpool threads. Without a lock, two
    threads racing to delete the same expired/over-capacity key raise
    KeyError. Reproduce with real threads hammering the shared SESSIONS dict."""

    def test_expire_sessions_concurrent_no_keyerror(self):
        import sys
        import threading

        # Tighten the GIL switch interval so threads actually interleave
        # inside the dict iteration/deletion — at the default interval the
        # race window is narrow enough that this test can pass even against
        # the unpatched (unlocked) implementation.
        old_interval = sys.getswitchinterval()
        sys.setswitchinterval(0.00001)
        try:
            auth_service.SESSIONS.clear()
            now = time.time()
            for i in range(2000):
                auth_service.SESSIONS[f"sess-{i}"] = {
                    "created_at": now - i, "expires_at": now - 1,  # all expired
                }

            errors = []

            def worker():
                try:
                    for _ in range(10):
                        auth_service._expire_sessions()
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=worker) for _ in range(16)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert errors == [], f"_expire_sessions raced: {errors}"
            assert len(auth_service.SESSIONS) == 0
        finally:
            sys.setswitchinterval(old_interval)
            auth_service.SESSIONS.clear()


# ===========================================================================
# Item 5: Redis-backed kill switches (in-memory fallback tests)
# ===========================================================================

class TestKillSwitches:
    """Item 5: Kill switch state management."""

    def test_global_kill_switch(self):
        order_security._KILL_SWITCHES.clear()
        order_security.set_kill_switch("global", enabled=True)
        assert ("global", None) in order_security._KILL_SWITCHES

        order_security.set_kill_switch("global", enabled=False)
        assert ("global", None) not in order_security._KILL_SWITCHES

    def test_asset_kill_switch_normalizes(self):
        order_security._KILL_SWITCHES.clear()
        order_security.set_kill_switch("asset", "aapl", enabled=True)
        assert ("asset", "AAPL") in order_security._KILL_SWITCHES

    def test_risk_gate_blocks_on_kill_switch(self):
        from fastapi import HTTPException
        order_security._KILL_SWITCHES.clear()
        order_security.set_kill_switch("global", enabled=True)

        with pytest.raises(HTTPException) as exc_info:
            order_security.assert_risk_gate(
                user_id="u1", asset="AAPL", side="buy", quantity=1,
                price=100, held_quantity=0, account_equity=100000,
                current_gross_exposure=0,
            )
        assert exc_info.value.status_code == 423
        order_security._KILL_SWITCHES.clear()


# ===========================================================================
# Item 6: Audit chain with signing_key_id
# ===========================================================================

class TestAuditChain:
    """Item 6: Audit entries record signing_key_id."""

    def test_audit_log_has_signing_key_id(self, db, test_user):
        entry = security_service.write_audit_log(
            db, test_user.id, "TEST_ACTION", "test", "res-1", {"item": 6}
        )
        assert entry.signing_key_id is not None

    def test_audit_chain_integrity(self, db, test_user):
        # Write a few entries
        security_service.write_audit_log(db, test_user.id, "CHAIN_TEST_1", "test", "c1")
        security_service.write_audit_log(db, test_user.id, "CHAIN_TEST_2", "test", "c2")
        assert security_service.verify_audit_chain(db)

    def test_audit_log_verification_uses_historical_key(self, db, test_user):
        entry = security_service.write_audit_log(
            db, test_user.id, "VERIFY_TEST", "test", "v1"
        )
        assert security_service.verify_audit_log(db, entry.id)


# ===========================================================================
# Item 7: HMAC-signed API requests
# ===========================================================================

class TestHMACApiAuth:
    """Item 7: HMAC secret generation and request signing."""

    def test_generate_api_key_returns_hmac_secret(self):
        raw, prefix, digest, hmac_secret = integration_service.generate_api_key()
        assert raw.startswith("qs_")
        assert len(hmac_secret) > 16
        assert prefix == raw[:11]
        assert digest == hashlib.sha256(raw.encode()).hexdigest()

    def test_hmac_secret_encryption_roundtrip(self):
        secret = secrets.token_urlsafe(32)
        encrypted = integration_service.encrypt_hmac_secret(secret)
        decrypted = integration_service.decrypt_hmac_secret(encrypted)
        assert decrypted == secret

    def test_verify_hmac_request(self, db, test_user):
        raw, prefix, digest, hmac_secret = integration_service.generate_api_key()
        hmac_enc = integration_service.encrypt_hmac_secret(hmac_secret)
        key = models.ApiKey(
            user_id=test_user.id, name="hmac-test", key_prefix=prefix,
            key_hash=digest, hmac_secret_encrypted=hmac_enc,
            scopes=["read", "trade"],
        )
        db.add(key)
        db.commit()
        db.refresh(key)

        # Build signed request
        ts = str(int(time.time()))
        nonce = secrets.token_urlsafe(16)
        body = b'{"asset":"AAPL"}'
        body_hash = hashlib.sha256(body).hexdigest()
        message = f"POST||/api/orders||{ts}||{nonce}||{body_hash}"
        signature = hmac.new(hmac_secret.encode(), message.encode(), hashlib.sha256).hexdigest()

        result = integration_service.verify_hmac_request(
            db, key.id, ts, nonce, signature, "POST", "/api/orders", body, "trade"
        )
        assert result is not None
        assert result.id == key.id

    def test_hmac_wrong_signature_rejected(self, db, test_user):
        raw, prefix, digest, hmac_secret = integration_service.generate_api_key()
        hmac_enc = integration_service.encrypt_hmac_secret(hmac_secret)
        key = models.ApiKey(
            user_id=test_user.id, name="hmac-fail-test", key_prefix=prefix,
            key_hash=digest, hmac_secret_encrypted=hmac_enc,
            scopes=["read"],
        )
        db.add(key)
        db.commit()
        db.refresh(key)

        ts = str(int(time.time()))
        result = integration_service.verify_hmac_request(
            db, key.id, ts, "nonce1", "wrong-signature", "GET", "/api/data", b"", "read"
        )
        assert result is None


# ===========================================================================
# Item 8: Server signing key history
# ===========================================================================

class TestServerSigningKeyHistory:
    """Item 8: ServerSigningKey table and registration."""

    def test_server_identity_registered(self, db):
        key_id = security_service.server_identity.register_in_db(db)
        assert key_id is not None

        record = db.query(models.ServerSigningKey).filter_by(key_id=key_id).first()
        assert record is not None
        assert record.algorithm == "ML-DSA-65"
        assert record.status == "active"
        assert record.fingerprint == security_service.server_identity.fingerprint

    def test_idempotent_registration(self, db):
        id1 = security_service.server_identity.register_in_db(db)
        id2 = security_service.server_identity.register_in_db(db)
        assert id1 == id2


# ===========================================================================
# Password hashing
# ===========================================================================

class TestPasswordHashing:
    """Argon2id hashing and PBKDF2 transparent upgrade."""

    def test_argon2id_roundtrip(self):
        hashed = auth_service.hash_password("SecurePass123!")
        assert hashed.startswith("$argon2id$")
        valid, needs_rehash = auth_service.verify_password("SecurePass123!", hashed)
        assert valid
        assert not needs_rehash

    def test_wrong_password_rejected(self):
        hashed = auth_service.hash_password("CorrectPassword")
        valid, _ = auth_service.verify_password("WrongPassword", hashed)
        assert not valid


# ===========================================================================
# Rate limiting
# ===========================================================================

class TestRateLimiting:
    """Brute-force rate limiter."""

    def test_no_lockout_initially(self):
        locked, _ = auth_service.check_rate_limit("ratelimit@test.dev", "1.2.3.4")
        assert not locked

    def test_lockout_after_threshold(self):
        email, ip = f"lock-{secrets.token_hex(4)}@test.dev", "10.0.0.1"
        for _ in range(5):
            auth_service.record_failed_attempt(email, ip)
        locked, retry = auth_service.check_rate_limit(email, ip)
        assert locked
        assert retry > 0

    def test_clear_resets_counter(self):
        email, ip = f"clear-{secrets.token_hex(4)}@test.dev", "10.0.0.2"
        for _ in range(3):
            auth_service.record_failed_attempt(email, ip)
        auth_service.clear_failed_attempts(email, ip)
        locked, _ = auth_service.check_rate_limit(email, ip)
        assert not locked


# ===========================================================================
# Canonical JSON and order security
# ===========================================================================

class TestCanonicalJson:
    """Deterministic JSON for signing."""

    def test_canonical_json_deterministic(self):
        a = order_security.canonical_json({"b": 2, "a": 1})
        b = order_security.canonical_json({"a": 1, "b": 2})
        assert a == b
        assert a == '{"a":1,"b":2}'

    def test_canonical_order_includes_protocol(self):
        result = order_security.canonical_order(
            order_id="o1", user_id="u1", asset="aapl", side="buy",
            quantity=10.0, order_type="market", limit_price=None,
            stop_price=None, time_in_force="day", timestamp=1000,
            expires_at=1100, nonce="test-nonce-12345678",
        )
        parsed = json.loads(result)
        assert parsed["protocol"] == "QS-ORDER-V1"
        assert parsed["asset"] == "AAPL"
        assert parsed["side"] == "BUY"
