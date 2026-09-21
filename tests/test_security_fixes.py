"""Regression tests for the six lower-severity findings from the full-platform
security audit, each verified live against the running dev server before being
fixed here:

1. No "logout everywhere" / no absolute refresh-token session lifetime —
   revoke_user_refresh_tokens() existed but was wired to nothing, and
   rotate_refresh_token() only ever enforced a SLIDING window, so a session
   kept alive purely by periodic rotation never expired.
2. Webhook SSRF DNS-rebinding TOCTOU — _is_public_https() validated the
   hostname's resolved IPs once, but emit_webhooks()'s requests.post() call
   re-resolved independently a moment later, so an attacker controlling DNS
   with a low TTL could return a public IP for the check and a private one
   for the real connection.
3. Hardcoded HMAC fallback secret + non-constant-time compare in
   experiment_registry.py — b"qs-experiment-signing-key" was a fixed literal
   readable in the (public) source, so anyone could forge a manifest
   signature whenever the ML-DSA signer wasn't used.
4. Register-endpoint account enumeration — the existing-email path returned
   almost instantly while the new-account path spent ~100ms+ on Argon2id +
   ML-KEM/ML-DSA keygen, making the two paths distinguishable by timing even
   though login already closes this gap via a constant-time dummy hash.

Follows the established pattern in tests/test_access_control.py: call
endpoint/service functions directly, bypassing FastAPI's Depends() injection.
"""
import datetime as dt
import hashlib
import hmac
import json
import socket
from urllib.parse import urlparse

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend import models, schemas
from backend.database import Base
from backend.services import auth_service, integration_service


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


def _make_user(db, email="trader@example.com"):
    user = models.User(email=email, password_hash="x", is_active=True)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


class _FakeRequest:
    def __init__(self, client_host="203.0.113.7"):
        class _Client:
            def __init__(self, host):
                self.host = host
        self.client = _Client(client_host) if client_host else None
        self.cookies = {}
        self.headers = {}
        self.method = "POST"


# ---------------------------------------------------------------------------
# 1a. Logout everywhere
# ---------------------------------------------------------------------------

class TestLogoutEverywhere:
    def test_logout_all_revokes_every_session(self, db):
        from backend import main

        user = _make_user(db)
        raw1, _ = auth_service.create_refresh_token(db, user.id)
        raw2, _ = auth_service.create_refresh_token(db, user.id)

        result = main.logout_all(_FakeRequest(), user=user, db=db)
        assert result.status_code == 200

        # Both sessions — not just the caller's own — must now be dead.
        assert auth_service.rotate_refresh_token(db, raw1) is None
        assert auth_service.rotate_refresh_token(db, raw2) is None

    def test_logout_all_does_not_touch_other_users(self, db):
        from backend import main

        victim = _make_user(db, "victim@example.com")
        other = _make_user(db, "other@example.com")
        other_raw, _ = auth_service.create_refresh_token(db, other.id)

        main.logout_all(_FakeRequest(), user=victim, db=db)

        # Another account's session must survive.
        result = auth_service.rotate_refresh_token(db, other_raw)
        assert result is not None


# ---------------------------------------------------------------------------
# 1b. Absolute refresh-token session lifetime
# ---------------------------------------------------------------------------

class TestAbsoluteSessionLifetime:
    def test_rotation_within_cap_still_succeeds(self, db, monkeypatch):
        monkeypatch.setattr(auth_service, "REFRESH_ABSOLUTE_SESSION_SECONDS", 3600)
        user = _make_user(db)
        raw, _ = auth_service.create_refresh_token(db, user.id)

        result = auth_service.rotate_refresh_token(db, raw)
        assert result is not None

    def test_rotation_past_absolute_cap_is_rejected(self, db, monkeypatch):
        """A family kept alive purely by sliding-window rotation must still
        die once it exceeds the absolute cap, forcing a real re-login."""
        monkeypatch.setattr(auth_service, "REFRESH_ABSOLUTE_SESSION_SECONDS", 60)
        user = _make_user(db)
        raw, family_id = auth_service.create_refresh_token(db, user.id)

        record = db.query(models.RefreshToken).filter_by(family_id=family_id).first()
        record.created_at = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=120)
        db.commit()

        result = auth_service.rotate_refresh_token(db, raw)
        assert result is None

        stale = db.query(models.RefreshToken).filter_by(family_id=family_id).first()
        assert stale.is_revoked is True

    def test_cap_measures_from_family_origin_not_latest_rotation(self, db, monkeypatch):
        """Each individual rotation must NOT reset the absolute clock —
        otherwise periodic refreshing defeats the cap entirely."""
        monkeypatch.setattr(auth_service, "REFRESH_ABSOLUTE_SESSION_SECONDS", 60)
        user = _make_user(db)
        raw, family_id = auth_service.create_refresh_token(db, user.id)

        record = db.query(models.RefreshToken).filter_by(family_id=family_id).first()
        record.created_at = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=30)
        db.commit()

        # First rotation succeeds (still within the 60s cap) and mints a
        # brand-new row with created_at = now — but the FAMILY is 30s old.
        result = auth_service.rotate_refresh_token(db, raw)
        assert result is not None
        new_raw, _, _ = result

        # 40 more seconds pass: the family is now 70s old (past the 60s cap)
        # even though the newest row is only 40s old.
        for row in db.query(models.RefreshToken).filter_by(family_id=family_id).all():
            row.created_at = row.created_at - dt.timedelta(seconds=40)
        db.commit()

        assert auth_service.rotate_refresh_token(db, new_raw) is None


# ---------------------------------------------------------------------------
# 2. Webhook SSRF DNS-rebinding TOCTOU
# ---------------------------------------------------------------------------

class TestWebhookDnsRebindingFix:
    def test_is_public_address_rejects_internal_ranges(self):
        assert integration_service._is_public_address("8.8.8.8") is True
        assert integration_service._is_public_address("127.0.0.1") is False
        assert integration_service._is_public_address("10.0.0.5") is False
        assert integration_service._is_public_address("169.254.1.1") is False
        assert integration_service._is_public_address("0.0.0.0") is False
        assert integration_service._is_public_address("224.0.0.1") is False

    def test_pin_dns_forces_exact_resolution_scoped_to_this_thread(self):
        with integration_service._pin_dns_to("pinned.invalid", 443, ["203.0.113.5"]):
            infos = socket.getaddrinfo("pinned.invalid", 443, type=socket.SOCK_STREAM)
            assert infos[0][4][0] == "203.0.113.5"

        # Outside the context the pin must no longer apply — this hostname
        # doesn't exist, so real resolution raises.
        with pytest.raises(socket.gaierror):
            socket.getaddrinfo("pinned.invalid", 443, type=socket.SOCK_STREAM)

    def test_emit_webhooks_immune_to_dns_rebinding(self, db, monkeypatch):
        """Simulate an attacker's low-TTL DNS: the resolver returns a PUBLIC
        IP on the first (validating) lookup and a PRIVATE IP on any
        subsequent lookup — exactly what an unpinned second `requests.post`
        resolution would hit. The real connection must still land on the
        address validated before delivery, never the rebound one."""
        call_count = {"n": 0}

        def fake_resolver(host, port, *a, **kw):
            call_count["n"] += 1
            ip = "1.1.1.1" if call_count["n"] == 1 else "10.0.0.1"
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))]

        monkeypatch.setattr(integration_service, "_real_getaddrinfo", fake_resolver)

        captured = {}

        class FakeResponse:
            def raise_for_status(self):
                pass

        def fake_post(url, **kwargs):
            # Mirrors what urllib3 does internally: resolve host:port at
            # connect time via socket.getaddrinfo.
            parsed = urlparse(url)
            infos = socket.getaddrinfo(parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM)
            captured["connect_ip"] = infos[0][4][0]
            return FakeResponse()

        monkeypatch.setattr(integration_service.requests, "post", fake_post)

        user = _make_user(db)
        hook = models.Webhook(
            user_id=user.id, url="https://rebind.example/hook",
            secret_hash=integration_service.encrypt_secret("s3cr3t-webhook-key"),
            event_types=["order.filled"], is_active=True,
        )
        db.add(hook)
        db.commit()

        integration_service.emit_webhooks(db, user.id, "order.filled", {"x": 1})

        assert call_count["n"] >= 1
        assert captured["connect_ip"] == "1.1.1.1"


# ---------------------------------------------------------------------------
# 3. experiment_registry hardcoded HMAC fallback secret
# ---------------------------------------------------------------------------

class TestExperimentManifestSigningKey:
    def test_fallback_key_derived_from_deployment_secret(self, monkeypatch):
        import backend.config as config
        from backend.services import experiment_registry as er

        monkeypatch.setattr(config, "CSRF_SECRET", "deployment-secret-a")
        key_a = er._fallback_hmac_key()
        monkeypatch.setattr(config, "CSRF_SECRET", "deployment-secret-b")
        key_b = er._fallback_hmac_key()

        assert key_a != key_b
        assert key_a == hashlib.sha256(b"qs-experiment-manifest-hmac:deployment-secret-a").digest()

    def test_signature_cannot_be_forged_with_old_hardcoded_key(self, monkeypatch):
        from backend.services import experiment_registry as er, security_service

        monkeypatch.setattr(
            security_service.server_identity, "sign",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("ML-DSA unavailable")),
        )

        manifest = {"experiment_id": "QS-FORGE-TEST", "dataset_hash": "abc123"}
        signature = er.sign_manifest(manifest)
        assert signature.startswith("hmac-sha256:")

        canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        forged = "hmac-sha256:" + hmac.new(
            b"qs-experiment-signing-key", canonical.encode(), hashlib.sha256
        ).hexdigest()

        assert forged != signature
        assert er.verify_manifest_signature(manifest, forged) is False
        # The legitimately produced signature must still verify correctly.
        assert er.verify_manifest_signature(manifest, signature) is True


# ---------------------------------------------------------------------------
# 4. Raw exception text leaking to clients
# ---------------------------------------------------------------------------

class TestNoRawExceptionLeakage:
    def test_backtest_500_hides_internal_exception_detail(self, db, monkeypatch):
        from backend import main
        from backend.services import backtest_service

        class ExplodingEngine:
            def __init__(self, config):
                pass

            def run(self):
                raise RuntimeError(
                    'psycopg2.OperationalError: password authentication failed for user "qs_prod"'
                )

        monkeypatch.setattr(backtest_service, "BacktestEngine", ExplodingEngine)
        user = _make_user(db)
        req = schemas.AdvancedBacktestRequest()

        with pytest.raises(main.HTTPException) as exc:
            main.advanced_backtest(req, user=user, db=db)

        assert exc.value.status_code == 500
        assert exc.value.detail == "Backtest failed"
        assert "psycopg2" not in str(exc.value.detail)
        assert "qs_prod" not in str(exc.value.detail)


# ---------------------------------------------------------------------------
# 5. Register-endpoint account enumeration
# ---------------------------------------------------------------------------

class TestRegisterEnumerationResistance:
    def test_existing_email_pays_same_crypto_cost_as_new_account(self, db, monkeypatch):
        """Before this fix, the existing-email path skipped Argon2id and PQC
        keygen entirely and returned almost instantly, while a genuinely new
        registration spent ~100ms+ on that work — a timing side channel that
        discloses whether an email is registered even without reading the
        response body."""
        from backend import main

        calls = {"hash": 0, "kem": 0, "dsa": 0}

        real_hash = main.auth_service.hash_password
        def spy_hash(pw):
            calls["hash"] += 1
            return real_hash(pw)
        monkeypatch.setattr(main.auth_service, "hash_password", spy_hash)

        real_kem = main.pqc.kem_keygen
        def spy_kem():
            calls["kem"] += 1
            return real_kem()
        monkeypatch.setattr(main.pqc, "kem_keygen", spy_kem)

        real_dsa = main.pqc.dsa_keygen
        def spy_dsa():
            calls["dsa"] += 1
            return real_dsa()
        monkeypatch.setattr(main.pqc, "dsa_keygen", spy_dsa)

        _make_user(db, "taken@example.com")
        req = schemas.RegisterRequest(
            email="taken@example.com",
            password="Str0ng!Passw0rd#42",
        )

        with pytest.raises(main.HTTPException) as exc:
            main.register(req, request=_FakeRequest(), db=db)

        assert exc.value.status_code == 409
        assert calls == {"hash": 1, "kem": 1, "dsa": 1}

    def test_new_account_still_registers_successfully(self, db):
        """Functional regression check: moving the crypto work earlier must
        not break successful registration."""
        from backend import main

        req = schemas.RegisterRequest(
            email="brandnew@example.com",
            password="Str0ng!Passw0rd#42",
        )
        result = main.register(req, request=_FakeRequest(), db=db)
        assert result["email"] == "brandnew@example.com"

        stored = db.query(models.User).filter_by(email="brandnew@example.com").first()
        assert stored is not None
        keys = db.query(models.KeyPair).filter_by(user_id=stored.id).all()
        assert {k.algorithm for k in keys} == {"ML-KEM-768", "ML-DSA-65"}
