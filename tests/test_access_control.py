"""Regression tests for two access-control defects found by live exploitation.

Both were proven against a running server before being fixed here:

1. CSRF double-submit bypass (backend/main.py:get_current_user). The token was
   read as `request.headers.get("X-CSRF-Token") or request.cookies.get("qs_csrf")`.
   The cookie fallback validated the cookie against itself — and browsers attach
   cookies to cross-site requests automatically — so a forged cross-origin
   request satisfied the check with zero knowledge of the token. Verified live:
   a session carrying only cookies and NO X-CSRF-Token header successfully
   placed an order (201) and armed a global kill switch (200).

2. Missing authorization on /api/risk/kill-switch. The docstring claimed
   "Requires admin-level user" but the only dependency was get_current_user,
   and no admin concept existed anywhere in the codebase. Verified live: an
   ordinary free-tier account halted a second account's trading platform-wide
   (victim's order -> 423 "trading kill switch is active"), and the targeted
   variant blocked only the victim while the attacker kept trading (201).

Follows the established pattern in tests/test_order_security.py: call the
endpoint functions directly against an in-memory SQLite user, bypassing
Depends() injection.
"""
import asyncio

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend import models
from backend.database import Base
from backend.services import auth_service, order_security


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


@pytest.fixture(autouse=True)
def _clear_switches():
    order_security._KILL_SWITCHES.clear()
    yield
    order_security._KILL_SWITCHES.clear()


class _FakeRequest:
    """get_current_user only touches .cookies, .headers and .method."""

    def __init__(self, cookies=None, headers=None, method="POST"):
        self.cookies = cookies or {}
        self.headers = headers or {}
        self.method = method


def _make_user(db, email="trader@example.com"):
    user = models.User(email=email, password_hash="x", is_active=True)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _session_cookies(user, include_csrf_cookie=True):
    access = auth_service.create_access_token(user.id, user.tier or "free")
    csrf = auth_service.generate_csrf_token(session_id=user.id)
    cookies = {"qs_access": access}
    if include_csrf_cookie:
        cookies["qs_csrf"] = csrf
    return cookies, csrf


class TestCsrfCookieFallbackRemoved:
    def test_cookie_only_request_is_rejected(self, db):
        """The exploited path: cookies present, no X-CSRF-Token header."""
        from backend import main

        user = _make_user(db)
        cookies, _ = _session_cookies(user)

        with pytest.raises(main.HTTPException) as exc:
            main.get_current_user(
                _FakeRequest(cookies=cookies, headers={}, method="POST"),
                authorization=None,
                db=db,
            )
        assert exc.value.status_code == 403

    def test_valid_header_token_still_accepted(self, db):
        """The legitimate path must keep working."""
        from backend import main

        user = _make_user(db)
        cookies, csrf = _session_cookies(user)

        got = main.get_current_user(
            _FakeRequest(cookies=cookies, headers={"X-CSRF-Token": csrf}, method="POST"),
            authorization=None,
            db=db,
        )
        assert got.id == user.id

    def test_safe_methods_need_no_csrf(self, db):
        from backend import main

        user = _make_user(db)
        cookies, _ = _session_cookies(user)

        got = main.get_current_user(
            _FakeRequest(cookies=cookies, headers={}, method="GET"),
            authorization=None,
            db=db,
        )
        assert got.id == user.id

    def test_another_users_csrf_token_rejected(self, db):
        """Session binding must still hold (this already worked pre-fix)."""
        from backend import main

        victim = _make_user(db, "victim@example.com")
        attacker = _make_user(db, "attacker@example.com")
        victim_cookies, _ = _session_cookies(victim)
        _, attacker_csrf = _session_cookies(attacker)

        with pytest.raises(main.HTTPException) as exc:
            main.get_current_user(
                _FakeRequest(cookies=victim_cookies,
                             headers={"X-CSRF-Token": attacker_csrf}, method="POST"),
                authorization=None,
                db=db,
            )
        assert exc.value.status_code == 403


class TestKillSwitchAuthorization:
    def test_non_admin_cannot_arm_global_switch(self, db, monkeypatch):
        from backend import main

        monkeypatch.setattr(main, "ADMIN_EMAILS", set())
        user = _make_user(db)

        with pytest.raises(main.HTTPException) as exc:
            asyncio.run(main.manage_kill_switch(
                {"scope": "global", "enabled": True}, user=user, db=db))
        assert exc.value.status_code == 403
        assert ("global", None) not in order_security._KILL_SWITCHES

    def test_non_admin_cannot_target_another_user(self, db, monkeypatch):
        from backend import main

        monkeypatch.setattr(main, "ADMIN_EMAILS", set())
        attacker = _make_user(db, "attacker@example.com")
        victim = _make_user(db, "victim@example.com")

        with pytest.raises(main.HTTPException) as exc:
            asyncio.run(main.manage_kill_switch(
                {"scope": "user", "identifier": victim.id, "enabled": True},
                user=attacker, db=db))
        assert exc.value.status_code == 403
        assert ("user", victim.id) not in order_security._KILL_SWITCHES

    def test_non_admin_cannot_arm_asset_switch(self, db, monkeypatch):
        from backend import main

        monkeypatch.setattr(main, "ADMIN_EMAILS", set())
        user = _make_user(db)

        with pytest.raises(main.HTTPException) as exc:
            asyncio.run(main.manage_kill_switch(
                {"scope": "asset", "identifier": "AAPL", "enabled": True},
                user=user, db=db))
        assert exc.value.status_code == 403

    def test_user_may_halt_their_own_trading(self, db, monkeypatch):
        """Self-service risk control stays available to everyone."""
        from backend import main

        monkeypatch.setattr(main, "ADMIN_EMAILS", set())
        user = _make_user(db)

        result = asyncio.run(main.manage_kill_switch(
            {"scope": "user", "identifier": user.id, "enabled": True},
            user=user, db=db))
        assert result["enabled"] is True
        assert ("user", user.id) in order_security._KILL_SWITCHES

    def test_admin_may_arm_global_switch(self, db, monkeypatch):
        from backend import main

        admin = _make_user(db, "ops@example.com")
        monkeypatch.setattr(main, "ADMIN_EMAILS", {"ops@example.com"})

        result = asyncio.run(main.manage_kill_switch(
            {"scope": "global", "enabled": True}, user=admin, db=db))
        assert result["enabled"] is True
        assert ("global", None) in order_security._KILL_SWITCHES

    def test_listing_hides_other_users_switches(self, db, monkeypatch):
        """A user-scoped entry discloses another account's id."""
        from backend import main

        monkeypatch.setattr(main, "ADMIN_EMAILS", set())
        user = _make_user(db, "me@example.com")
        order_security._KILL_SWITCHES.add(("user", "someone-elses-id"))
        order_security._KILL_SWITCHES.add(("user", user.id))
        order_security._KILL_SWITCHES.add(("global", None))

        result = asyncio.run(main.list_kill_switches_endpoint(user=user))
        identifiers = {s["identifier"] for s in result["kill_switches"]}
        assert "someone-elses-id" not in identifiers
        assert user.id in identifiers
        assert None in identifiers  # global still visible — it affects them
