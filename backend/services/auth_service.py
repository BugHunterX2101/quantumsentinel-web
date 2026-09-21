"""QuantumSentinel — Auth service.

Security layers
---------------
1. Argon2id password hashing (OWASP recommended parameters).
   Legacy PBKDF2-SHA256 hashes are detected and transparently upgraded to
   Argon2id on the next successful login — zero user disruption.

2. Brute-force rate limiter.
   5 failed login attempts per (email, IP) within 15 minutes triggers a
   temporary lockout (also 15 minutes). Returns HTTP 429 with Retry-After.

3. HaveIBeenPwned k-anonymity check (register only).
   Only the first 5 hex characters of SHA-1(password) are sent to the HIBP
   API — the full password never leaves the server.

4. JWT access tokens signed with RS256 (short-lived, delivered as HttpOnly cookies).

5. Refresh-token rotation with family-based reuse detection.
   Dual-write to PostgreSQL (durable) and Redis (fast lookup).

6. PQC session store with atomic nonce replay protection.

7. Full-transcript handshake signing (protocol version + all keys + nonces).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
import urllib.request
import urllib.error
from collections import defaultdict
from threading import Lock

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, VerificationError, InvalidHashError
from sqlalchemy.orm import Session

from .. import models
from ..crypto import pqc
from ..config import (JWT_SIGNING_KEY, JWT_VERIFY_KEY, JWT_ALGORITHM, JWT_EXPIRE_SECONDS,
                      REFRESH_TOKEN_SECRET, REFRESH_TOKEN_SECONDS, CSRF_SECRET)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Argon2id — OWASP recommended parameters (2024)
# ---------------------------------------------------------------------------
_PH = PasswordHasher(
    time_cost=3,
    memory_cost=65536,
    parallelism=4,
    hash_len=32,
    salt_len=16,
    encoding="utf-8",
)

_PBKDF2_ITERATIONS = 200_000


def _is_pbkdf2_hash(stored: str) -> bool:
    return not stored.startswith("$")


def hash_password(password: str) -> str:
    """Hash a password with Argon2id."""
    return _PH.hash(password)


def verify_password(password: str, stored: str) -> tuple[bool, bool]:
    """Verify password. Returns (is_valid, needs_rehash)."""
    if _is_pbkdf2_hash(stored):
        try:
            raw = base64.b64decode(stored)
            salt, digest = raw[:16], raw[16:]
            check = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PBKDF2_ITERATIONS)
            valid = hmac.compare_digest(digest, check)
            return valid, valid
        except Exception:
            return False, False
    else:
        try:
            _PH.verify(stored, password)
            return True, _PH.check_needs_rehash(stored)
        except VerifyMismatchError:
            return False, False
        except (VerificationError, InvalidHashError):
            return False, False


# ---------------------------------------------------------------------------
# Brute-force / rate limiter
# ---------------------------------------------------------------------------
_RATE_LOCK = Lock()
_FAIL_WINDOW_SECONDS = 900
_MAX_ATTEMPTS = 5
_LOCKOUT_SECONDS = 900

_FAIL_ATTEMPTS: dict[str, list[float]] = defaultdict(list)
_LOCKED_UNTIL: dict[str, float] = {}


def _rate_key(email: str, ip: str | None) -> str:
    return f"{email.lower()}:{ip or 'unknown'}"


def check_rate_limit(email: str, ip: str | None) -> tuple[bool, int]:
    """Return (is_locked, retry_after_seconds). Call before verifying password."""
    key = _rate_key(email, ip)
    now = time.time()
    with _RATE_LOCK:
        locked_until = _LOCKED_UNTIL.get(key, 0)
        if now < locked_until:
            return True, int(locked_until - now)
        _FAIL_ATTEMPTS[key] = [t for t in _FAIL_ATTEMPTS[key] if now - t < _FAIL_WINDOW_SECONDS]
    return False, 0


def record_failed_attempt(email: str, ip: str | None) -> int | None:
    """Record a failed login. Returns lockout seconds if threshold reached, else None."""
    key = _rate_key(email, ip)
    now = time.time()
    with _RATE_LOCK:
        _FAIL_ATTEMPTS[key].append(now)
        count = len(_FAIL_ATTEMPTS[key])
        if count >= _MAX_ATTEMPTS:
            _LOCKED_UNTIL[key] = now + _LOCKOUT_SECONDS
            _FAIL_ATTEMPTS[key].clear()
            log.warning("Account locked for %s after %d failed attempts", email, count)
            return _LOCKOUT_SECONDS
    return None


def clear_failed_attempts(email: str, ip: str | None) -> None:
    """Clear failure counter on successful login."""
    key = _rate_key(email, ip)
    with _RATE_LOCK:
        _FAIL_ATTEMPTS.pop(key, None)
        _LOCKED_UNTIL.pop(key, None)


# ---------------------------------------------------------------------------
# HaveIBeenPwned k-anonymity breach check
# ---------------------------------------------------------------------------
def check_hibp(password: str) -> int:
    """Return breach count from HIBP. Uses k-anonymity (5-char SHA-1 prefix only).
    Returns -1 on network error (treat as unknown, non-blocking).
    """
    sha1 = hashlib.sha1(password.encode("utf-8"), usedforsecurity=False).hexdigest().upper()
    prefix, suffix = sha1[:5], sha1[5:]
    url = f"https://api.pwnedpasswords.com/range/{prefix}"
    try:
        req = urllib.request.Request(
            url,
            headers={"Add-Padding": "true", "User-Agent": "QuantumSentinel/1.0"}
        )
        with urllib.request.urlopen(req, timeout=4) as resp:
            body = resp.read().decode("utf-8")
        for line in body.splitlines():
            parts = line.split(":")
            if len(parts) == 2 and parts[0].upper() == suffix:
                return int(parts[1])
        return 0
    except Exception as exc:
        log.debug("HIBP check failed (non-fatal): %s", exc)
        return -1


# ---------------------------------------------------------------------------
# JWT (short-lived access token)
# ---------------------------------------------------------------------------
def create_access_token(user_id: str, tier: str) -> str:
    now = int(time.time())
    payload = {"sub": user_id, "tier": tier, "iat": now, "exp": now + JWT_EXPIRE_SECONDS}
    return jwt.encode(payload, JWT_SIGNING_KEY, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, JWT_VERIFY_KEY, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        return None


# ---------------------------------------------------------------------------
# CSRF token (double-submit cookie pattern)
# ---------------------------------------------------------------------------
def generate_csrf_token(session_id: str) -> str:
    """Generate a CSRF token bound to a session identifier."""
    payload = f"{session_id}:{int(time.time())}".encode()
    sig = hmac.new(CSRF_SECRET.encode(), payload, hashlib.sha256).hexdigest()[:32]
    return base64.urlsafe_b64encode(payload + b":" + sig.encode()).decode()


def verify_csrf_token(token: str, session_id: str | None = None, max_age: int = 86400) -> bool:
    """Verify CSRF token validity and, if `session_id` is given, that the
    token was issued to that exact session. Binding the token to the caller's
    own session (from the already-verified access-token JWT) is what makes
    the double-submit pattern meaningful — without it, any validly-signed
    CSRF token from any account would pass this check."""
    try:
        decoded = base64.urlsafe_b64decode(token.encode())
        parts = decoded.rsplit(b":", 1)
        if len(parts) != 2:
            return False
        payload, sig = parts
        expected = hmac.new(CSRF_SECRET.encode(), payload, hashlib.sha256).hexdigest()[:32]
        if not hmac.compare_digest(sig.decode(), expected):
            return False
        # payload is "<session_id>:<timestamp>" — session_id itself may
        # contain ':' (it does not, UUIDs don't, but split from the right to
        # be safe), so only the trailing timestamp segment is stripped off.
        payload_str = payload.decode()
        token_session_id, _, ts_str = payload_str.rpartition(":")
        ts = int(ts_str)
        if time.time() - ts > max_age:
            return False
        if session_id is not None and not hmac.compare_digest(token_session_id, session_id):
            return False
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Refresh-token rotation (dual-write: PostgreSQL + Redis)
# ---------------------------------------------------------------------------
import datetime as dt


def create_refresh_token(db: Session, user_id: str) -> tuple[str, str]:
    """Create a new refresh token. Returns (raw_token, family_id).

    The raw token is sent as an HttpOnly cookie. Its SHA-256 hash is stored
    in PostgreSQL. Redis caches the token→user mapping for fast lookups.
    """
    raw_token = secrets.token_urlsafe(48)
    token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
    family_id = models.gen_uuid()
    expires_at = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=REFRESH_TOKEN_SECONDS)

    record = models.RefreshToken(
        user_id=user_id,
        token_hash=token_hash,
        family_id=family_id,
        expires_at=expires_at,
    )
    db.add(record)
    db.commit()
    return raw_token, family_id


def rotate_refresh_token(db: Session, raw_token: str, redis_client=None) -> tuple[str, str, str] | None:
    """Rotate a refresh token. Returns (new_raw_token, user_id, new_family_id) or None.

    Family-based reuse detection: if the old token was already used, revoke
    the entire family (all tokens in the chain) — this detects stolen tokens.
    """
    token_hash = hashlib.sha256(raw_token.encode()).hexdigest()

    # Row-locked read: without FOR UPDATE, two concurrent rotations of the
    # same not-yet-used token (e.g. a thief racing the legitimate client)
    # can both read is_used=False before either commits, both mint a valid
    # child token, and the reuse-detection family revocation below never
    # fires. with_for_update() makes the second caller block until the
    # first commits, so it then sees is_used=True and revokes the family.
    record = (
        db.query(models.RefreshToken)
        .filter_by(token_hash=token_hash)
        .with_for_update()
        .first()
    )
    if not record:
        return None

    # Check expiry
    expires_at = record.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=dt.timezone.utc)
    if expires_at < dt.datetime.now(dt.timezone.utc):
        return None

    if record.is_revoked:
        return None

    # REUSE DETECTION: if the token was already used, revoke the entire family
    if record.is_used:
        log.warning("Refresh token reuse detected for family %s, user %s — revoking family",
                     record.family_id, record.user_id)
        _revoke_family(db, record.family_id)
        return None

    # Mark old token as used
    record.is_used = True
    user_id = record.user_id

    # Issue new token in the same family
    new_raw = secrets.token_urlsafe(48)
    new_hash = hashlib.sha256(new_raw.encode()).hexdigest()
    new_expires = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=REFRESH_TOKEN_SECONDS)

    new_record = models.RefreshToken(
        user_id=user_id,
        token_hash=new_hash,
        family_id=record.family_id,
        expires_at=new_expires,
    )
    db.add(new_record)
    db.commit()

    return new_raw, user_id, record.family_id


def revoke_refresh_token(db: Session, raw_token: str) -> bool:
    """Revoke a specific refresh token."""
    token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
    record = db.query(models.RefreshToken).filter_by(token_hash=token_hash).first()
    if record:
        record.is_revoked = True
        db.commit()
        return True
    return False


def revoke_user_refresh_tokens(db: Session, user_id: str) -> int:
    """Revoke all refresh tokens for a user (logout everywhere)."""
    from sqlalchemy import update
    result = db.execute(
        update(models.RefreshToken).where(
            models.RefreshToken.user_id == user_id,
            models.RefreshToken.is_revoked.is_(False),
        ).values(is_revoked=True)
    )
    db.commit()
    return result.rowcount


def _revoke_family(db: Session, family_id: str) -> None:
    """Revoke all tokens in a family (reuse detection response)."""
    from sqlalchemy import update
    db.execute(
        update(models.RefreshToken).where(
            models.RefreshToken.family_id == family_id,
        ).values(is_revoked=True)
    )
    db.commit()


# ---------------------------------------------------------------------------
# PQC session store (in-memory, TTL-managed) — with atomic nonce replay
# ---------------------------------------------------------------------------
SESSIONS: dict[str, dict] = {}
NONCES:   dict[str, float] = {}
NONCE_TTL_SECONDS = 300
MAX_SESSIONS = 10_000
_NONCE_LOCK = Lock()
_SESSIONS_LOCK = Lock()


def _expire_sessions() -> None:
    """Expire old sessions. Nonce dedup is now handled atomically by Redis/set_nx.

    perform_handshake() is a sync route, so FastAPI runs concurrent requests
    to it on separate threadpool threads — without a lock, two threads can
    each snapshot the same "expired"/over-capacity key list and the second
    thread's `del SESSIONS[k]` then raises KeyError on a key the first
    thread already removed.
    """
    now = time.time()
    with _SESSIONS_LOCK:
        for k in [k for k, v in SESSIONS.items() if v.get("expires_at", 0) < now]:
            SESSIONS.pop(k, None)
        if len(SESSIONS) > MAX_SESSIONS:
            by_created = sorted(SESSIONS.items(), key=lambda x: x[1].get("created_at", 0))
            for k, _ in by_created[:len(SESSIONS) - MAX_SESSIONS]:
                SESSIONS.pop(k, None)


def _consume_nonce_local(nonce_bytes: bytes) -> bool:
    """In-memory atomic nonce consumption (development fallback)."""
    nonce_hash = hashlib.sha256(nonce_bytes).hexdigest()
    now = time.time()
    with _NONCE_LOCK:
        # Clean expired
        expired = [k for k, t in NONCES.items() if t < now - NONCE_TTL_SECONDS]
        for k in expired:
            del NONCES[k]
        if nonce_hash in NONCES:
            return False  # replay
        NONCES[nonce_hash] = now
        return True


def perform_handshake(db: Session, user_id: str, client_x25519_pub_b64: str,
                      client_kem_pub_b64: str | None, client_nonce_b64: str,
                      redis_client=None) -> dict:
    """Server-side hybrid X25519 + ML-KEM-768 handshake.

    Security improvements (v2):
    - Atomic nonce consumption via Redis SET NX EX 300 (Item 2)
    - Full-transcript signature binding (Item 3)
    - Server identity fingerprint (Item 4)
    """
    from ..services import security_service

    _expire_sessions()

    client_x25519_pub = pqc.unb64(client_x25519_pub_b64)
    client_nonce      = pqc.unb64(client_nonce_b64)

    # --- Item 2: Atomic nonce replay protection ---
    # Redis: SET qs:pqc:nonce:<hash> 1 NX EX 300
    # If SET NX fails → HTTP 409 HANDSHAKE_REPLAY
    #
    # Dispatched via redis_store.run_sync onto the application's own event
    # loop rather than a throwaway `asyncio.new_event_loop()` — the async
    # redis client's connections are bound to the loop that created them, so
    # a fresh loop per call either hangs or raises "attached to a different
    # loop", and if it falls back silently the replay guard is defeated.
    if redis_client:
        from .redis_store import consume_pqc_nonce, run_sync
        accepted = run_sync(consume_pqc_nonce(redis_client, client_nonce))
        if accepted is None:
            # Redis call failed/unavailable — fail safe by also requiring
            # the local in-memory check rather than silently accepting.
            accepted = _consume_nonce_local(client_nonce)
    else:
        accepted = _consume_nonce_local(client_nonce)

    if not accepted:
        from fastapi import HTTPException
        raise HTTPException(409, "HANDSHAKE_REPLAY: client nonce has already been used")

    simulated_client_kem = False
    if client_kem_pub_b64:
        client_kem_pub = pqc.unb64(client_kem_pub_b64)
    else:
        simulated_client_kem = True
        client_kem_pub, _client_kem_sk, _ = pqc.kem_keygen()

    server_x25519_pub, server_x25519_sk = pqc.x25519_keygen()
    x25519_shared = pqc.x25519_shared_secret(server_x25519_sk, client_x25519_pub)

    kem_ciphertext, kem_shared, kem_ms = pqc.kem_encapsulate(client_kem_pub)

    server_nonce = os.urandom(32)
    session_key  = pqc.derive_session_key(x25519_shared, kem_shared, client_nonce, server_nonce)

    # --- Item 3: Full-transcript signature ---
    # Sign canonical JSON of the entire handshake transcript, not just raw bytes
    from .order_security import canonical_json
    transcript = canonical_json({
        "protocol_version": pqc.HANDSHAKE_PROTOCOL_VERSION,
        "client_x25519_pub": pqc.b64(client_x25519_pub),
        "client_ml_kem_pub": pqc.b64(client_kem_pub),
        "client_nonce": pqc.b64(client_nonce),
        "server_x25519_pub": pqc.b64(server_x25519_pub),
        "server_kem_ciphertext": pqc.b64(kem_ciphertext),
        "server_nonce": pqc.b64(server_nonce),
        "kem_algorithm": "ML-KEM-768",
        "dsa_algorithm": "ML-DSA-65",
        "kex_algorithm": "X25519",
        "session_context": user_id,
    })
    transcript_hash = hashlib.sha256(transcript.encode()).hexdigest()
    # Ensures the key used for this ServerHello is resolvable in
    # server_signing_keys (see ServerIdentity.ensure_registered) — the
    # /api/security/server-keys history and future audit verification both
    # depend on the signing key having been registered before it signs.
    security_service.server_identity.ensure_registered(db)
    signature = security_service.server_identity.sign(transcript_hash.encode())

    # --- Item 4: Server identity fingerprint ---
    server_fingerprint = hashlib.sha256(security_service.server_identity.dsa_pk).hexdigest()

    session_id = pqc.b64(os.urandom(16))
    SESSIONS[session_id] = {
        "user_id":    user_id,
        "session_key": session_key,
        "created_at": time.time(),
        "expires_at": time.time() + 3600,
    }

    return {
        "session_id":               session_id,
        "server_x25519_public_key": pqc.b64(server_x25519_pub),
        "ml_kem_ciphertext":        pqc.b64(kem_ciphertext),
        "server_nonce":             pqc.b64(server_nonce),
        "ml_dsa_signature":         pqc.b64(signature),
        "server_dsa_public_key":    pqc.b64(security_service.server_identity.dsa_pk),
        "server_dsa_fingerprint":   server_fingerprint,
        "server_dsa_key_id":        getattr(security_service.server_identity, 'key_id', None),
        "transcript_hash":          transcript_hash,
        "protocol_version":         pqc.HANDSHAKE_PROTOCOL_VERSION,
        "session_token":            pqc.session_token(session_key, client_nonce, server_nonce),
        "kem_encapsulate_ms":       round(kem_ms, 3),
        "simulated_client_kem_keypair": simulated_client_kem,
        "algorithm_sizes": {
            "x25519_shared_secret_bytes": len(x25519_shared),
            "ml_kem_ciphertext_bytes":    len(kem_ciphertext),
            "ml_kem_shared_secret_bytes": len(kem_shared),
            "ml_dsa_signature_bytes":     len(signature),
            "session_key_bytes":          len(session_key),
        },
    }
