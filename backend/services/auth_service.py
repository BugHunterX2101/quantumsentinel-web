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

4. JWT tokens signed with an asymmetric algorithm (from config).

5. PQC session store (in-memory, TTL-managed).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
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
from ..config import JWT_SIGNING_KEY, JWT_VERIFY_KEY, JWT_ALGORITHM, JWT_EXPIRE_SECONDS

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
# JWT
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
# PQC session store (in-memory, TTL-managed)
# ---------------------------------------------------------------------------
SESSIONS: dict[str, dict] = {}
NONCES:   dict[str, float] = {}
NONCE_TTL_SECONDS = 300
MAX_SESSIONS = 10_000


def _expire_nonces() -> None:
    cutoff = time.time() - NONCE_TTL_SECONDS
    for k in [k for k, v in NONCES.items() if v < cutoff]:
        del NONCES[k]
    now = time.time()
    for k in [k for k, v in SESSIONS.items() if v.get("expires_at", 0) < now]:
        del SESSIONS[k]
    if len(SESSIONS) > MAX_SESSIONS:
        by_created = sorted(SESSIONS.items(), key=lambda x: x[1].get("created_at", 0))
        for k, _ in by_created[:len(SESSIONS) - MAX_SESSIONS]:
            del SESSIONS[k]


def perform_handshake(db: Session, user_id: str, client_x25519_pub_b64: str,
                      client_kem_pub_b64: str | None, client_nonce_b64: str) -> dict:
    """Server-side hybrid X25519 + ML-KEM-768 handshake."""
    from ..services import security_service

    _expire_nonces()

    client_x25519_pub = pqc.unb64(client_x25519_pub_b64)
    client_nonce      = pqc.unb64(client_nonce_b64)

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

    server_hello_payload = server_x25519_pub + kem_ciphertext + client_nonce + server_nonce
    signature = security_service.server_identity.sign(server_hello_payload)

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
