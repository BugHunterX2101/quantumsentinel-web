"""Scoped SDK API keys with HMAC-signed requests and signed webhook delivery.

Item 7 enhancements:
- API keys now generate an HMAC secret for request signing
- verify_hmac_request() validates X-QS-SIGNATURE headers
- Signature = HMAC-SHA256(secret, method || path || timestamp || nonce || SHA256(body))
"""
import contextlib
import datetime as dt
import hashlib
import hmac as hmac_mod
import ipaddress
import json
import secrets
import socket
import threading
from urllib.parse import urlparse

import requests
from cryptography.fernet import Fernet
from sqlalchemy.orm import Session

from .. import models
from ..config import WEBHOOK_ENCRYPTION_KEY

_FERNET = Fernet(WEBHOOK_ENCRYPTION_KEY.encode() if WEBHOOK_ENCRYPTION_KEY else Fernet.generate_key())


def generate_api_key() -> tuple[str, str, str, str]:
    """Generate an API key with HMAC secret.
    Returns (raw_key, prefix, key_hash, hmac_secret).
    """
    raw = "qs_" + secrets.token_urlsafe(32)
    hmac_secret = secrets.token_urlsafe(32)
    return raw, raw[:11], hashlib.sha256(raw.encode()).hexdigest(), hmac_secret


def encrypt_hmac_secret(value: str) -> str:
    """Encrypt HMAC secret for storage."""
    return _FERNET.encrypt(value.encode()).decode()


def decrypt_hmac_secret(value: str) -> str:
    """Decrypt HMAC secret from storage."""
    return _FERNET.decrypt(value.encode()).decode()


def encrypt_secret(value: str) -> str:
    return _FERNET.encrypt(value.encode()).decode()


def verify_api_key(db: Session, raw_key: str, scope: str) -> models.ApiKey | None:
    key = db.query(models.ApiKey).filter(models.ApiKey.key_hash == hashlib.sha256(raw_key.encode()).hexdigest(),
                                         models.ApiKey.is_revoked.is_(False)).first()
    if not key or (scope not in (key.scopes or []) and "admin" not in (key.scopes or [])):
        return None
    if key.expires_at:
        # Make expires_at timezone-aware before comparison to avoid TypeError.
        # DB may store as naive UTC or as tz-aware; handle both cases.
        expires_aware = key.expires_at if key.expires_at.tzinfo else key.expires_at.replace(tzinfo=dt.timezone.utc)
        if expires_aware < dt.datetime.now(dt.timezone.utc):
            return None
    key.last_used_at = dt.datetime.now(dt.timezone.utc)
    db.commit()
    return key


def verify_hmac_request(db: Session, key_id: str, timestamp: str, nonce: str,
                         signature: str, method: str, path: str, body: bytes,
                         scope: str) -> models.ApiKey | None:
    """Verify an HMAC-signed API request (Item 7).

    Validates:
    - Key exists and is active
    - Scope is sufficient
    - Timestamp is within ±30 second window
    - Signature = HMAC-SHA256(secret, method || path || timestamp || nonce || SHA256(body))

    Nonce dedup is handled by the caller via Redis.
    """
    # Find key by ID (not hash — HMAC requests use key_id header)
    key = db.query(models.ApiKey).filter(
        models.ApiKey.id == key_id,
        models.ApiKey.is_revoked.is_(False),
    ).first()
    if not key or (scope not in (key.scopes or []) and "admin" not in (key.scopes or [])):
        return None

    if key.expires_at:
        expires_aware = key.expires_at if key.expires_at.tzinfo else key.expires_at.replace(tzinfo=dt.timezone.utc)
        if expires_aware < dt.datetime.now(dt.timezone.utc):
            return None

    # Check timestamp window (±30 seconds)
    try:
        ts = int(timestamp)
        now = int(dt.datetime.now(dt.timezone.utc).timestamp())
        if abs(now - ts) > 30:
            return None
    except (ValueError, TypeError):
        return None

    # Verify signature
    if not key.hmac_secret_encrypted:
        return None
    try:
        hmac_secret = decrypt_hmac_secret(key.hmac_secret_encrypted)
    except Exception:
        return None

    body_hash = hashlib.sha256(body).hexdigest()
    message = f"{method.upper()}||{path}||{timestamp}||{nonce}||{body_hash}"
    expected_sig = hmac_mod.new(hmac_secret.encode(), message.encode(), hashlib.sha256).hexdigest()

    if not hmac_mod.compare_digest(signature, expected_sig):
        return None

    key.last_used_at = dt.datetime.now(dt.timezone.utc)
    db.commit()
    return key


def _is_public_address(addr: str) -> bool:
    ip = ipaddress.ip_address(addr)
    return not (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_multicast or ip.is_reserved or ip.is_unspecified)


def _resolve_public_addresses(hostname: str, port: int) -> list[str] | None:
    """Resolve hostname:port and return its IPs iff every one is public.

    Returns None (not eligible for delivery) if resolution fails or ANY
    resolved address is private/loopback/link-local/etc — a hostname that
    round-robins between a public and an internal IP must not be trusted.
    """
    try:
        infos = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except (socket.gaierror, ValueError):
        return None
    addresses = sorted({info[4][0] for info in infos})
    if not addresses or not all(_is_public_address(a) for a in addresses):
        return None
    return addresses


def _is_public_https(url: str) -> bool:
    """Cheap eligibility check used at webhook-registration time.

    This alone is NOT sufficient to authorize the actual outbound delivery —
    see _pin_dns_to below, which re-resolves and pins the connection so the
    delivery can't be redirected by DNS changing between check and use.
    """
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        return False
    return _resolve_public_addresses(parsed.hostname, parsed.port or 443) is not None


_real_getaddrinfo = socket.getaddrinfo
_dns_pins = threading.local()


def _patched_getaddrinfo(host, port, *args, **kwargs):
    pinned = getattr(_dns_pins, "map", None)
    if pinned is not None:
        addresses = pinned.get((host, port))
        if addresses is not None:
            return [(socket.AF_INET6 if ":" in a else socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, port))
                    for a in addresses]
    return _real_getaddrinfo(host, port, *args, **kwargs)


socket.getaddrinfo = _patched_getaddrinfo


@contextlib.contextmanager
def _pin_dns_to(hostname: str, port: int, addresses: list[str]):
    """Force DNS resolution of exactly (hostname, port) on THIS thread to the
    already-validated `addresses` for the duration of the block, closing the
    TOCTOU window between _resolve_public_addresses's check and the real
    connection a moment later (an attacker controlling the webhook's DNS with
    a low TTL could otherwise return a public IP for the check and a private/
    internal IP for the real request — "DNS rebinding").

    Scoped via thread-local storage (not a process-wide lock) so it never
    affects DNS lookups for any other host, and concurrent webhook deliveries
    on other threads are unaffected. TLS SNI/certificate hostname validation
    is untouched: urllib3 still uses `hostname` (never the resolved IP) for
    the Host header and TLS handshake — only the underlying socket connect
    target changes.
    """
    pinned = getattr(_dns_pins, "map", None)
    if pinned is None:
        pinned = _dns_pins.map = {}
    pinned[(hostname, port)] = addresses
    try:
        yield
    finally:
        pinned.pop((hostname, port), None)


def emit_webhooks(db: Session, user_id: str, event_type: str, payload: dict) -> None:
    """Best-effort delivery. Failed delivery never changes an order result."""
    hooks = db.query(models.Webhook).filter(models.Webhook.user_id == user_id,
                                             models.Webhook.is_active.is_(True)).all()
    envelope = json.dumps({"event": event_type, "data": payload}, sort_keys=True, separators=(",", ":"))
    for hook in hooks:
        try:
            if event_type not in (hook.event_types or []):
                continue
            parsed = urlparse(hook.url)
            if parsed.scheme != "https" or not parsed.hostname:
                continue
            port = parsed.port or 443
            addresses = _resolve_public_addresses(parsed.hostname, port)
            if addresses is None:
                continue
            secret = _FERNET.decrypt(hook.secret_hash.encode())
            signature = hmac_mod.new(secret, envelope.encode(), hashlib.sha256).hexdigest()
            with _pin_dns_to(parsed.hostname, port, addresses):
                requests.post(hook.url, data=envelope, timeout=3, allow_redirects=False,
                              headers={"Content-Type": "application/json", "X-QS-Event": event_type,
                                       "X-QS-Signature": f"sha256={signature}"}).raise_for_status()
            hook.last_delivery_at = dt.datetime.now(dt.timezone.utc)
        except Exception:
            # Webhook delivery is explicitly best-effort and must never turn
            # a successful trade/key rotation into a 500 response.
            continue
    db.commit()
