"""Scoped SDK API keys and signed webhook delivery for the reference API."""
import datetime as dt
import hashlib
import hmac
import ipaddress
import json
import secrets
import socket
from urllib.parse import urlparse

import requests
from cryptography.fernet import Fernet
from sqlalchemy.orm import Session

from .. import models
from ..config import WEBHOOK_ENCRYPTION_KEY

_FERNET = Fernet(WEBHOOK_ENCRYPTION_KEY.encode() if WEBHOOK_ENCRYPTION_KEY else Fernet.generate_key())


def generate_api_key() -> tuple[str, str, str]:
    raw = "qs_" + secrets.token_urlsafe(32)
    return raw, raw[:11], hashlib.sha256(raw.encode()).hexdigest()


def encrypt_secret(value: str) -> str:
    return _FERNET.encrypt(value.encode()).decode()


def verify_api_key(db: Session, raw_key: str, scope: str) -> models.ApiKey | None:
    key = db.query(models.ApiKey).filter(models.ApiKey.key_hash == hashlib.sha256(raw_key.encode()).hexdigest(),
                                         models.ApiKey.is_revoked.is_(False)).first()
    if not key or (scope not in (key.scopes or []) and "admin" not in (key.scopes or [])):
        return None
    if key.expires_at and key.expires_at.replace(tzinfo=dt.timezone.utc) < dt.datetime.now(dt.timezone.utc):
        return None
    key.last_used_at = dt.datetime.now(dt.timezone.utc)
    db.commit()
    return key


def _is_public_https(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        return False
    try:
        addresses = socket.getaddrinfo(parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM)
        return all(not ipaddress.ip_address(item[4][0]).is_private and not ipaddress.ip_address(item[4][0]).is_loopback
                   and not ipaddress.ip_address(item[4][0]).is_link_local for item in addresses)
    except (socket.gaierror, ValueError):
        return False


def emit_webhooks(db: Session, user_id: str, event_type: str, payload: dict) -> None:
    """Best-effort delivery. Failed delivery never changes an order result."""
    hooks = db.query(models.Webhook).filter(models.Webhook.user_id == user_id,
                                             models.Webhook.is_active.is_(True)).all()
    envelope = json.dumps({"event": event_type, "data": payload}, sort_keys=True, separators=(",", ":"))
    for hook in hooks:
        try:
            if event_type not in (hook.event_types or []) or not _is_public_https(hook.url):
                continue
            secret = _FERNET.decrypt(hook.secret_hash.encode())
            signature = hmac.new(secret, envelope.encode(), hashlib.sha256).hexdigest()
            requests.post(hook.url, data=envelope, timeout=3, allow_redirects=False,
                          headers={"Content-Type": "application/json", "X-QS-Event": event_type,
                                   "X-QS-Signature": f"sha256={signature}"}).raise_for_status()
            hook.last_delivery_at = dt.datetime.now(dt.timezone.utc)
        except Exception:
            # Webhook delivery is explicitly best-effort and must never turn
            # a successful trade/key rotation into a 500 response.
            continue
    db.commit()
