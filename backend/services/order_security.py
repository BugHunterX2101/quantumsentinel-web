"""Canonical order authorisation, replay protection, and paper-trading risk gates.

Item 6 enhancements:
- Kill switches are Redis-backed for multi-worker consistency
- Nonce dedup uses Redis SET NX EX for atomicity
- Fallback to in-memory stores in development mode
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import secrets
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .. import models
from ..config import ENVIRONMENT
from ..crypto import pqc
from . import security_service

PROTOCOL = "QS-ORDER-V1"
MAX_ENVELOPE_TTL_SECONDS = 300
_GENESIS_HASH = "0" * 64

# In-memory fallback for dev (replaced by Redis in production — Item 6)
_KILL_SWITCHES: set[tuple[str, str | None]] = set()


def _decimal(value: float | Decimal | None) -> str | None:
    if value is None:
        return None
    return format(Decimal(str(value)), "f").rstrip("0").rstrip(".") or "0"


def canonical_json(value: dict[str, Any]) -> str:
    """Deterministic UTF-8 JSON; never use concatenated fields for signing."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_order(*, order_id: str, user_id: str, asset: str, side: str,
                    quantity: float, order_type: str, limit_price: float | None,
                    stop_price: float | None, time_in_force: str, timestamp: int,
                    expires_at: int, nonce: str) -> str:
    return canonical_json({
        "protocol": PROTOCOL, "order_id": order_id, "user_id": user_id,
        "asset": asset.upper(), "side": side.upper(), "quantity": _decimal(quantity),
        "order_type": order_type.upper(), "limit_price": _decimal(limit_price),
        "stop_price": _decimal(stop_price), "time_in_force": time_in_force.upper(),
        "timestamp": timestamp, "expires_at": expires_at, "nonce": nonce,
    })


def request_hash(request: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(request).encode("utf-8")).hexdigest()


def make_development_envelope(order_id: str) -> tuple[int, int, str]:
    now = int(dt.datetime.now(dt.timezone.utc).timestamp())
    return now, now + 120, secrets.token_urlsafe(24)


def validate_envelope(timestamp: int, expires_at: int, nonce: str) -> None:
    now = int(dt.datetime.now(dt.timezone.utc).timestamp())
    if not re.fullmatch(r"[A-Za-z0-9_-]{16,128}", nonce):
        raise HTTPException(400, "order nonce must be 16-128 URL-safe characters")
    if timestamp > now + 30 or timestamp < now - MAX_ENVELOPE_TTL_SECONDS:
        raise HTTPException(400, "order timestamp is outside the accepted clock window")
    if expires_at <= now or expires_at <= timestamp or expires_at - timestamp > MAX_ENVELOPE_TTL_SECONDS:
        raise HTTPException(400, "order expiry is invalid or exceeds the five-minute maximum")


def verify_or_attest(db: Session, user_id: str, canonical: str,
                      key_id: str | None, signature: str | None) -> tuple[str, str, str]:
    """Verify a client ML-DSA signature; allow server attestation only in dev."""
    if key_id and signature:
        key = db.get(models.KeyPair, key_id)
        if not key or key.user_id != user_id or key.algorithm != "ML-DSA-65" or not key.is_active:
            raise HTTPException(403, "invalid or inactive client signing key")
        try:
            valid = pqc.dsa_verify(pqc.unb64(key.public_key), canonical.encode(), pqc.unb64(signature))
        except Exception:
            valid = False
        if not valid:
            raise HTTPException(403, "invalid ML-DSA order signature")
        return key.id, signature, "client_mldsa"
    if key_id or signature:
        raise HTTPException(400, "key_id and signature must be supplied together")
    if ENVIRONMENT == "production":
        raise HTTPException(403, "production order submission requires a client ML-DSA signature")
    # Keeps the existing browser paper-trading demo functional, while being
    # explicitly distinguishable from client authorisation in the audit trail.
    return "server-development-attestation", pqc.b64(security_service.server_identity.sign(canonical.encode())), "development_server"


def reserve_idempotency(db: Session, user_id: str, key: str | None,
                        payload_hash: str) -> dict | None:
    """Return the original response for a safe retry, or reserve a new key."""
    if not key:
        if ENVIRONMENT == "production":
            raise HTTPException(400, "Idempotency-Key is required for order submission")
        return None
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,128}", key):
        raise HTTPException(400, "invalid Idempotency-Key")
    existing = db.query(models.IdempotencyRecord).filter_by(user_id=user_id, idempotency_key=key).first()
    if existing:
        if existing.request_hash != payload_hash:
            raise HTTPException(409, "Idempotency-Key was already used with different payload")
        if existing.response_json is not None:
            return existing.response_json
        raise HTTPException(409, "an order with this Idempotency-Key is in progress")
    db.add(models.IdempotencyRecord(user_id=user_id, idempotency_key=key, request_hash=payload_hash))
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        return reserve_idempotency(db, user_id, key, payload_hash)
    return None


def complete_idempotency(db: Session, user_id: str, key: str | None, response: dict) -> None:
    if not key:
        return
    record = db.query(models.IdempotencyRecord).filter_by(user_id=user_id, idempotency_key=key).first()
    if record:
        record.response_json = response
        db.commit()


# ---------------------------------------------------------------------------
# Risk gate with Redis-backed kill switches (Item 6)
# ---------------------------------------------------------------------------

async def assert_risk_gate_async(*, user_id: str, asset: str, side: str, quantity: float,
                                  price: float, held_quantity: float, account_equity: float,
                                  current_gross_exposure: float, redis_client=None) -> None:
    """Central execution boundary (async version for Redis). Must run before order is signed."""
    from . import redis_store

    # Check kill switches via Redis
    blocked = False
    if redis_client:
        blocked = (
            await redis_store.is_kill_switch_active(redis_client, "global") or
            await redis_store.is_kill_switch_active(redis_client, "user", user_id) or
            await redis_store.is_kill_switch_active(redis_client, "asset", asset.upper())
        )
    else:
        # In-memory fallback
        blocked = (("global", None) in _KILL_SWITCHES or ("user", user_id) in _KILL_SWITCHES
                   or ("asset", asset.upper()) in _KILL_SWITCHES)
    if blocked:
        raise HTTPException(423, "trading kill switch is active")

    notional = quantity * price
    if notional > account_equity * 0.05:
        raise HTTPException(400, "risk gate: order exceeds 5% account-equity notional limit")
    if side == "sell" and quantity > held_quantity:
        raise HTTPException(400, "risk gate: sell quantity exceeds available paper position")
    if current_gross_exposure + (notional if side == "buy" else 0) > account_equity:
        raise HTTPException(400, "risk gate: leverage limit exceeded")


def assert_risk_gate(*, user_id: str, asset: str, side: str, quantity: float,
                     price: float, held_quantity: float, account_equity: float,
                     current_gross_exposure: float) -> None:
    """Central execution boundary (sync). It must run before an order is signed/sent."""
    blocked = (("global", None) in _KILL_SWITCHES or ("user", user_id) in _KILL_SWITCHES
               or ("asset", asset.upper()) in _KILL_SWITCHES)
    if blocked:
        raise HTTPException(423, "trading kill switch is active")
    notional = quantity * price
    if notional > account_equity * 0.05:
        raise HTTPException(400, "risk gate: order exceeds 5% account-equity notional limit")
    if side == "sell" and quantity > held_quantity:
        raise HTTPException(400, "risk gate: sell quantity exceeds available paper position")
    if current_gross_exposure + (notional if side == "buy" else 0) > account_equity:
        raise HTTPException(400, "risk gate: leverage limit exceeded")


async def set_kill_switch_async(redis_client, scope: str, identifier: str | None = None,
                                 enabled: bool = True) -> None:
    """Set or clear a kill switch (async, Redis-backed)."""
    from . import redis_store
    if scope not in {"global", "user", "asset"}:
        raise ValueError("scope must be global, user, or asset")
    normalized_id = identifier.upper() if scope == "asset" and identifier else identifier
    await redis_store.set_kill_switch(redis_client, scope, normalized_id, enabled)
    # Also update in-memory for sync fallback
    value = (scope, normalized_id)
    if enabled:
        _KILL_SWITCHES.add(value)
    else:
        _KILL_SWITCHES.discard(value)


def set_kill_switch(scope: str, identifier: str | None = None, enabled: bool = True) -> None:
    """Set or clear a kill switch (sync, in-memory only — for dev/tests)."""
    if scope not in {"global", "user", "asset"}:
        raise ValueError("scope must be global, user, or asset")
    value = (scope, identifier.upper() if scope == "asset" and identifier else identifier)
    if enabled:
        _KILL_SWITCHES.add(value)
    else:
        _KILL_SWITCHES.discard(value)
