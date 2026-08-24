"""QuantumSentinel — Security service: server PQC identity, audit logging, key rotation."""
import json
import datetime as dt
from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy.orm import Session
from sqlalchemy import select

from .. import models
from ..crypto import pqc
from ..config import (PRIVATE_KEY_ENCRYPTION_KEY, SERVER_DSA_PRIVATE_KEY,
                      SERVER_DSA_PUBLIC_KEY, SERVER_DSA_CREATED_AT)

import logging as _logging

_log = _logging.getLogger(__name__)

SERVER_KEY_ROTATION_DAYS = 90
if PRIVATE_KEY_ENCRYPTION_KEY:
    _PRIVATE_KEY_FERNET = Fernet(PRIVATE_KEY_ENCRYPTION_KEY.encode())
else:
    # FIX S2: generate an ephemeral key ONLY in development. Every process restart
    # will generate a new key, making previously encrypted private keys permanently
    # unreadable. Log CRITICAL so this is never silently ignored in production.
    _ephemeral_key = Fernet.generate_key()
    _PRIVATE_KEY_FERNET = Fernet(_ephemeral_key)
    _log.critical(
        "PRIVATE_KEY_ENCRYPTION_KEY is not set. An ephemeral Fernet key was "
        "generated for this process. All user DSA private keys encrypted in "
        "previous sessions are now UNREADABLE. Set PRIVATE_KEY_ENCRYPTION_KEY "
        "in your environment (.env or secret manager) and restart."
    )



def protect_private_key(value: str) -> str:
    return "enc:" + _PRIVATE_KEY_FERNET.encrypt(value.encode()).decode()


def unprotect_private_key(value: str) -> str:
    if not value.startswith("enc:"):
        # Legacy demo rows were base64 only; keep reads compatible so a
        # deployment can rotate them without losing access to old orders.
        return value
    try:
        return _PRIVATE_KEY_FERNET.decrypt(value[4:].encode()).decode()
    except InvalidToken as exc:
        raise ValueError("private key cannot be decrypted with the configured key") from exc


class ServerIdentity:
    """The server's own long-lived ML-DSA-65 signing keypair, generated once
    at process start. Signs ServerHello handshake payloads and audit logs —
    mirrors the PQC Crypto Service role in the full architecture."""
    def __init__(self):
        if SERVER_DSA_PRIVATE_KEY and SERVER_DSA_PUBLIC_KEY:
            pk = pqc.unb64(SERVER_DSA_PUBLIC_KEY)
            sk = pqc.unb64(SERVER_DSA_PRIVATE_KEY)
            ms = 0.0
        else:
            # ML-DSA reference key generation is intentionally expensive. Do
            # not block process import/startup; generate on first signature in
            # development, while production requires a persisted identity.
            pk, sk, ms = None, None, None
        self.dsa_pk, self.dsa_sk = pk, sk
        self.created_at = dt.datetime.fromisoformat(SERVER_DSA_CREATED_AT) if SERVER_DSA_CREATED_AT else dt.datetime.now(dt.timezone.utc)
        if self.created_at.tzinfo is None:
            self.created_at = self.created_at.replace(tzinfo=dt.timezone.utc)
        self.keygen_ms = ms

    def sign(self, message: bytes) -> bytes:
        if self.dsa_sk is None:
            self._ensure_keypair()
        sig, _ = pqc.dsa_sign(self.dsa_sk, message)
        return sig

    def _ensure_keypair(self):
        pk, sk, ms = pqc.dsa_keygen()
        self.dsa_pk, self.dsa_sk, self.keygen_ms = pk, sk, ms
        self.created_at = dt.datetime.now(dt.timezone.utc)

    def rotate(self):
        self.dsa_pk = self.dsa_sk = None
        self._ensure_keypair()


server_identity = ServerIdentity()


def write_audit_log(db: Session, user_id: str | None, action: str,
                     resource_type: str | None = None, resource_id: str | None = None,
                     metadata: dict | None = None) -> models.AuditLog:
    metadata = metadata or {}
    payload = json.dumps({
        "action": action, "user_id": user_id, "resource_type": resource_type,
        "resource_id": resource_id, "metadata": metadata,
    }, sort_keys=True).encode()
    signature = server_identity.sign(payload)
    entry = models.AuditLog(
        user_id=user_id, action=action, resource_type=resource_type,
        resource_id=resource_id, metadata_json=metadata,
        pqc_signature=pqc.b64(signature),
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


def verify_audit_log(db: Session, log_id: str) -> bool:
    entry = db.get(models.AuditLog, log_id)
    if not entry or not entry.pqc_signature:
        return False
    if server_identity.dsa_pk is None:
        # Server keypair was ephemeral (no SERVER_DSA keys configured);
        # signatures from a prior process cannot be verified — return None-like False
        # but do not crash.
        return False
    payload = json.dumps({
        "action": entry.action, "user_id": entry.user_id,
        "resource_type": entry.resource_type, "resource_id": entry.resource_id,
        "metadata": entry.metadata_json or {},
    }, sort_keys=True).encode()
    return pqc.dsa_verify(server_identity.dsa_pk, payload, pqc.unb64(entry.pqc_signature))


def key_health(db: Session, user_id: str) -> dict:
    # Use SQLAlchemy 2.0-style select() for consistency with the rest of the codebase
    keys = db.execute(
        select(models.KeyPair).where(
            models.KeyPair.user_id == user_id, models.KeyPair.is_active.is_(True)
        )
    ).scalars().all()
    now = dt.datetime.now(dt.timezone.utc)
    report = []
    threat_level = "GREEN"
    for k in keys:
        created = k.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=dt.timezone.utc)
        age_days = (now - created).days
        # Both KEM and DSA keys use the same 90-day rotation policy.
        # (The previous code used 1 day for ML-KEM-768, which incorrectly
        # flagged all KEM keys RED after 24 hours.)
        rotation_period = SERVER_KEY_ROTATION_DAYS
        due_in = rotation_period - age_days
        status = "GREEN"
        if due_in <= 0:
            status = "RED"
        elif due_in <= 14:  # aligned with frontend badge threshold (< 15 days)
            status = "YELLOW"
        if status == "RED":
            threat_level = "RED"
        elif status == "YELLOW" and threat_level != "RED":
            threat_level = "YELLOW"
        report.append({
            "algorithm": k.algorithm, "key_id": k.id, "age_days": age_days,
            "rotation_due_in_days": due_in, "status": status,
            "rotation_count": k.rotation_count,
        })
    return {"keys": report, "quantum_threat_level": threat_level}
