"""QuantumSentinel — Security service: server PQC identity, audit logging, key rotation.

Item 8 enhancements:
- ServerSigningKey table tracks all historical server signing keys
- Each audit entry records signing_key_id for historical verification
- Key rotation stores the old key before generating a new one
- verify_audit_log uses the historical key, not the current server identity
- Audit chain sequence uses nextval() on PostgreSQL for atomic ordering
"""
import json
import hashlib
import datetime as dt
from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy.orm import Session
from sqlalchemy import func, select, text

from .. import models
from ..crypto import pqc
from ..config import (PRIVATE_KEY_ENCRYPTION_KEY, SERVER_DSA_PRIVATE_KEY,
                      SERVER_DSA_PUBLIC_KEY, SERVER_DSA_CREATED_AT, DATABASE_URL)

import logging as _logging

_log = _logging.getLogger(__name__)

SERVER_KEY_ROTATION_DAYS = 90
_is_postgres = DATABASE_URL.startswith(("postgresql://", "postgresql+"))

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
    mirrors the PQC Crypto Service role in the full architecture.

    Item 8: Tracks key_id and fingerprint for audit key history.
    """
    def __init__(self):
        if SERVER_DSA_PRIVATE_KEY and SERVER_DSA_PUBLIC_KEY:
            pk = pqc.unb64(SERVER_DSA_PUBLIC_KEY)
            sk = pqc.unb64(SERVER_DSA_PRIVATE_KEY)
            ms = 0.0
        else:
            pk, sk, ms = None, None, None
        self.dsa_pk, self.dsa_sk = pk, sk
        self.created_at = dt.datetime.fromisoformat(SERVER_DSA_CREATED_AT) if SERVER_DSA_CREATED_AT else dt.datetime.now(dt.timezone.utc)
        if self.created_at.tzinfo is None:
            self.created_at = self.created_at.replace(tzinfo=dt.timezone.utc)
        self.keygen_ms = ms
        # Key identification
        self.key_id: str | None = None
        self.fingerprint: str | None = None
        if pk:
            self.fingerprint = hashlib.sha256(pk).hexdigest()

    def sign(self, message: bytes) -> bytes:
        if self.dsa_sk is None:
            self._ensure_keypair()
        sig, _ = pqc.dsa_sign(self.dsa_sk, message)
        return sig

    def _ensure_keypair(self):
        pk, sk, ms = pqc.dsa_keygen()
        self.dsa_pk, self.dsa_sk, self.keygen_ms = pk, sk, ms
        self.created_at = dt.datetime.now(dt.timezone.utc)
        self.fingerprint = hashlib.sha256(pk).hexdigest()
        self.key_id = models.gen_uuid()

    def register_in_db(self, db: Session) -> str | None:
        """Register the current key in the server_signing_keys table if not already present."""
        if not self.dsa_pk:
            return None
        if not self.fingerprint:
            self.fingerprint = hashlib.sha256(self.dsa_pk).hexdigest()
        # Check if already registered
        existing = db.execute(
            select(models.ServerSigningKey).where(
                models.ServerSigningKey.fingerprint == self.fingerprint
            )
        ).scalars().first()
        if existing:
            self.key_id = existing.key_id
            return existing.key_id
        # Register new
        key_id = self.key_id or models.gen_uuid()
        self.key_id = key_id
        record = models.ServerSigningKey(
            key_id=key_id,
            algorithm="ML-DSA-65",
            public_key=pqc.b64(self.dsa_pk),
            fingerprint=self.fingerprint,
            status="active",
            activated_at=self.created_at,
        )
        db.add(record)
        db.commit()
        return key_id

    def rotate(self, db: Session | None = None):
        """Rotate the server signing key, storing the old key in history."""
        old_pk, old_fingerprint, old_key_id = self.dsa_pk, self.fingerprint, self.key_id
        old_created = self.created_at

        # Retire old key in DB
        if db and old_key_id:
            old_record = db.execute(
                select(models.ServerSigningKey).where(
                    models.ServerSigningKey.key_id == old_key_id
                )
            ).scalars().first()
            if old_record:
                old_record.status = "retired"
                old_record.retired_at = dt.datetime.now(dt.timezone.utc)
                db.commit()
        elif db and old_pk and old_fingerprint:
            # Key was never registered — register it as retired
            record = models.ServerSigningKey(
                key_id=old_key_id or models.gen_uuid(),
                algorithm="ML-DSA-65",
                public_key=pqc.b64(old_pk),
                fingerprint=old_fingerprint,
                status="retired",
                created_at=old_created,
                activated_at=old_created,
                retired_at=dt.datetime.now(dt.timezone.utc),
            )
            db.add(record)
            db.commit()

        # Generate new key
        self.dsa_pk = self.dsa_sk = None
        self._ensure_keypair()

        # Register new key
        if db:
            self.register_in_db(db)


server_identity = ServerIdentity()


def _get_audit_sequence(db: Session) -> int:
    """Get the next audit chain sequence number.

    On PostgreSQL: uses nextval() for atomic ordering (no race condition).
    On SQLite: uses MAX()+1 (safe because single-writer).
    """
    if _is_postgres:
        # Create sequence if it doesn't exist (idempotent)
        try:
            db.execute(text("CREATE SEQUENCE IF NOT EXISTS audit_chain_sequence"))
            db.commit()
        except Exception:
            db.rollback()
        result = db.execute(text("SELECT nextval('audit_chain_sequence')")).scalar()
        return int(result)
    else:
        return (db.execute(select(func.max(models.AuditChainLink.sequence))).scalar() or 0) + 1


def write_audit_log(db: Session, user_id: str | None, action: str,
                     resource_type: str | None = None, resource_id: str | None = None,
                     metadata: dict | None = None) -> models.AuditLog:
    metadata = metadata or {}
    payload = json.dumps({
        "action": action, "user_id": user_id, "resource_type": resource_type,
        "resource_id": resource_id, "metadata": metadata,
    }, sort_keys=True).encode()
    signature = server_identity.sign(payload)
    signing_key_id = server_identity.key_id
    entry = models.AuditLog(
        user_id=user_id, action=action, resource_type=resource_type,
        resource_id=resource_id, metadata_json=metadata,
        pqc_signature=pqc.b64(signature),
        signing_key_id=signing_key_id,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    # The regular ML-DSA signature protects an individual event. The hash
    # chain below additionally makes deletion, modification, and reordering
    # observable when the history is verified.
    sequence = _get_audit_sequence(db)
    previous = db.execute(
        select(models.AuditChainLink).order_by(models.AuditChainLink.sequence.desc())
    ).scalars().first()
    previous_hash = previous.entry_hash if previous else "0" * 64
    chain_payload = json.dumps({
        "audit_log_id": entry.id,
        "created_at": entry.created_at.isoformat() if entry.created_at else "",
        "payload": json.loads(payload.decode()),
        "previous_hash": previous_hash,
    }, sort_keys=True, separators=(",", ":")).encode()
    entry_hash = hashlib.sha256(chain_payload).hexdigest()
    checkpoint = pqc.b64(server_identity.sign(entry_hash.encode()))
    db.add(models.AuditChainLink(
        sequence=sequence, audit_log_id=entry.id, previous_hash=previous_hash,
        entry_hash=entry_hash, checkpoint_signature=checkpoint,
        signing_key_id=signing_key_id,
    ))
    db.commit()
    return entry


def verify_audit_log(db: Session, log_id: str) -> bool:
    """Verify an audit entry using its historical signing key."""
    entry = db.get(models.AuditLog, log_id)
    if not entry or not entry.pqc_signature:
        return False

    # Item 8: Use the historical key for verification, not the current server identity
    verification_pk = None
    if entry.signing_key_id:
        key_record = db.execute(
            select(models.ServerSigningKey).where(
                models.ServerSigningKey.key_id == entry.signing_key_id
            )
        ).scalars().first()
        if key_record:
            verification_pk = pqc.unb64(key_record.public_key)

    # Fallback to current server identity if no historical key found
    if verification_pk is None:
        if server_identity.dsa_pk is None:
            return False
        verification_pk = server_identity.dsa_pk

    payload = json.dumps({
        "action": entry.action, "user_id": entry.user_id,
        "resource_type": entry.resource_type, "resource_id": entry.resource_id,
        "metadata": entry.metadata_json or {},
    }, sort_keys=True).encode()
    return pqc.dsa_verify(verification_pk, payload, pqc.unb64(entry.pqc_signature))


def verify_audit_chain(db: Session) -> bool:
    """Verify link ordering, hashes, and ML-DSA checkpoint signatures."""
    links = db.execute(select(models.AuditChainLink).order_by(models.AuditChainLink.sequence)).scalars().all()
    previous_hash = "0" * 64
    for link in links:
        entry = db.get(models.AuditLog, link.audit_log_id)
        if not entry or link.previous_hash != previous_hash:
            return False

        # Determine verification key for this chain link
        verification_pk = None
        if link.signing_key_id:
            key_record = db.execute(
                select(models.ServerSigningKey).where(
                    models.ServerSigningKey.key_id == link.signing_key_id
                )
            ).scalars().first()
            if key_record:
                verification_pk = pqc.unb64(key_record.public_key)
        if verification_pk is None:
            if server_identity.dsa_pk is None:
                return False
            verification_pk = server_identity.dsa_pk

        event_payload = {
            "action": entry.action, "user_id": entry.user_id,
            "resource_type": entry.resource_type, "resource_id": entry.resource_id,
            "metadata": entry.metadata_json or {},
        }
        chain_payload = json.dumps({
            "audit_log_id": entry.id,
            "created_at": entry.created_at.isoformat() if entry.created_at else "",
            "payload": event_payload, "previous_hash": previous_hash,
        }, sort_keys=True, separators=(",", ":")).encode()
        calculated = hashlib.sha256(chain_payload).hexdigest()
        if calculated != link.entry_hash or not pqc.dsa_verify(
            verification_pk, link.entry_hash.encode(), pqc.unb64(link.checkpoint_signature)
        ):
            return False
        previous_hash = link.entry_hash
    return True


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
