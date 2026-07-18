"""QuantumSentinel — Security service: server PQC identity, audit logging, key rotation."""
import json
import datetime as dt
from sqlalchemy.orm import Session

from .. import models
from ..crypto import pqc

SERVER_KEY_ROTATION_DAYS = 90


class ServerIdentity:
    """The server's own long-lived ML-DSA-65 signing keypair, generated once
    at process start. Signs ServerHello handshake payloads and audit logs —
    mirrors the PQC Crypto Service role in the full architecture."""
    def __init__(self):
        pk, sk, ms = pqc.dsa_keygen()
        self.dsa_pk, self.dsa_sk = pk, sk
        self.created_at = dt.datetime.now(dt.timezone.utc)
        self.keygen_ms = ms

    def sign(self, message: bytes) -> bytes:
        sig, _ = pqc.dsa_sign(self.dsa_sk, message)
        return sig

    def rotate(self):
        pk, sk, ms = pqc.dsa_keygen()
        self.dsa_pk, self.dsa_sk = pk, sk
        self.created_at = dt.datetime.now(dt.timezone.utc)
        self.keygen_ms = ms


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
    payload = json.dumps({
        "action": entry.action, "user_id": entry.user_id,
        "resource_type": entry.resource_type, "resource_id": entry.resource_id,
        "metadata": entry.metadata_json or {},
    }, sort_keys=True).encode()
    return pqc.dsa_verify(server_identity.dsa_pk, payload, pqc.unb64(entry.pqc_signature))


def key_health(db: Session, user_id: str) -> dict:
    keys = db.query(models.KeyPair).filter(
        models.KeyPair.user_id == user_id, models.KeyPair.is_active.is_(True)
    ).all()
    now = dt.datetime.now(dt.timezone.utc)
    report = []
    threat_level = "GREEN"
    for k in keys:
        created = k.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=dt.timezone.utc)
        age_days = (now - created).days
        rotation_period = SERVER_KEY_ROTATION_DAYS if k.algorithm.startswith("ML-DSA") else 1
        due_in = rotation_period - age_days
        status = "GREEN"
        if due_in <= 0:
            status = "RED"
        elif due_in <= 7:
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
