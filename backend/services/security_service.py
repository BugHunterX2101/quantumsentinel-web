"""QuantumSentinel — Security service: server PQC identity, audit logging, key rotation.

Item 8 enhancements:
- ServerSigningKey table tracks all historical server signing keys
- Each audit entry records signing_key_id for historical verification
- Key rotation stores the old key before generating a new one
- verify_audit_log uses the historical key, not the current server identity
- Audit chain appends are serialised by a PostgreSQL advisory lock
"""
import json
import hashlib
import hmac
import threading
import datetime as dt
from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy.orm import Session
from sqlalchemy import func, insert, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

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
        # True once THIS process has confirmed the current key is persisted
        # in server_signing_keys. See ensure_registered() for why this matters.
        self._registered = False
        # Concurrent first requests must not each generate (and register)
        # a key of their own: one generates, the rest adopt it.
        self._key_lock = threading.Lock()
        if pk:
            self.fingerprint = hashlib.sha256(pk).hexdigest()

    def sign(self, message: bytes) -> bytes:
        if self.dsa_sk is None:
            with self._key_lock:
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
        self._registered = False

    def ensure_registered(self, db: Session) -> None:
        """Guarantee the active signing key is resolvable in server_signing_keys
        *before* it signs anything that will later be verified by key_id
        (audit-chain checkpoints, audit-log entries).

        Without this, a key generated lazily on first use (`sign()` calling
        `_ensure_keypair()`) can sign several entries with `signing_key_id`
        set to an id that isn't in the DB yet. `verify_audit_log` /
        `verify_audit_chain` then fall back to whatever key is *currently*
        active — which is correct only until the process restarts or the
        key rotates, at which point those earlier entries become permanently
        unverifiable even though nothing about them was tampered with. In
        production this gap is already closed by the FastAPI lifespan
        calling `register_in_db()` before the app accepts traffic; this
        method closes it for every other caller (tests, scripts, request
        paths that sign before that hook has run) with the same idempotent
        registration, at the cost of one no-op check per process instead of
        a bug in the audit trail's core accountability guarantee.
        """
        if self._registered:
            return
        with self._key_lock:
            if self._registered:
                return
            if self.dsa_sk is None:
                self._ensure_keypair()
            self.register_in_db(db)
            self._registered = True

    def register_in_db(self, db: Session) -> str | None:
        """Register the current key in the server_signing_keys table if not already present."""
        if not self.dsa_pk:
            return None
        if not self.fingerprint:
            self.fingerprint = hashlib.sha256(self.dsa_pk).hexdigest()
        # Every process sharing this key (each API and research worker)
        # registers it at startup, often at the same instant: insert-if-absent
        # is one atomic statement, and all of them then adopt whichever row won.
        db.execute(
            pg_insert(models.ServerSigningKey).values(
                key_id=self.key_id or models.gen_uuid(),
                algorithm="ML-DSA-65",
                public_key=pqc.b64(self.dsa_pk),
                fingerprint=self.fingerprint,
                status="active",
                created_at=dt.datetime.now(dt.timezone.utc),
                activated_at=self.created_at,
            ).on_conflict_do_nothing(index_elements=["fingerprint"])
        )
        key_id = db.execute(
            select(models.ServerSigningKey.key_id).where(
                models.ServerSigningKey.fingerprint == self.fingerprint)
        ).scalar_one()
        db.commit()
        self.key_id = key_id
        self._registered = True
        return key_id

    def rotate(self, db: Session | None = None):
        """Rotate the server signing key, storing the old key in history."""
        old_pk, old_fingerprint, old_key_id = self.dsa_pk, self.fingerprint, self.key_id
        old_created = self.created_at

        # Retire old key in DB
        old_record = None
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
        if db and old_record is None and old_pk and old_fingerprint:
            # Key was never registered here — record it as retired, or
            # everything it signed becomes unverifiable after rotation.
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


def enforce_identity_pin(pinned: str | None) -> None:
    """Refuse to run if the signing key is not the pinned one.

    TRUSTED_SERVER_DSA_FINGERPRINT pins the deployment's ML-DSA identity:
    a mismatch means the configured key material is not the key operators
    registered, so the process must not sign audit logs or handshakes.
    """
    pinned = (pinned or "").strip().lower()
    if not pinned:
        return
    actual = (server_identity.fingerprint or "").lower()
    if not hmac.compare_digest(pinned, actual):
        raise RuntimeError("server ML-DSA key does not match TRUSTED_SERVER_DSA_FINGERPRINT")


# Orders chain appends between threads of this process before they contend
# for the database-wide advisory lock below (which orders them across
# processes), so at most one connection per process waits on that lock.
_audit_chain_lock = threading.Lock()


def _chain_timestamp(created_at: dt.datetime | None) -> str:
    """An event's created_at as the chain hashes it: the instant, in UTC."""
    if created_at is None:
        return ""
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=dt.timezone.utc)
    return created_at.astimezone(dt.timezone.utc).isoformat()


def _legacy_chain_timestamps(created_at: dt.datetime | None, server_tz) -> list[str]:
    """Other renderings of the same instant that earlier releases hashed.

    Releases running on SQLite hashed the naive UTC value; releases on
    PostgreSQL before sessions were pinned to UTC hashed it in the server's
    own TimeZone. Each names exactly the same instant, so accepting them adds
    no way to alter an event: its content and the signed hash still must match.
    """
    if created_at is None:
        return []
    utc = created_at.astimezone(dt.timezone.utc) if created_at.tzinfo else created_at
    forms = [utc.replace(tzinfo=None).isoformat()]
    if server_tz is not None:
        forms.append(utc.replace(tzinfo=dt.timezone.utc).astimezone(server_tz).isoformat())
    return forms


def _server_timezone(db: Session):
    """The server's configured TimeZone (what sessions used before UTC pinning)."""
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    name = db.execute(text("SELECT reset_val FROM pg_settings WHERE name = 'TimeZone'")).scalar()
    try:
        return ZoneInfo(name) if name else None
    except (ZoneInfoNotFoundError, ValueError):
        return None


def _chain_hash(audit_log_id: str, created_at: str, event_payload: dict, previous_hash: str) -> str:
    return hashlib.sha256(json.dumps({
        "audit_log_id": audit_log_id, "created_at": created_at,
        "payload": event_payload, "previous_hash": previous_hash,
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_audit_log(db: Session, user_id: str | None, action: str,
                     resource_type: str | None = None, resource_id: str | None = None,
                     metadata: dict | None = None) -> models.AuditLog:
    metadata = metadata or {}
    # Must happen before signing — see ServerIdentity.ensure_registered().
    server_identity.ensure_registered(db)
    payload = json.dumps({
        "action": action, "user_id": user_id, "resource_type": resource_type,
        "resource_id": resource_id, "metadata": metadata,
    }, sort_keys=True).encode()
    signature = server_identity.sign(payload)
    signing_key_id = server_identity.key_id
    # Persist whatever the caller left pending first, so a failed audit write
    # can never roll the caller's own changes back with it.
    db.commit()

    # The ML-DSA signature protects an individual event; the hash chain makes
    # deletion, modification and reordering observable when the history is
    # verified. Each link must point at the link sequenced immediately before
    # it, so choosing the sequence number, reading the chain head and
    # inserting the new link form one critical section: the lock is taken
    # before any of them, and the event and its link commit together.
    #
    # Every process waits on this section, so it caps audited writes per
    # second across the deployment: it makes as few round trips as it can
    # and does no ORM work.
    with _audit_chain_lock:
        try:
            # Lock, then read the head, in one round trip. The statements of a
            # multi-statement query run in order, each with its own snapshot,
            # so the head is read only once the lock is held and sees every
            # link committed before it. The lock is transaction-scoped:
            # released by the commit below.
            cursor = db.connection().connection.cursor()
            try:
                cursor.execute(
                    "SELECT pg_advisory_xact_lock(hashtext('quantumsentinel_audit_chain'));"
                    " SELECT sequence, entry_hash FROM audit_chain_links ORDER BY sequence DESC LIMIT 1"
                )
                cursor.nextset()
                previous = cursor.fetchone()
            finally:
                cursor.close()
            sequence = previous[0] + 1 if previous else 1
            previous_hash = previous[1] if previous else "0" * 64
            # The hash covers created_at as stored: timestamptz keeps every
            # microsecond of a Python datetime, so audit_chain_status reads
            # back this same instant.
            log_id, created_at = models.gen_uuid(), models.utcnow()
            entry_hash = _chain_hash(log_id, _chain_timestamp(created_at),
                                     json.loads(payload.decode()), previous_hash)
            checkpoint = pqc.b64(server_identity.sign(entry_hash.encode()))
            # The event and its link in one statement (a data-modifying CTE);
            # the link's foreign key to the event is still enforced.
            event = insert(models.AuditLog.__table__).values(
                id=log_id, user_id=user_id, action=action, resource_type=resource_type,
                resource_id=resource_id, metadata_json=metadata, pqc_signature=pqc.b64(signature),
                signing_key_id=signing_key_id, created_at=created_at,
            ).cte("audit_event")
            db.execute(insert(models.AuditChainLink.__table__).values(
                id=models.gen_uuid(), sequence=sequence, audit_log_id=log_id,
                previous_hash=previous_hash, entry_hash=entry_hash,
                checkpoint_signature=checkpoint, signing_key_id=signing_key_id,
                created_at=models.utcnow(),
            ).add_cte(event))
            db.commit()
        except Exception:
            db.rollback()
            raise
    return db.get(models.AuditLog, log_id)


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


def audit_chain_status(db: Session) -> dict:
    """Verify link ordering, hashes and ML-DSA checkpoint signatures.

    Returns ``valid`` plus, when invalid, the first failing sequence number
    and why. ``unchained_events`` counts audit events with no chain link,
    which the chain alone cannot vouch for.
    """
    link_count = int(db.execute(select(func.count()).select_from(models.AuditChainLink)).scalar() or 0)
    # NOT EXISTS, not NOT IN: PostgreSQL plans it as an anti join, where
    # NOT IN (subquery) degrades to a per-row subplan scan on large tables.
    unchained = int(db.execute(
        select(func.count()).select_from(models.AuditLog).where(
            ~select(models.AuditChainLink.id).where(
                models.AuditChainLink.audit_log_id == models.AuditLog.id).exists())
    ).scalar() or 0)
    status = {"valid": True, "links": link_count, "unchained_events": unchained,
              "first_invalid_sequence": None, "reason": None}

    def fail(link, reason: str) -> dict:
        status.update(valid=False, first_invalid_sequence=link.sequence, reason=reason)
        return status

    # Every query is made before the chain is streamed, so the stream's
    # server-side cursor is the only statement open while links are checked.
    keys: dict[str, bytes] = {
        key_id: pqc.unb64(public_key) for key_id, public_key in db.execute(
            select(models.ServerSigningKey.key_id, models.ServerSigningKey.public_key))
    }
    server_tz = _server_timezone(db)  # for links hashed in a legacy timestamp form
    # Each link with its event in one streamed query, instead of one query
    # per link and the whole chain held in memory.
    rows = db.execute(
        select(models.AuditChainLink, models.AuditLog)
        .outerjoin(models.AuditLog, models.AuditLog.id == models.AuditChainLink.audit_log_id)
        .order_by(models.AuditChainLink.sequence)
        .execution_options(yield_per=500)
    )
    try:
        previous_hash = "0" * 64
        for link, entry in rows:
            if not entry:
                return fail(link, "audit event missing")
            if link.previous_hash != previous_hash:
                return fail(link, "link does not point at the preceding link")

            # A link signed by a key missing from history falls back to the
            # current server key, as before.
            verification_pk = keys.get(link.signing_key_id) or server_identity.dsa_pk
            if verification_pk is None:
                return fail(link, "no verification key")

            event_payload = {
                "action": entry.action, "user_id": entry.user_id,
                "resource_type": entry.resource_type, "resource_id": entry.resource_id,
                "metadata": entry.metadata_json or {},
            }
            timestamps = [_chain_timestamp(entry.created_at)]
            if _chain_hash(entry.id, timestamps[0], event_payload, previous_hash) != link.entry_hash:
                if not any(_chain_hash(entry.id, ts, event_payload, previous_hash) == link.entry_hash
                           for ts in _legacy_chain_timestamps(entry.created_at, server_tz)):
                    return fail(link, "event content does not match its link hash")
            if not pqc.dsa_verify(verification_pk, link.entry_hash.encode(),
                                  pqc.unb64(link.checkpoint_signature)):
                return fail(link, "checkpoint signature invalid")
            previous_hash = link.entry_hash
    finally:
        # Close the server-side cursor now, also when a failure returns early.
        rows.close()
    return status


def verify_audit_chain(db: Session) -> bool:
    """True when every chain link verifies (see audit_chain_status)."""
    return audit_chain_status(db)["valid"]


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
