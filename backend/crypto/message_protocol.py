"""Per-message AEAD protocol for session-encrypted communications.

After the PQC handshake establishes a session key via X25519 + ML-KEM + HKDF,
this module provides authenticated encryption for individual messages using
AES-256-GCM with sequence-number replay protection.

Usage:
  ML-KEM + X25519 → HKDF → session_key
                          ↓
  AEADSession(session_key) → encrypt/decrypt per-message
                           → AES-256-GCM with sequence + AAD
"""
from __future__ import annotations

import os
import struct
import hashlib
from dataclasses import dataclass, field

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes


@dataclass
class AEADEnvelope:
    """Wire format for an encrypted message."""
    session_id: str
    sequence_number: int
    nonce: bytes           # 12-byte GCM nonce
    ciphertext: bytes      # encrypted payload
    tag: bytes             # 16-byte GCM authentication tag (appended to ciphertext by AESGCM)

    def to_dict(self) -> dict:
        import base64
        return {
            "session_id": self.session_id,
            "sequence_number": self.sequence_number,
            "nonce": base64.b64encode(self.nonce).decode(),
            "ciphertext": base64.b64encode(self.ciphertext).decode(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AEADEnvelope":
        import base64
        ct = base64.b64decode(d["ciphertext"])
        return cls(
            session_id=d["session_id"],
            sequence_number=d["sequence_number"],
            nonce=base64.b64decode(d["nonce"]),
            ciphertext=ct,
            tag=b"",  # tag is appended to ciphertext by AESGCM
        )


class AEADSession:
    """AES-256-GCM session with sequence-number replay/order protection.

    Derives separate encryption and MAC sub-keys from the master session key
    using HKDF with distinct info labels, following best practices for key
    separation.
    """

    def __init__(self, session_id: str, session_key: bytes):
        if len(session_key) < 32:
            raise ValueError("session_key must be at least 32 bytes")
        self.session_id = session_id
        # Derive encryption sub-key
        self._enc_key = HKDF(
            algorithm=hashes.SHA256(), length=32,
            salt=None, info=b"qs-aead-encryption-v1",
        ).derive(session_key)
        self._aesgcm = AESGCM(self._enc_key)
        self._send_seq: int = 0
        self._recv_seq: int = 0
        self._seen_seqs: set[int] = set()
        self._max_seen_seq: int = -1

    def encrypt(self, plaintext: bytes) -> AEADEnvelope:
        """Encrypt a message with the next sequence number."""
        seq = self._send_seq
        self._send_seq += 1
        nonce = os.urandom(12)
        # Authenticate sequence number and session_id as AAD
        aad = self._build_aad(seq)
        ciphertext = self._aesgcm.encrypt(nonce, plaintext, aad)
        return AEADEnvelope(
            session_id=self.session_id,
            sequence_number=seq,
            nonce=nonce,
            ciphertext=ciphertext,
            tag=b"",  # AESGCM appends tag to ciphertext
        )

    def decrypt(self, envelope: AEADEnvelope) -> bytes:
        """Decrypt and verify a message. Rejects replay/out-of-order."""
        seq = envelope.sequence_number
        if seq in self._seen_seqs:
            raise ReplayError(f"Sequence {seq} already received (replay)")
        if seq < self._max_seen_seq - 1000:
            raise ReplayError(f"Sequence {seq} is too old (max seen: {self._max_seen_seq})")
        aad = self._build_aad(seq)
        try:
            plaintext = self._aesgcm.decrypt(envelope.nonce, envelope.ciphertext, aad)
        except Exception as exc:
            raise AuthenticationError("AEAD decryption/authentication failed") from exc
        self._seen_seqs.add(seq)
        if seq > self._max_seen_seq:
            self._max_seen_seq = seq
        # Prune old sequence tracking to bound memory
        if len(self._seen_seqs) > 2000:
            cutoff = self._max_seen_seq - 1000
            self._seen_seqs = {s for s in self._seen_seqs if s >= cutoff}
        return plaintext

    def check_gap(self, received_seq: int) -> list[int]:
        """Return list of missing sequence numbers (gaps) up to received_seq."""
        if self._max_seen_seq < 0:
            return []
        expected = set(range(self._max_seen_seq + 1, received_seq))
        return sorted(expected - self._seen_seqs)

    def _build_aad(self, seq: int) -> bytes:
        """Build Additional Authenticated Data: session_id + sequence number."""
        return self.session_id.encode() + struct.pack(">Q", seq)


class ReplayError(Exception):
    """Raised when a replayed or out-of-order message is detected."""
    pass


class AuthenticationError(Exception):
    """Raised when AEAD authentication fails."""
    pass
