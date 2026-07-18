"""QuantumSentinel — Auth service: password hashing + JWT + PQC handshake sessions."""
import base64
import hashlib
import hmac
import os
import time
import jwt
from sqlalchemy.orm import Session

from .. import models
from ..crypto import pqc
from ..config import JWT_SECRET, JWT_ALGORITHM, JWT_EXPIRE_SECONDS

PBKDF2_ITERATIONS = 200_000

# in-memory PQC session store (mirrors the Redis session:{user_id} pattern)
SESSIONS: dict[str, dict] = {}
NONCES: dict[str, float] = {}


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ITERATIONS)
    return base64.b64encode(salt + digest).decode()


def verify_password(password: str, stored: str) -> bool:
    raw = base64.b64decode(stored)
    salt, digest = raw[:16], raw[16:]
    check = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ITERATIONS)
    return hmac.compare_digest(digest, check)


def create_access_token(user_id: str, tier: str) -> str:
    now = int(time.time())
    payload = {"sub": user_id, "tier": tier, "iat": now, "exp": now + JWT_EXPIRE_SECONDS}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        return None


def perform_handshake(db: Session, user_id: str, client_x25519_pub_b64: str,
                       client_kem_pub_b64: str | None, client_nonce_b64: str) -> dict:
    """Server side of the hybrid X25519 + ML-KEM-768 handshake. If the client
    doesn't supply its own ML-KEM public key (typical for a browser demo
    that can't run lattice crypto natively), the server generates a
    demonstration client-analog keypair so the FULL real protocol still
    executes end-to-end — this is flagged clearly to the caller."""
    from ..services import security_service

    client_x25519_pub = pqc.unb64(client_x25519_pub_b64)
    client_nonce = pqc.unb64(client_nonce_b64)

    simulated_client_kem = False
    if client_kem_pub_b64:
        client_kem_pub = pqc.unb64(client_kem_pub_b64)
    else:
        simulated_client_kem = True
        client_kem_pub, client_kem_sk, _ = pqc.kem_keygen()

    # Server X25519 ephemeral keypair
    server_x25519_pub, server_x25519_sk = pqc.x25519_keygen()
    x25519_shared = pqc.x25519_shared_secret(server_x25519_sk, client_x25519_pub)

    # ML-KEM-768 encapsulation against the client's KEM public key
    kem_ciphertext, kem_shared, kem_ms = pqc.kem_encapsulate(client_kem_pub)

    server_nonce = os.urandom(32)
    session_key = pqc.derive_session_key(x25519_shared, kem_shared, client_nonce, server_nonce)

    server_hello_payload = (
        server_x25519_pub + kem_ciphertext + client_nonce + server_nonce
    )
    signature = security_service.server_identity.sign(server_hello_payload)

    session_id = pqc.b64(os.urandom(16))
    SESSIONS[session_id] = {
        "user_id": user_id, "session_key": session_key,
        "created_at": time.time(), "expires_at": time.time() + 3600,
    }

    return {
        "session_id": session_id,
        "server_x25519_public_key": pqc.b64(server_x25519_pub),
        "ml_kem_ciphertext": pqc.b64(kem_ciphertext),
        "server_nonce": pqc.b64(server_nonce),
        "ml_dsa_signature": pqc.b64(signature),
        "server_dsa_public_key": pqc.b64(security_service.server_identity.dsa_pk),
        "session_token": pqc.session_token(session_key, client_nonce, server_nonce),
        "kem_encapsulate_ms": round(kem_ms, 3),
        "simulated_client_kem_keypair": simulated_client_kem,
        "algorithm_sizes": {
            "x25519_shared_secret_bytes": len(x25519_shared),
            "ml_kem_ciphertext_bytes": len(kem_ciphertext),
            "ml_kem_shared_secret_bytes": len(kem_shared),
            "ml_dsa_signature_bytes": len(signature),
            "session_key_bytes": len(session_key),
        },
    }
