"""QuantumSentinel — PQC Crypto Layer.

Real NIST FIPS 203 (ML-KEM-768) and FIPS 204 (ML-DSA-65) reference algorithms,
via the pure-Python `kyber-py` / `dilithium-py` packages (used in production
liboqs test-vector validation). Byte sizes match the spec exactly:
  ML-KEM-768:  pk 1184B, sk 2400B, ciphertext 1088B, shared-secret 32B
  ML-DSA-65:   pk 1952B, sk 4032B, signature 3309B

A pure-Python implementation is slower than liboqs' C/AVX2 code (~28ms keygen
vs ~10µs), but it is byte-for-byte spec compliant and requires no compiled
system library — ideal for a portable, auditable web deployment.

Classical leg of the hybrid handshake uses X25519 from `cryptography`
(OpenSSL-backed). Session key = HKDF-SHA256(X25519_secret || ML-KEM_secret).
"""
import base64
import hashlib
import hmac
import time

from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey, X25519PublicKey,
)
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes

from kyber_py.ml_kem import ML_KEM_768
from dilithium_py.ml_dsa import ML_DSA_65
from ..config import ENVIRONMENT, PQC_PROVIDER, PQC_PROVIDER_URL

HKDF_SALT_CONTEXT = b"QuantumSentinel-v1"


def _assert_pqc_backend():
    if ENVIRONMENT == "production":
        # The bundled packages are reference implementations. An external
        # provider adapter must be integrated before production crypto calls;
        # never silently downgrade to Python reference code.
        raise RuntimeError(
            f"PQC provider '{PQC_PROVIDER}' at {PQC_PROVIDER_URL!r} requires the reviewed external adapter"
        )


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def unb64(data: str) -> bytes:
    return base64.b64decode(data)


# --------------------------------------------------------------------------
# ML-KEM-768 (FIPS 203)
# --------------------------------------------------------------------------
def kem_keygen():
    _assert_pqc_backend()
    t0 = time.perf_counter()
    pk, sk = ML_KEM_768.keygen()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return pk, sk, elapsed_ms


def kem_encapsulate(pk: bytes):
    _assert_pqc_backend()
    t0 = time.perf_counter()
    shared_secret, ciphertext = ML_KEM_768.encaps(pk)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return ciphertext, shared_secret, elapsed_ms


def kem_decapsulate(sk: bytes, ciphertext: bytes):
    _assert_pqc_backend()
    t0 = time.perf_counter()
    shared_secret = ML_KEM_768.decaps(sk, ciphertext)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return shared_secret, elapsed_ms


# --------------------------------------------------------------------------
# ML-DSA-65 (FIPS 204)
# --------------------------------------------------------------------------
def dsa_keygen():
    _assert_pqc_backend()
    t0 = time.perf_counter()
    pk, sk = ML_DSA_65.keygen()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return pk, sk, elapsed_ms


def dsa_sign(sk: bytes, message: bytes):
    _assert_pqc_backend()
    t0 = time.perf_counter()
    signature = ML_DSA_65.sign(sk, message)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return signature, elapsed_ms


def dsa_verify(pk: bytes, message: bytes, signature: bytes) -> bool:
    _assert_pqc_backend()
    return ML_DSA_65.verify(pk, message, signature)


# --------------------------------------------------------------------------
# Hybrid Handshake: X25519 (classical) + ML-KEM-768 (post-quantum)
# --------------------------------------------------------------------------
def x25519_keygen():
    sk = X25519PrivateKey.generate()
    pk = sk.public_key()
    pk_bytes = pk.public_bytes_raw()
    sk_bytes = sk.private_bytes_raw()
    return pk_bytes, sk_bytes


def x25519_shared_secret(private_key_bytes: bytes, peer_public_key_bytes: bytes) -> bytes:
    sk = X25519PrivateKey.from_private_bytes(private_key_bytes)
    pk = X25519PublicKey.from_public_bytes(peer_public_key_bytes)
    return sk.exchange(pk)


def derive_session_key(x25519_shared: bytes, ml_kem_shared: bytes,
                        client_nonce: bytes, server_nonce: bytes) -> bytes:
    """session_key = HKDF-SHA256(X25519_secret || ML-KEM_secret, salt=client||server nonce)."""
    combined_ikm = x25519_shared + ml_kem_shared
    salt = client_nonce + server_nonce
    hkdf = HKDF(algorithm=hashes.SHA256(), length=32, salt=salt, info=HKDF_SALT_CONTEXT)
    return hkdf.derive(combined_ikm)


def session_token(session_key: bytes, client_nonce: bytes, server_nonce: bytes) -> str:
    mac = hmac.new(session_key, client_nonce + server_nonce, hashlib.sha256).digest()
    return b64(mac)


# --------------------------------------------------------------------------
# Algorithm Registry — crypto-agility (Section 5.5 of the architecture doc)
# --------------------------------------------------------------------------
ALGORITHM_REGISTRY = {
    "ML-KEM-768": {
        "type": "KEM", "security_level": 3, "public_key_size": 1184,
        "secret_key_size": 2400, "ct_or_sig_size": 1088, "default": True,
        "fips_standard": "FIPS 203",
    },
    "ML-DSA-65": {
        "type": "SIG", "security_level": 3, "public_key_size": 1952,
        "secret_key_size": 4032, "ct_or_sig_size": 3309, "default": True,
        "fips_standard": "FIPS 204",
    },
    "X25519": {
        "type": "KEM-classical", "security_level": None, "public_key_size": 32,
        "secret_key_size": 32, "ct_or_sig_size": 32, "default": True,
        "fips_standard": "RFC 7748 (hybrid leg)",
    },
}
