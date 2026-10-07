"""QuantumSentinel — PQC Crypto Layer.

Real NIST FIPS 203 (ML-KEM-768) and FIPS 204 (ML-DSA-65) reference algorithms,
via the pure-Python `kyber-py` / `dilithium-py` packages (used in production
liboqs test-vector validation). Byte sizes match the spec exactly:
  ML-KEM-768:  pk 1184B, sk 2400B, ciphertext 1088B, shared-secret 32B
  ML-DSA-65:   pk 1952B, sk 4032B, signature 3309B

A pure-Python implementation is slower than liboqs' C/AVX2 code (~28ms keygen
vs ~10µs), but it is byte-for-byte spec compliant and requires no compiled
system library — ideal for a portable, auditable web deployment. ML-DSA uses
the native OpenSSL implementation through `cryptography` when available (see
the ML-DSA section below).

Classical leg of the hybrid handshake uses X25519 from `cryptography`
(OpenSSL-backed). Session key = HKDF-SHA256(X25519_secret || ML-KEM_secret).
"""
import base64
import functools
import hashlib
import hmac
import os
import threading
import time

from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey, X25519PublicKey,
)
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes

from kyber_py.ml_kem import ML_KEM_768
from dilithium_py.ml_dsa import ML_DSA_65
from ..config import ENVIRONMENT, PQC_PROVIDER, PQC_PROVIDER_URL

# Passed as HKDF's `info` parameter (domain-separation context), NOT `salt` —
# the per-session client/server nonces below serve as the actual salt. Named
# for what HKDF's RFC 5869 calls this parameter's *purpose* (protocol/context
# binding), not its keyword name, to avoid it being misread as the salt input.
HKDF_INFO_CONTEXT = b"QuantumSentinel-v1"
HANDSHAKE_PROTOCOL_VERSION = "QS-HANDSHAKE-V2"


def _assert_pqc_backend():
    """Guard: in development, the pure-Python reference packages are allowed.
    In production, an external liboqs/HSM adapter must be configured;
    otherwise we refuse to proceed rather than silently use slower,
    unreviewed reference code in a live environment.
    """
    if ENVIRONMENT == "production" and (PQC_PROVIDER == "reference" or not PQC_PROVIDER_URL):
        raise RuntimeError(
            f"Production PQC requires a configured external provider. "
            f"Set PQC_PROVIDER and PQC_PROVIDER_URL; currently: "
            f"provider={PQC_PROVIDER!r}, url={PQC_PROVIDER_URL!r}"
        )
    # Development/staging: allow pure-Python reference implementations (kyber-py / dilithium-py).


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
# Signatures and public keys are the FIPS 204 encodings whichever
# implementation handles them, so either one verifies the other's signatures.
# Native ML-DSA (`cryptography` >= 48 on OpenSSL >= 3.5, which its wheels
# bundle) verifies ~65x and signs ~30x faster than dilithium-py and is used
# whenever the installed build supports it; dilithium-py remains the fallback.
#
# Private keys come in two forms. A 32-byte seed (FIPS 204's xi, what
# dsa_keygen returns) works with both implementations. The 4032-byte expanded
# key that earlier releases generated can only be loaded by dilithium-py, so
# keys already stored in that form keep signing through it.
DSA_SEED_BYTES = 32

try:
    from cryptography.exceptions import InvalidSignature as _InvalidSignature
    from cryptography.hazmat.primitives.asymmetric import mldsa as _mldsa

    # The module exists from cryptography 47, but only works on a backend
    # with ML-DSA support: prove a sign/verify round trip before relying on it.
    _probe_key = _mldsa.MLDSA65PrivateKey.generate()
    _probe_key.public_key().verify(_probe_key.sign(b"probe"), b"probe")
    NATIVE_ML_DSA = True
    del _probe_key
except Exception:  # ImportError, UnsupportedAlgorithm, or a backend fault
    NATIVE_ML_DSA = False

# Without the optional `xoflib` package, dilithium-py hashes through one
# module-level SHAKE-256 object whose buffer is shared by every caller, so two
# threads signing or verifying at once read each other's XOF stream and
# produce invalid signatures. Serialise all dilithium-py operations; the
# pure-Python implementation holds the GIL throughout, so this costs no
# parallelism. The native implementation is thread-safe and needs no lock.
_ml_dsa_lock = threading.Lock()


def dsa_keygen():
    """A new keypair: (public key, 32-byte seed private key, elapsed ms)."""
    _assert_pqc_backend()
    t0 = time.perf_counter()
    if NATIVE_ML_DSA:
        key = _mldsa.MLDSA65PrivateKey.generate()
        pk, sk = key.public_key().public_bytes_raw(), key.private_bytes_raw()
    else:
        sk = os.urandom(DSA_SEED_BYTES)
        with _ml_dsa_lock:
            pk, _expanded = ML_DSA_65.key_derive(sk)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return pk, sk, elapsed_ms


def dsa_public_key(sk: bytes) -> bytes:
    """The public key belonging to a private key in either form."""
    if len(sk) == DSA_SEED_BYTES:
        if NATIVE_ML_DSA:
            return _mldsa.MLDSA65PrivateKey.from_seed_bytes(sk).public_key().public_bytes_raw()
        with _ml_dsa_lock:
            return ML_DSA_65.key_derive(sk)[0]
    with _ml_dsa_lock:
        return ML_DSA_65.pk_from_sk(sk)


@functools.lru_cache(maxsize=16)
def _native_signing_key(seed: bytes):
    # Expanding a seed costs ~15% of a signature; the server signs with one key.
    return _mldsa.MLDSA65PrivateKey.from_seed_bytes(seed)


def dsa_sign(sk: bytes, message: bytes):
    _assert_pqc_backend()
    t0 = time.perf_counter()
    if len(sk) == DSA_SEED_BYTES and NATIVE_ML_DSA:
        signature = _native_signing_key(bytes(sk)).sign(message)
    else:
        with _ml_dsa_lock:
            if len(sk) == DSA_SEED_BYTES:
                sk = ML_DSA_65.key_derive(sk)[1]
            signature = ML_DSA_65.sign(sk, message)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return signature, elapsed_ms


@functools.lru_cache(maxsize=256)
def _native_public_key(pk: bytes):
    # Parsing a public key costs ~10% of a verification, and verifiers check
    # many signatures from few keys (every audit link is signed by a server key).
    return _mldsa.MLDSA65PublicKey.from_public_bytes(pk)


def dsa_verify(pk: bytes, message: bytes, signature: bytes) -> bool:
    """True when ``signature`` is valid. A malformed signature is simply
    invalid; a public key of the wrong length raises ValueError, as
    dilithium-py does."""
    _assert_pqc_backend()
    if NATIVE_ML_DSA:
        public_key = _native_public_key(bytes(pk))
        try:
            public_key.verify(signature, message)
        except _InvalidSignature:
            return False
        return True
    with _ml_dsa_lock:
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
    hkdf = HKDF(algorithm=hashes.SHA256(), length=32, salt=salt, info=HKDF_INFO_CONTEXT)
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
