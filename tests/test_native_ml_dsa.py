"""ML-DSA-65 through native OpenSSL, with dilithium-py as the fallback.

Both implementations produce FIPS 204 encodings, so every key and signature
made by one must work with the other, and keys stored by earlier releases
(dilithium-py's 4032-byte expanded form) must keep signing and verifying.
"""
import threading

import pytest
from dilithium_py.ml_dsa import ML_DSA_65

from backend.crypto import pqc

MODES = [pytest.param(True, id="native"), pytest.param(False, id="dilithium-py")]


@pytest.fixture(params=MODES)
def mode(request, monkeypatch):
    if request.param and not pqc.NATIVE_ML_DSA:
        pytest.skip("this cryptography build has no native ML-DSA")
    monkeypatch.setattr(pqc, "NATIVE_ML_DSA", request.param)
    return request.param


def test_native_ml_dsa_is_available_with_the_pinned_cryptography():
    # requirements.txt pins cryptography >= 48, whose wheels bundle an OpenSSL
    # with ML-DSA; losing it would silently fall back to ~65x slower verifies.
    assert pqc.NATIVE_ML_DSA


def test_keygen_returns_a_seed_that_signs_and_verifies(mode):
    pk, sk, _ = pqc.dsa_keygen()
    assert (len(pk), len(sk)) == (1952, pqc.DSA_SEED_BYTES)
    assert pqc.dsa_public_key(sk) == pk
    signature, _ = pqc.dsa_sign(sk, b"order")
    assert len(signature) == 3309
    assert pqc.dsa_verify(pk, b"order", signature)
    assert not pqc.dsa_verify(pk, b"order!", signature)


@pytest.mark.parametrize("signer_native", [True, False], ids=["native-signs", "dilithium-signs"])
def test_signatures_cross_verify_between_implementations(monkeypatch, signer_native):
    if not pqc.NATIVE_ML_DSA:
        pytest.skip("this cryptography build has no native ML-DSA")
    pk, sk, _ = pqc.dsa_keygen()
    monkeypatch.setattr(pqc, "NATIVE_ML_DSA", signer_native)
    signature, _ = pqc.dsa_sign(sk, b"checkpoint")
    monkeypatch.setattr(pqc, "NATIVE_ML_DSA", not signer_native)
    assert pqc.dsa_verify(pk, b"checkpoint", signature)
    assert pqc.dsa_public_key(sk) == pk


def test_expanded_keys_from_earlier_releases_still_sign_and_verify(mode):
    pk, expanded_sk = ML_DSA_65.keygen()  # what dsa_keygen returned before
    assert len(expanded_sk) == 4032
    assert pqc.dsa_public_key(expanded_sk) == pk
    signature, _ = pqc.dsa_sign(expanded_sk, b"audit")
    assert pqc.dsa_verify(pk, b"audit", signature)
    # A signature stored by an earlier release verifies too.
    assert pqc.dsa_verify(pk, b"audit", ML_DSA_65.sign(expanded_sk, b"audit"))


def test_malformed_input_is_rejected_as_before(mode):
    pk, sk, _ = pqc.dsa_keygen()
    signature, _ = pqc.dsa_sign(sk, b"m")
    for bad in (b"", signature[:-1], signature + b"\x00", bytes(len(signature))):
        assert pqc.dsa_verify(pk, b"m", bad) is False
    with pytest.raises(ValueError):
        pqc.dsa_verify(pk[:-1], b"m", signature)


def test_concurrent_signatures_all_verify(mode):
    pk, sk, _ = pqc.dsa_keygen()
    results = []

    def signer(n):
        for i in range(4):
            message = f"signer-{n}-{i}".encode()
            signature, _ = pqc.dsa_sign(sk, message)
            results.append(pqc.dsa_verify(pk, message, signature))

    threads = [threading.Thread(target=signer, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == [True] * 16


def test_generate_server_key_prints_a_matching_seed_keypair(capsys):
    import hashlib

    from backend import manage

    assert manage.main(["generate-server-key"]) == 0
    settings = dict(line.split("=", 1) for line in capsys.readouterr().out.splitlines())
    sk = pqc.unb64(settings["SERVER_DSA_PRIVATE_KEY"])
    pk = pqc.unb64(settings["SERVER_DSA_PUBLIC_KEY"])
    assert len(sk) == pqc.DSA_SEED_BYTES and pqc.dsa_public_key(sk) == pk
    assert settings["TRUSTED_SERVER_DSA_FINGERPRINT"] == hashlib.sha256(pk).hexdigest()
    assert settings["SERVER_DSA_CREATED_AT"].endswith("+00:00")
    signature, _ = pqc.dsa_sign(sk, b"audit")
    assert pqc.dsa_verify(pk, b"audit", signature)
