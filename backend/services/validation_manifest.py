"""QuantumSentinel — strategy validation manifests (manifest v3, R3-4c).

A strategy validation's manifest says what its verdict was reached on, in
one signed document: the candidate strategy version, the evaluation it
judged (with that evaluation's return-series hash and datasets), the
dataset family's holdout start when the verdict was reached, the policy,
the trial matrix, the verdict itself (by hash), the event engine's version,
the dependency lock, the code commit and the key that signed it.

The worker builds and signs it in the transaction that records the verdict
(``record``, called from ``strategy_validation.finalize_validation_job``)
and stores it once, in ``validation_manifests`` (migration 0009):

* ``canonical`` is the manifest's canonical JSON (``dataset_format.
  canonical_json``, RFC 8785 on strings, integers, booleans and nulls only),
  and the manifest's id is its hash: ``qs-vm-sha256-v1:<sha256>``.
* ``signature`` signs exactly those bytes: ``ml-dsa:<hex>`` with the server's
  ML-DSA-65 key, named by ``signing_key_id`` (a row of
  ``server_signing_keys``, so the key that signed it stays resolvable after
  it is rotated). Outside production, if ML-DSA is unavailable the manifest
  is signed ``hmac-sha256:<hex>`` with no key id; in production that is
  refused, and the job fails.

The verdict holds floats, which canonical JSON refuses, so the manifest binds
it by ``verdict_sha256``: the SHA-256 of its JSON with sorted keys, no
whitespace, unescaped Unicode and no NaN (``VERDICT_SCHEME``). A float
survives a JSON round trip exactly, so the verdict read back from the
database hashes the same.

Manifest v3 is separate from the experiment manifest (v2,
``experiment_registry.build_manifest``), which is unchanged. Verdicts
recorded before this module have no manifest, and none is made for them.
"""
from __future__ import annotations

import datetime as dt
import functools
import hashlib
import hmac
import json
import logging
import re
import subprocess  # nosec B404 - runs git with fixed arguments only
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from .. import models
from . import dataset_format

log = logging.getLogger(__name__)

MANIFEST_VERSION = 3
MANIFEST_KIND = "qs-strategy-validation"
MANIFEST_SCHEME = "qs-vm-sha256-v1"
VERDICT_SCHEME = "qs-verdict-json-sha256-v1"
_COMMIT = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_REPO_ROOT = Path(__file__).resolve().parents[2]


class ManifestError(Exception):
    """A manifest that cannot be built or signed."""


# --------------------------------------------------------------------------
# What the manifest records
# --------------------------------------------------------------------------

def verdict_sha256(verdict: dict) -> str:
    """The hash a manifest binds its verdict by (``VERDICT_SCHEME``)."""
    text = json.dumps(verdict, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    if json.loads(text) != verdict:
        raise ManifestError("the verdict does not survive a JSON round trip")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _commit_or_none(value: str | None, source: str) -> str | None:
    value = (value or "").strip().lower()
    if not value:
        return None
    if _COMMIT.fullmatch(value) is None:
        log.warning("ignoring a code commit from %s that is not a 40- or 64-digit hex hash", source)
        return None
    return value


@functools.lru_cache(maxsize=1)
def code_commit() -> str | None:
    """The commit this code was built from: QS_CODE_COMMIT (baked into the
    image at build time), else the checkout's git HEAD, else None. A value
    that is not a full commit hash counts as none."""
    from backend import config

    if (config.CODE_COMMIT or "").strip():
        return _commit_or_none(config.CODE_COMMIT, "QS_CODE_COMMIT")
    try:
        run = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,  # nosec B603 B607
                             timeout=5, cwd=_REPO_ROOT)
    except (OSError, subprocess.SubprocessError):
        return None
    return _commit_or_none(run.stdout, "git") if run.returncode == 0 else None


def dependency_lock_sha256() -> str | None:
    from .experiment_registry import dependency_lock_hash

    value = dependency_lock_hash()
    return None if value == "unknown" else value


def build(*, job_id: str, created_at: dt.datetime, evaluation: models.ResearchTrialEvaluation,
          dataset_family: str, holdout_start: dt.date | None, policy_id: str, matrix_id: str,
          verdict: dict, signing_key_id: str | None) -> dict:
    """The manifest of one verdict (every value a string, bool, int or None)."""
    from .event_simulator import ENGINE_VERSION

    return {
        "manifest_version": MANIFEST_VERSION,
        "kind": MANIFEST_KIND,
        "validation_job_id": job_id,
        "created_at": created_at.astimezone(dt.timezone.utc).isoformat(),
        "strategy_version_id": evaluation.strategy_version_id,
        "evaluation": {
            "job_id": evaluation.job_id,
            "research_family": evaluation.family_hash,
            "returns_sha256": evaluation.returns_sha256,
            "dataset_ids": sorted(evaluation.dataset_ids),
        },
        "dataset_family": dataset_family,
        "dataset_holdout_start": holdout_start.isoformat() if holdout_start is not None else None,
        "policy_id": policy_id,
        "matrix_id": matrix_id,
        "verdict": {"passed": verdict["passed"], "scheme": VERDICT_SCHEME, "sha256": verdict_sha256(verdict)},
        "engine_version": ENGINE_VERSION,
        "dependency_lock_sha256": dependency_lock_sha256(),
        "code_commit": code_commit(),
        "signing_key_id": signing_key_id,
    }


def manifest_id(canonical: bytes) -> str:
    return f"{MANIFEST_SCHEME}:{hashlib.sha256(canonical).hexdigest()}"


# --------------------------------------------------------------------------
# Signing
# --------------------------------------------------------------------------

def _hmac_key() -> bytes:
    from .experiment_registry import _fallback_hmac_key
    return _fallback_hmac_key()


def _production() -> bool:
    from backend import config
    return config.ENVIRONMENT == "production"


def _signing_key_id(db: Session) -> str:
    """The id of the server key in ``db``, registered in the caller's
    transaction if it is not there yet.

    ``ServerIdentity.register_in_db`` commits, which would commit the
    finalizer's half-written outcome, so this inserts the row without
    committing: it is kept only if the verdict is. The identity's own state
    is left alone, so a rolled-back registration leaves nothing it believes
    is stored."""
    from . import security_service
    from backend.crypto import pqc

    identity = security_service.server_identity
    if identity.dsa_sk is None:
        identity.sign(b"")  # generates the key
    public_key, fingerprint = identity.dsa_pk, identity.fingerprint
    db.execute(insert(models.ServerSigningKey).values(
        key_id=identity.key_id or models.gen_uuid(), algorithm="ML-DSA-65", public_key=pqc.b64(public_key),
        fingerprint=fingerprint, status="active", created_at=dt.datetime.now(dt.timezone.utc),
        activated_at=identity.created_at).on_conflict_do_nothing(index_elements=["fingerprint"]))
    return db.execute(select(models.ServerSigningKey.key_id)
                      .where(models.ServerSigningKey.fingerprint == fingerprint)).scalar_one()


def _sign(canonical: bytes, signing_key_id: str | None) -> str:
    if signing_key_id is None:
        return "hmac-sha256:" + hmac.new(_hmac_key(), canonical, hashlib.sha256).hexdigest()
    from . import security_service
    return "ml-dsa:" + security_service.server_identity.sign(canonical).hex()


def record(db: Session, *, job: models.ResearchJob, created_at: dt.datetime,
           evaluation: models.ResearchTrialEvaluation, dataset_family: str, policy_id: str, matrix_id: str,
           verdict: dict) -> models.ValidationManifest:
    """Build, sign and store the manifest of the verdict just recorded for
    ``job``, in the caller's transaction."""
    from . import dataset_holdouts

    try:
        with db.begin_nested():  # a failed registration must not abort the transaction
            key_id = _signing_key_id(db)
    except Exception as exc:
        if _production():
            raise ManifestError("the server's ML-DSA key is unavailable") from exc
        log.warning("strategy validation %s: ML-DSA unavailable, signing its manifest with HMAC: %s", job.id, exc)
        key_id = None
    manifest = build(job_id=job.id, created_at=created_at, evaluation=evaluation, dataset_family=dataset_family,
                     holdout_start=dataset_holdouts.current(db).get(dataset_family), policy_id=policy_id,
                     matrix_id=matrix_id, verdict=verdict, signing_key_id=key_id)
    canonical = dataset_format.canonical_json(manifest)
    row = models.ValidationManifest(manifest_id=manifest_id(canonical), job_id=job.id, canonical=canonical,
                                    signature=_sign(canonical, key_id), signing_key_id=key_id,
                                    created_at=created_at)
    db.add(row)
    db.flush()
    return row


# --------------------------------------------------------------------------
# Reading and verifying
# --------------------------------------------------------------------------

def get(db: Session, job_id: str) -> models.ValidationManifest | None:
    return db.execute(select(models.ValidationManifest)
                      .where(models.ValidationManifest.job_id == job_id)).scalar_one_or_none()


def signature_valid(db: Session, row: models.ValidationManifest) -> bool:
    """Whether ``row.signature`` signs ``row.canonical``, with the key the
    manifest names (an HMAC signature is never valid in production)."""
    signature, canonical = row.signature or "", bytes(row.canonical)
    if signature.startswith("hmac-sha256:"):
        if _production() or row.signing_key_id is not None:
            return False
        expected = "hmac-sha256:" + hmac.new(_hmac_key(), canonical, hashlib.sha256).hexdigest()
        return hmac.compare_digest(signature, expected)
    if not signature.startswith("ml-dsa:") or row.signing_key_id is None:
        return False
    key = db.get(models.ServerSigningKey, row.signing_key_id)
    if key is None:
        return False
    from backend.crypto import pqc
    try:
        return bool(pqc.dsa_verify(pqc.unb64(key.public_key), canonical, bytes.fromhex(signature[len("ml-dsa:"):])))
    except Exception:
        return False


def matches(row: models.ValidationManifest, validation: models.StrategyValidation, manifest: dict) -> bool:
    """Whether the manifest is this verdict's: its id is its hash, and it
    names the verdict's job, evaluation, candidate, policy, matrix and
    verdict (re-hashed from the verdict as stored)."""
    try:
        verdict_hash = verdict_sha256(validation.verdict)
    except (ManifestError, ValueError, TypeError):
        return False
    evaluation = manifest.get("evaluation") if isinstance(manifest.get("evaluation"), dict) else {}
    return (row.manifest_id == manifest_id(bytes(row.canonical))
            and manifest.get("manifest_version") == MANIFEST_VERSION
            and manifest.get("validation_job_id") == validation.job_id == row.job_id
            and manifest.get("signing_key_id") == row.signing_key_id
            and manifest.get("strategy_version_id") == validation.strategy_version_id
            and evaluation.get("job_id") == validation.evaluation_job_id
            and evaluation.get("research_family") == validation.family_hash
            and manifest.get("policy_id") == validation.policy_id
            and manifest.get("matrix_id") == validation.matrix_id
            and manifest.get("verdict") == {"passed": validation.passed, "scheme": VERDICT_SCHEME,
                                            "sha256": verdict_hash})


def view(db: Session, validation: models.StrategyValidation) -> dict:
    """The manifest of one verdict, with whether its signature verifies and
    whether it is this verdict's; ``manifest`` is None for a verdict
    recorded before manifests existed."""
    row = get(db, validation.job_id)
    if row is None:
        return {"job_id": validation.job_id, "manifest_id": None, "manifest": None, "canonical": None,
                "signature": None, "signature_valid": None, "matches_verdict": None,
                "reason": "This validation was recorded before validation manifests existed, so it has none"}
    canonical = bytes(row.canonical)
    manifest = json.loads(canonical)
    return {"job_id": validation.job_id, "manifest_id": row.manifest_id, "manifest": manifest,
            "canonical": canonical.decode("utf-8"), "signature": row.signature,
            "signature_valid": signature_valid(db, row), "matches_verdict": matches(row, validation, manifest),
            "reason": None}
