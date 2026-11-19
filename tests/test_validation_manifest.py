"""Strategy validation manifests (manifest v3, R3-4c): the signed manifest
the worker records with each verdict (services/validation_manifest.py,
migration 0009), and the endpoint that serves it.

The job tests run validations end to end on the synthetic market of
test_trial_evaluations and check every field of the manifest against an
independent source: the evaluation row, the verdict read back from the
database, requirements.txt, git, the holdout registry and the signing key's
row.
"""
import datetime as dt
import hashlib
import hmac
import json
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy import select, text
from sqlalchemy.exc import DBAPIError

from backend import config, main, models
from backend.crypto import pqc
from backend.services import (dataset_format, dataset_holdouts, event_simulator, security_service,
                              strategy_validation, validation_manifest)
from backend.services.research_tasks import TaskError
from test_strategy_validation import CURRENT_ID, rows, validate
from test_trial_evaluations import Session, db, make_user, market  # noqa: F401
from test_trial_matrix import FAMILY, evaluated

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def fresh_commit_cache():
    validation_manifest.code_commit.cache_clear()
    yield
    validation_manifest.code_commit.cache_clear()


def validated(db, user=None, job_process=None):
    user = user or make_user(db)
    candidate, view = evaluated(db, user)
    done = validate(db, user, view["job_id"], job_process)
    return user, candidate, view, done


def manifest_row(db, job_id):
    db.expire_all()
    return db.execute(select(models.ValidationManifest)
                      .where(models.ValidationManifest.job_id == job_id)).scalar_one_or_none()


def git_head():
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO,
                          check=True).stdout.strip()


# --------------------------------------------------------------------------
# The manifest a validation records
# --------------------------------------------------------------------------

def test_a_validation_records_a_signed_manifest_of_what_its_verdict_was_reached_on(db, market):
    user, candidate, view, done = validated(db)
    assert done["status"] == "succeeded", done
    [verdict_row] = rows(db, models.StrategyValidation)
    row = manifest_row(db, done["job_id"])
    evaluation = db.get(models.ResearchTrialEvaluation, view["job_id"])

    canonical = bytes(row.canonical)
    manifest = json.loads(canonical)
    # The id is the hash of the stored bytes, which are the manifest's canonical JSON.
    assert row.manifest_id == f"qs-vm-sha256-v1:{hashlib.sha256(canonical).hexdigest()}"
    assert dataset_format.canonical_json(manifest) == canonical

    stored_verdict = verdict_row.verdict  # as read back from the database
    verdict_text = json.dumps(stored_verdict, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                              allow_nan=False)
    requirements = (REPO / "requirements.txt").read_bytes().replace(b"\r\n", b"\n")
    key = db.execute(select(models.ServerSigningKey).where(
        models.ServerSigningKey.fingerprint == security_service.server_identity.fingerprint)).scalar_one()
    assert manifest == {
        "manifest_version": 3,
        "kind": "qs-strategy-validation",
        "validation_job_id": done["job_id"],
        "created_at": verdict_row.created_at.astimezone(dt.timezone.utc).isoformat(),
        "strategy_version_id": candidate,
        "evaluation": {"job_id": view["job_id"], "research_family": FAMILY,
                       "returns_sha256": evaluation.returns_sha256,
                       "dataset_ids": sorted(view["result"]["market_data"]["datasets"])},
        "dataset_family": "EQUITY_EOD",
        "dataset_holdout_start": None,
        "policy_id": CURRENT_ID,
        "matrix_id": verdict_row.matrix_id,
        "verdict": {"passed": False, "scheme": "qs-verdict-json-sha256-v1",
                    "sha256": hashlib.sha256(verdict_text.encode()).hexdigest()},
        "engine_version": "qs-event-engine-1",
        "dependency_lock_sha256": hashlib.sha256(requirements).hexdigest(),
        "code_commit": git_head(),
        "signing_key_id": key.key_id,
    }
    assert (row.job_id, row.signing_key_id, row.created_at) == (done["job_id"], key.key_id, verdict_row.created_at)
    # The signature is ML-DSA over exactly the stored bytes, with the key the manifest names.
    assert row.signature.startswith("ml-dsa:")
    assert pqc.dsa_verify(pqc.unb64(key.public_key), canonical, bytes.fromhex(row.signature[len("ml-dsa:"):]))
    assert not pqc.dsa_verify(pqc.unb64(key.public_key), canonical + b" ",
                              bytes.fromhex(row.signature[len("ml-dsa:"):]))
    # The job result and the verdict are as before.
    assert "manifest" not in done["result"]


def test_the_manifest_records_the_dataset_familys_holdout_start_when_the_verdict_was_reached(db, market):
    user = make_user(db)
    _, view = evaluated(db, user)
    dataset_holdouts.set_start(db, "EQUITY_EOD", dt.date(2025, 1, 2), set_by="ops", reason="test")
    dataset_holdouts.set_start(db, "CRYPTO_SPOT_EOD", dt.date(2026, 1, 2), set_by="ops", reason="other family")
    first = validate(db, user, view["job_id"])
    dataset_holdouts.set_start(db, "EQUITY_EOD", dt.date(2025, 6, 2), set_by="ops", reason="rolled forward")
    second = validate(db, user, view["job_id"])
    assert first["status"] == second["status"] == "succeeded", (first, second)
    starts = [json.loads(bytes(manifest_row(db, done["job_id"]).canonical))["dataset_holdout_start"]
              for done in (first, second)]
    assert starts == ["2025-01-02", "2025-06-02"]


def test_each_verdict_has_its_own_manifest(db, market):
    user = make_user(db)
    _, view = evaluated(db, user)
    first, second = validate(db, user, view["job_id"]), validate(db, user, view["job_id"])
    a, b = manifest_row(db, first["job_id"]), manifest_row(db, second["job_id"])
    assert a.manifest_id != b.manifest_id
    assert json.loads(bytes(a.canonical))["validation_job_id"] == first["job_id"]
    assert json.loads(bytes(b.canonical))["validation_job_id"] == second["job_id"]
    assert len(rows(db, models.ValidationManifest)) == 2


def test_the_manifest_records_the_commit_baked_into_the_image(db, market, monkeypatch):
    monkeypatch.setattr(config, "CODE_COMMIT", "AB" * 20)
    _, _, _, done = validated(db)
    assert json.loads(bytes(manifest_row(db, done["job_id"]).canonical))["code_commit"] == "ab" * 20


# --------------------------------------------------------------------------
# The signing key, registered with the verdict
# --------------------------------------------------------------------------

def _fresh_identity(monkeypatch):
    """A process whose server key is not yet generated or registered."""
    identity = security_service.server_identity
    for name, value in (("dsa_pk", None), ("dsa_sk", None), ("fingerprint", None), ("key_id", None),
                        ("_registered", False)):
        monkeypatch.setattr(identity, name, value)
    return identity


def test_a_key_not_yet_registered_is_registered_with_the_verdict(db, market, monkeypatch):
    user = make_user(db)
    _, view = evaluated(db, user)
    identity = _fresh_identity(monkeypatch)
    done = validate(db, user, view["job_id"])
    assert done["status"] == "succeeded", done
    row = manifest_row(db, done["job_id"])
    key = db.get(models.ServerSigningKey, row.signing_key_id)
    assert key.fingerprint == identity.fingerprint == hashlib.sha256(identity.dsa_pk).hexdigest()
    assert main.get_strategy_validation_manifest(done["job_id"], user=user, db=db)["signature_valid"] is True


def test_a_refused_outcome_keeps_neither_the_manifest_nor_the_key_it_registered(db, market, monkeypatch):
    """The finalizer registers the key in the job's transaction, not by
    committing: if the outcome is refused after the manifest is written, the
    verdict, the manifest and the key's row all go, and the process does not
    believe its key is registered."""
    user = make_user(db)
    _, view = evaluated(db, user)
    identity = _fresh_identity(monkeypatch)
    record = validation_manifest.record

    def refused_after_recording(*args, **kwargs):
        record(*args, **kwargs)
        raise TaskError(500, "refused after the manifest was written")

    monkeypatch.setattr(validation_manifest, "record", refused_after_recording)
    done = validate(db, user, view["job_id"])
    assert done["status"] == "failed" and done["error"]["detail"] == "refused after the manifest was written"
    assert identity.fingerprint is not None and identity._registered is False
    assert rows(db, models.StrategyValidation) == [] and rows(db, models.ValidationManifest) == []
    assert db.execute(select(models.ServerSigningKey).where(
        models.ServerSigningKey.fingerprint == identity.fingerprint)).scalar_one_or_none() is None

    monkeypatch.setattr(validation_manifest, "record", record)
    again = validate(db, user, view["job_id"])
    assert again["status"] == "succeeded", again
    found = main.get_strategy_validation_manifest(again["job_id"], user=user, db=db)
    assert found["signature_valid"] is True and found["matches_verdict"] is True
    assert db.get(models.ServerSigningKey, found["manifest"]["signing_key_id"]).fingerprint == identity.fingerprint


def test_outside_production_a_manifest_without_ml_dsa_is_signed_by_hmac(db, market, monkeypatch):
    def unavailable(session):
        raise RuntimeError("no ML-DSA here")

    monkeypatch.setattr(validation_manifest, "_signing_key_id", unavailable)
    user, _, _, done = validated(db)
    assert done["status"] == "succeeded", done
    row = manifest_row(db, done["job_id"])
    assert row.signing_key_id is None and row.signature.startswith("hmac-sha256:")
    assert json.loads(bytes(row.canonical))["signing_key_id"] is None
    found = main.get_strategy_validation_manifest(done["job_id"], user=user, db=db)
    assert found["signature_valid"] is True and found["matches_verdict"] is True
    # Production never accepts the HMAC.
    monkeypatch.setattr(validation_manifest, "_production", lambda: True)
    assert main.get_strategy_validation_manifest(done["job_id"], user=user, db=db)["signature_valid"] is False


def test_a_registration_the_database_refuses_does_not_abort_the_verdict(db, market, monkeypatch):
    """The key is registered in a savepoint: a statement the database refuses
    there rolls back to it, so outside production the verdict is still
    recorded, with an HMAC-signed manifest."""
    def refused(session):
        session.execute(text("SELECT 1 / 0"))

    monkeypatch.setattr(validation_manifest, "_signing_key_id", refused)
    user, _, _, done = validated(db)
    assert done["status"] == "succeeded", done
    assert len(rows(db, models.StrategyValidation)) == 1
    row = manifest_row(db, done["job_id"])
    assert row.signing_key_id is None and row.signature.startswith("hmac-sha256:")
    assert main.get_strategy_validation_manifest(done["job_id"], user=user, db=db)["signature_valid"] is True


def test_in_production_a_manifest_that_cannot_be_signed_by_ml_dsa_fails_the_job(db, market, monkeypatch):
    def unavailable(session):
        raise RuntimeError("no ML-DSA here")

    monkeypatch.setattr(validation_manifest, "_signing_key_id", unavailable)
    monkeypatch.setattr(validation_manifest, "_production", lambda: True)
    _, _, _, done = validated(db)
    assert done["status"] == "failed"
    assert done["error"] == {"status_code": 500, "detail": "The validation's manifest could not be signed"}
    assert rows(db, models.StrategyValidation) == [] and rows(db, models.ValidationManifest) == []
    assert rows(db, models.TrialMatrix) == []


@pytest.mark.parametrize("environment, production", [("production", True), ("development", False),
                                                     ("test", False)])
def test_production_is_the_configured_environment(monkeypatch, environment, production):
    monkeypatch.setattr(config, "ENVIRONMENT", environment)
    assert validation_manifest._production() is production


# --------------------------------------------------------------------------
# Reading a manifest
# --------------------------------------------------------------------------

def test_the_manifest_can_be_read_by_the_verdicts_owner_only(db, market):
    owner, _, _, done = validated(db)
    other = make_user(db, "other@example.com")
    found = main.get_strategy_validation_manifest(done["job_id"], user=owner, db=db)
    row = manifest_row(db, done["job_id"])
    assert found == {"job_id": done["job_id"], "manifest_id": row.manifest_id,
                     "manifest": json.loads(bytes(row.canonical)), "canonical": bytes(row.canonical).decode(),
                     "signature": row.signature, "signature_valid": True, "matches_verdict": True, "reason": None}
    for user, job_id in ((other, done["job_id"]), (owner, "no-such-job")):
        with pytest.raises(HTTPException) as caught:
            main.get_strategy_validation_manifest(job_id, user=user, db=db)
        assert caught.value.status_code == 404 and caught.value.detail == "Validation not found"


def test_a_verdict_recorded_before_manifests_existed_reports_none(db, market):
    owner, _, _, done = validated(db)
    [row] = rows(db, models.StrategyValidation)
    values = {c.name: getattr(row, c.name) for c in models.StrategyValidation.__table__.columns}
    db.add(models.StrategyValidation(**{**values, "job_id": "an-r3-4a-job"}))
    db.commit()
    found = main.get_strategy_validation_manifest("an-r3-4a-job", user=owner, db=db)
    assert found["manifest"] is None and found["signature_valid"] is None and found["matches_verdict"] is None
    assert "before validation manifests existed" in found["reason"]
    assert main.get_strategy_validation("an-r3-4a-job", user=owner, db=db)["job_id"] == "an-r3-4a-job"


def _transient(row, **changes):
    values = {c.name: getattr(row, c.name) for c in models.ValidationManifest.__table__.columns}
    return models.ValidationManifest(**{**values, **changes})


def test_a_manifest_whose_bytes_or_signature_changed_does_not_verify(db, market):
    _, _, _, done = validated(db)
    row = manifest_row(db, done["job_id"])
    assert validation_manifest.signature_valid(db, row) is True
    canonical = bytes(row.canonical)
    other_key = pqc.dsa_keygen()
    forged = "ml-dsa:" + pqc.dsa_sign(other_key[1], canonical)[0].hex()
    right_hmac = "hmac-sha256:" + hmac.new(validation_manifest._hmac_key(), canonical, hashlib.sha256).hexdigest()
    for changes in ({"canonical": canonical.replace(b'"passed":false', b'"passed":true')},
                    {"signature": forged},
                    {"signature": "ml-dsa:zz"},
                    {"signature": "rsa:" + row.signature[len("ml-dsa:"):]},
                    {"signature": "ed-dsa:" + row.signature[len("ml-dsa:"):]},
                    {"signing_key_id": None},
                    {"signing_key_id": "no-such-key"},
                    {"signature": "hmac-sha256:" + "0" * 64},
                    {"signature": "hmac-sha256:" + "0" * 64, "signing_key_id": None},
                    {"signature": right_hmac}):  # an HMAC never names a key
        assert validation_manifest.signature_valid(db, _transient(row, **changes)) is False, changes
    assert validation_manifest.signature_valid(db, _transient(row, signature=right_hmac,
                                                              signing_key_id=None)) is True


def test_a_manifest_that_does_not_name_its_verdict_does_not_match_it(db, market):
    _, _, _, done = validated(db)
    row = manifest_row(db, done["job_id"])
    [verdict_row] = rows(db, models.StrategyValidation)
    manifest = json.loads(bytes(row.canonical))
    assert validation_manifest.matches(row, verdict_row, manifest) is True
    values = {c.name: getattr(verdict_row, c.name) for c in models.StrategyValidation.__table__.columns}

    def other_verdict(**changes):
        return models.StrategyValidation(**{**values, **changes})

    altered = json.loads(json.dumps(verdict_row.verdict))
    altered["gates"]["overfitting"]["reason"] = "changed"
    for validation in (other_verdict(verdict=altered), other_verdict(passed=True),
                       other_verdict(job_id="another-job"), other_verdict(evaluation_job_id="another-evaluation"),
                       other_verdict(strategy_version_id="qs-sv-sha256-v1:" + "0" * 64),
                       other_verdict(family_hash="0" * 64), other_verdict(policy_id="qs-vp-sha256-v1:" + "0" * 64),
                       other_verdict(matrix_id="qs-tm-sha256-v1:" + "0" * 64),
                       other_verdict(verdict={**verdict_row.verdict, "nan": math.nan})):
        assert validation_manifest.matches(row, validation, manifest) is False
    # The endpoint's view says so.
    found = validation_manifest.view(db, other_verdict(verdict=altered))
    assert found["signature_valid"] is True and found["matches_verdict"] is False
    for change in ({"manifest_version": 2}, {"signing_key_id": "another-key"},
                   {"verdict": {**manifest["verdict"], "scheme": "another"}}, {"evaluation": None}):
        assert validation_manifest.matches(row, verdict_row, {**manifest, **change}) is False, change
    assert validation_manifest.matches(_transient(row, manifest_id="qs-vm-sha256-v1:" + "0" * 64),
                                       verdict_row, manifest) is False


# --------------------------------------------------------------------------
# What the manifest is made of
# --------------------------------------------------------------------------

def test_the_verdict_hash_is_its_sorted_json_and_survives_a_round_trip():
    verdict = {"b": [1, 0.1, None, True], "a": {"é": 2.5e-17}, "passed": False}
    text_ = '{"a":{"é":2.5e-17},"b":[1,0.1,null,true],"passed":false}'
    assert validation_manifest.verdict_sha256(verdict) == hashlib.sha256(text_.encode("utf-8")).hexdigest()
    assert validation_manifest.verdict_sha256(json.loads(json.dumps(verdict))) == \
        validation_manifest.verdict_sha256(verdict)
    with pytest.raises(ValueError):
        validation_manifest.verdict_sha256({"x": math.inf})
    with pytest.raises(validation_manifest.ManifestError):
        validation_manifest.verdict_sha256({"x": (1, 2)})  # a tuple comes back a list


def test_the_manifest_sorts_its_datasets_and_records_its_time_in_utc():
    evaluation = SimpleNamespace(job_id="e", strategy_version_id="sv", family_hash="f", returns_sha256="r",
                                 dataset_ids=["qs-ds-sha256-v1:b", "qs-ds-sha256-v1:a"])
    created_at = dt.datetime(2026, 11, 19, 17, 30, tzinfo=dt.timezone(dt.timedelta(hours=5, minutes=30)))
    manifest = validation_manifest.build(job_id="j", created_at=created_at, evaluation=evaluation,
                                         dataset_family="EQUITY_EOD", holdout_start=dt.date(2026, 1, 2),
                                         policy_id="p", matrix_id="m", verdict={"passed": True},
                                         signing_key_id=None)
    assert manifest["evaluation"]["dataset_ids"] == ["qs-ds-sha256-v1:a", "qs-ds-sha256-v1:b"]
    assert manifest["created_at"] == "2026-11-19T12:00:00+00:00"
    assert manifest["dataset_holdout_start"] == "2026-01-02"


@pytest.mark.parametrize("value, expected", [
    ("0123456789abcdef0123456789abcdef01234567", "0123456789abcdef0123456789abcdef01234567"),
    ("  0123456789ABCDEF0123456789ABCDEF01234567\n", "0123456789abcdef0123456789abcdef01234567"),
    ("ab" * 32, "ab" * 32),
    ("unknown", None), ("abc1234", None), ("ab" * 21, None), ("g" * 40, None),
])
def test_the_baked_in_commit_must_be_a_full_hash(monkeypatch, value, expected):
    monkeypatch.setattr(config, "CODE_COMMIT", value)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("git must not run when the commit is baked in"))
    assert validation_manifest.code_commit() == expected


@pytest.mark.parametrize("value", [None, "", "   "])
def test_without_a_baked_in_commit_the_checkouts_head_is_used(monkeypatch, value):
    monkeypatch.setattr(config, "CODE_COMMIT", value)
    assert validation_manifest.code_commit() == git_head()


@pytest.mark.parametrize("outcome", ["missing", "fails", "garbage", "timeout"])
def test_without_a_commit_or_git_the_manifest_records_none(monkeypatch, outcome):
    monkeypatch.setattr(config, "CODE_COMMIT", None)

    def run(*args, **kwargs):
        if outcome == "missing":
            raise FileNotFoundError("git")
        if outcome == "timeout":
            raise subprocess.TimeoutExpired("git", 5)
        return subprocess.CompletedProcess(args, 128 if outcome == "fails" else 0,
                                           stdout="ab" * 20 if outcome == "fails" else "fatal: not a repo")

    monkeypatch.setattr(subprocess, "run", run)
    assert validation_manifest.code_commit() is None


def test_the_commit_is_read_once_per_process(monkeypatch):
    calls = []
    monkeypatch.setattr(config, "CODE_COMMIT", None)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append(1) or
                        subprocess.CompletedProcess(a, 0, stdout="cd" * 20))
    assert validation_manifest.code_commit() == validation_manifest.code_commit() == "cd" * 20
    assert len(calls) == 1


def test_an_unreadable_dependency_lock_is_recorded_as_none(monkeypatch):
    from backend.services import experiment_registry
    monkeypatch.setattr(experiment_registry, "dependency_lock_hash", lambda: "unknown")
    assert validation_manifest.dependency_lock_sha256() is None


def test_the_event_engines_version_is_pinned():
    """Bump it, and this test, whenever what a strategy run computes changes."""
    assert event_simulator.ENGINE_VERSION == "qs-event-engine-1"


# --------------------------------------------------------------------------
# The table (migration 0009)
# --------------------------------------------------------------------------

@pytest.fixture
def stored(db, market):
    _, _, _, done = validated(db)
    assert done["status"] == "succeeded", done
    return manifest_row(db, done["job_id"])


def _insert(db, row, **changes):
    values = {c.name: getattr(row, c.name) for c in models.ValidationManifest.__table__.columns}
    db.add(models.ValidationManifest(**{**values, **changes}))
    db.flush()


def _another_verdict(db, row):
    verdict = db.get(models.StrategyValidation, row.job_id)
    values = {c.name: getattr(verdict, c.name) for c in models.StrategyValidation.__table__.columns}
    db.add(models.StrategyValidation(**{**values, "job_id": "another-job"}))
    db.flush()


def _canonical(row, **changes):
    canonical = dataset_format.canonical_json({**json.loads(bytes(row.canonical)), **changes})
    return {"canonical": canonical, "manifest_id": validation_manifest.manifest_id(canonical)}


@pytest.mark.parametrize("change, constraint", [
    ("id", "ck_validation_manifests_id_is_hash"),
    ("job", "ck_validation_manifests_names_its_row"),
    ("key_in_manifest", "ck_validation_manifests_names_its_row"),
    ("version", "ck_validation_manifests_names_its_row"),
    ("not_an_object", "ck_validation_manifests_names_its_row"),
    ("signature_form", "ck_validation_manifests_signature_form"),
    ("odd_hex", "ck_validation_manifests_signature_form"),
    ("ml_dsa_without_key", "ck_validation_manifests_signature_form"),
    ("hmac_with_key", "ck_validation_manifests_signature_form"),
    ("unknown_key", "fkey"),
    ("no_such_verdict", "fkey"),
    ("second_manifest", "validation_manifests_job_id_key"),
])
def test_the_schema_refuses_a_manifest_that_does_not_match_its_row(db, stored, change, constraint):
    _another_verdict(db, stored)
    job = {"job_id": "another-job", **_canonical(stored, validation_job_id="another-job")}
    changes = {
        "id": {**job, "manifest_id": "qs-vm-sha256-v1:" + "0" * 64},
        "job": {**_canonical(stored), "job_id": "another-job"},
        "key_in_manifest": {"job_id": "another-job",
                            **_canonical(stored, validation_job_id="another-job", signing_key_id="another")},
        "version": {"job_id": "another-job", **_canonical(stored, validation_job_id="another-job",
                                                          manifest_version=2)},
        "not_an_object": {"job_id": "another-job", "canonical": b"[]",
                          "manifest_id": validation_manifest.manifest_id(b"[]")},
        "signature_form": {**job, "signature": "rsa:00"},
        "odd_hex": {**job, "signature": "ml-dsa:abc"},
        "ml_dsa_without_key": {"job_id": "another-job", "signing_key_id": None,
                               **_canonical(stored, validation_job_id="another-job", signing_key_id=None)},
        "hmac_with_key": {**job, "signature": "hmac-sha256:" + "0" * 64},
        "unknown_key": {"job_id": "another-job", "signing_key_id": "no-such-key",
                        **_canonical(stored, validation_job_id="another-job", signing_key_id="no-such-key")},
        "no_such_verdict": {"job_id": "no-such-job", **_canonical(stored, validation_job_id="no-such-job")},
        "second_manifest": _canonical(stored, created_at="2020-01-01T00:00:00+00:00"),
    }[change]
    with pytest.raises(DBAPIError, match=constraint):
        _insert(db, stored, **changes)
    db.rollback()
    _another_verdict(db, stored)
    _insert(db, stored, **job)  # the same manifest for another verdict is accepted
    db.rollback()


@pytest.mark.parametrize("statement", [
    "UPDATE validation_manifests SET signature = signature", "DELETE FROM validation_manifests",
    "TRUNCATE validation_manifests",
])
def test_manifests_are_immutable_even_for_the_owner(db, stored, statement):
    with pytest.raises(DBAPIError, match="its rows are immutable"):
        db.execute(text(statement))
    db.rollback()
    assert db.execute(text("SELECT count(*) FROM validation_manifests")).scalar() == 1


def test_a_verdict_with_a_manifest_cannot_be_truncated_away(db, stored):
    with pytest.raises(DBAPIError):
        db.execute(text("TRUNCATE strategy_validations"))
    db.rollback()
    with pytest.raises(DBAPIError, match="its rows are immutable"):
        db.execute(text("TRUNCATE strategy_validations CASCADE"))
    db.rollback()
    assert strategy_validation.get_for_user(db, db.get(models.StrategyValidation, stored.job_id).user_id,
                                            stored.job_id) is not None
