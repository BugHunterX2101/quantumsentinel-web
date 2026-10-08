"""Research governance: manifest v2, historical-key verification, true replay,
validation gates and four-eyes approval."""
import hashlib
import inspect

import pytest
from fastapi import HTTPException
from sqlalchemy.orm import sessionmaker

from backend import main, models
from backend.services import experiment_registry as er
from backend.services import security_service
from backend.services.l2_event_replay import obi_momentum_strategy
from research_job_helpers import result_of


@pytest.fixture
def db(make_engine):
    session = sessionmaker(bind=make_engine())()
    yield session
    session.close()


@pytest.fixture
def owner(db):
    user = models.User(email="researcher@example.com", password_hash="x")
    db.add(user)
    db.commit()
    return user


@pytest.fixture
def operator(db):
    user = models.User(email="risk@example.com", password_hash="x", role="risk_admin")
    db.add(user)
    db.commit()
    return user


@pytest.fixture
def restore_identity():
    identity = security_service.server_identity
    saved = {k: getattr(identity, k) for k in ("dsa_pk", "dsa_sk", "key_id", "fingerprint",
                                                "created_at", "_registered", "keygen_ms")}
    yield identity
    for k, v in saved.items():
        setattr(identity, k, v)


BARS = [{"timestamp": 1_700_000_000 + i * 86400, "open": 100 + 0.5 * i, "high": 102 + 0.5 * i,
         "low": 99 + 0.5 * i, "close": 100.5 + 0.5 * i, "volume": 200_000} for i in range(6)]
PARAMS = {"obi_threshold": 0.2, "events_per_bar": 120, "warmup_events": 30, "latency_preset": "cloud"}


def create(db, user, strategy_id="obi_momentum", dataset=BARS, parameters=PARAMS):
    return main.experiment_create({"strategy_id": strategy_id, "strategy_version": "1.0",
                                   "dataset_id": "synthetic-bars", "dataset": dataset,
                                   "parameters": parameters, "random_seed": 11}, user, db)


class TestManifest:
    def test_signed_manifest_still_verifies_after_reloading_from_the_database(self, db, owner):
        exp = create(db, owner, strategy_id="external-model", parameters={"a": 1})
        er.PersistentExperimentRegistry(db, owner.id).complete(exp["experiment_id"], {"sharpe": 1.2})
        db.expire_all()
        manifest = main.experiment_manifest(exp["experiment_id"], owner, db)
        assert manifest["signature_valid"] is True
        assert manifest["consistent_with_record"] is True

    def test_manifest_v2_binds_code_dependencies_engine_and_key(self, db, owner):
        exp = create(db, owner)
        result_of(db, main.experiment_run(exp["experiment_id"], owner, db))
        manifest = main.experiment_manifest(exp["experiment_id"], owner, db)
        source_hash = hashlib.sha256(b"source:" + inspect.getsource(obi_momentum_strategy).encode()).hexdigest()
        assert manifest["manifest_version"] == 2
        assert manifest["strategy_hash"] == source_hash
        assert len(manifest["dependency_lock_hash"]) == 64
        assert manifest["execution_engine_version"] == er.ENGINE_VERSION
        assert manifest["signing_key_id"] == security_service.server_identity.key_id

    def test_manifest_verifies_with_its_historical_key_after_rotation(self, db, owner, restore_identity):
        exp = create(db, owner, strategy_id="external-model", parameters={})
        er.PersistentExperimentRegistry(db, owner.id).complete(exp["experiment_id"], {"sharpe": 1.0})
        signed_with = security_service.server_identity.key_id
        restore_identity.rotate(db)
        assert security_service.server_identity.key_id != signed_with
        assert main.experiment_manifest(exp["experiment_id"], owner, db)["signature_valid"] is True

    def test_tampering_with_the_stored_record_is_detected(self, db, owner):
        exp = create(db, owner, strategy_id="external-model", parameters={})
        er.PersistentExperimentRegistry(db, owner.id).complete(exp["experiment_id"], {"sharpe": 1.0})
        row = db.get(models.ResearchExperiment, exp["experiment_id"])
        row.result_hash = "0" * 64
        db.commit()
        assert main.experiment_manifest(exp["experiment_id"], owner, db)["consistent_with_record"] is False

    def test_production_never_downgrades_to_hmac(self, monkeypatch):
        monkeypatch.setattr(er, "_environment", lambda: "production")
        monkeypatch.setattr(security_service.server_identity, "sign",
                            lambda _m: (_ for _ in ()).throw(RuntimeError("hsm down")))
        with pytest.raises(er.ManifestSigningError):
            er.sign_manifest({"experiment_id": "X"})
        monkeypatch.setattr(er, "_environment", lambda: "development")
        signature = er.sign_manifest({"experiment_id": "X"})
        assert signature.startswith("hmac-sha256:")
        monkeypatch.setattr(er, "_environment", lambda: "production")
        assert er.verify_manifest_signature({"experiment_id": "X"}, signature) is False


class TestTrueReplay:
    def test_run_records_results_once_and_replay_re_executes(self, db, owner):
        exp = create(db, owner)
        ran = result_of(db, main.experiment_run(exp["experiment_id"], owner, db))
        assert ran["status"] == "COMPLETED" and ran["results"]["data_source"] == "synthetic"
        with pytest.raises(HTTPException) as exc:
            main.experiment_run(exp["experiment_id"], owner, db)
        assert exc.value.status_code == 422
        replay = result_of(db, main.experiment_replay(exp["experiment_id"], {}, owner, db))
        assert replay["replay_status"] == "re-executed"
        assert replay["result_reproduced"] is True
        assert replay["result_hash"] == ran["result_hash"]

    def test_changed_inputs_are_not_replayed(self, db, owner):
        exp = create(db, owner)
        result_of(db, main.experiment_run(exp["experiment_id"], owner, db))
        replay = result_of(db, main.experiment_replay(exp["experiment_id"], {"random_seed": 12}, owner, db))
        assert replay["matches_experiment"] is False
        assert replay["replay_status"] == "deterministic_input_verification"

    def test_non_executable_strategies_cannot_be_run(self, db, owner):
        exp = create(db, owner, strategy_id="external-model", parameters={})
        with pytest.raises(HTTPException) as exc:
            main.experiment_run(exp["experiment_id"], owner, db)
        assert exc.value.status_code == 422

    def test_invalid_strategy_parameters_are_rejected(self, db, owner):
        exp = create(db, owner, parameters={"obi_threshold": 0.2, "leverage": 10})
        with pytest.raises(HTTPException) as exc:
            main.experiment_run(exp["experiment_id"], owner, db)
        assert exc.value.status_code == 422


class TestValidationAndApproval:
    def test_integrity_gates_are_evaluated_for_real(self, db, owner):
        exp = create(db, owner)
        result_of(db, main.experiment_run(exp["experiment_id"], owner, db))
        report = result_of(db, main.experiment_validate(exp["experiment_id"], owner, db))
        for gate in ("signature_valid", "manifest_consistent", "inputs_unchanged", "result_reproducible"):
            assert report["gates"][gate]["passed"] is True, gate
        assert {"net_pnl", "fill_ratio", "implementation_shortfall"} <= set(report["gates"])

    def test_tampered_inputs_fail_validation(self, db, owner):
        exp = create(db, owner)
        result_of(db, main.experiment_run(exp["experiment_id"], owner, db))
        row = db.get(models.ResearchExperiment, exp["experiment_id"])
        row.dataset_json = BARS[:-1]
        db.commit()
        report = result_of(db, main.experiment_validate(exp["experiment_id"], owner, db))
        assert report["gates"]["inputs_unchanged"]["passed"] is False
        assert report["all_gates_passed"] is False

    def test_approval_requires_validation_an_operator_and_a_second_person(self, db, owner, operator, monkeypatch):
        # Result-quality gates depend on market outcomes; hold them passing so the
        # governance workflow is tested. Integrity gates remain real.
        real = er.validate_for_deployment

        def quality_passes(exp):
            report = real(exp)
            for gate in report["gates"].values():
                gate["passed"] = True
            return report

        exp = create(db, owner)
        exp_id = exp["experiment_id"]
        result_of(db, main.experiment_run(exp_id, owner, db))
        with pytest.raises(HTTPException) as exc:
            main.experiment_approve(exp_id, operator, db)
        assert exc.value.status_code == 409                       # not validated yet

        monkeypatch.setattr(er, "validate_for_deployment", quality_passes)
        assert result_of(db, main.experiment_validate(exp_id, owner, db))["all_gates_passed"] is True

        with pytest.raises(HTTPException) as exc:
            main.experiment_approve(exp_id, owner, db)
        assert exc.value.status_code == 403                       # owner is not an operator

        operator_exp = create(db, operator)
        result_of(db, main.experiment_run(operator_exp["experiment_id"], operator, db))
        result_of(db, main.experiment_validate(operator_exp["experiment_id"], operator, db))
        with pytest.raises(HTTPException) as exc:
            main.experiment_approve(operator_exp["experiment_id"], operator, db)
        assert exc.value.status_code == 409                       # four-eyes: no self-approval

        approved = main.experiment_approve(exp_id, operator, db)
        assert approved["status"] == "APPROVED"
        row = db.get(models.ResearchExperiment, exp_id)
        assert row.approved_by == operator.id and row.approved_at is not None
        assert db.query(models.AuditLog).filter_by(action="EXPERIMENT_APPROVED", resource_id=exp_id).count() == 1
