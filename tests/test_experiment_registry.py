"""Tests for experiment_registry.py.

Covers:
- Experiment creation with provenance tracking
- Dataset/parameter/result hashing (deterministic)
- Manifest building and signing
- Signature verification
- Deployment validation gates
- Experiment lifecycle (create → start → complete → validate → approve)
"""

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend import models
from backend.database import Base
from backend.services.experiment_registry import (
    Experiment,
    ExperimentRegistry,
    PersistentExperimentRegistry,
    ExperimentStatus,
    build_manifest,
    hash_dataset,
    hash_parameters,
    hash_results,
    sign_manifest,
    validate_for_deployment,
    verify_manifest_signature,
)


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

class TestHashing:
    def test_dataset_hash_deterministic(self):
        data = [{"price": 100.0}, {"price": 101.0}]
        h1 = hash_dataset(data)
        h2 = hash_dataset(data)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_different_data_different_hash(self):
        h1 = hash_dataset([1, 2, 3])
        h2 = hash_dataset([1, 2, 4])
        assert h1 != h2

    def test_parameter_hash(self):
        params = {"lookback": 20, "threshold": 0.5}
        h = hash_parameters(params)
        assert len(h) == 64

    def test_result_hash(self):
        results = {"sharpe": 1.5, "max_drawdown": -0.08}
        h = hash_results(results)
        assert len(h) == 64


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

class TestManifest:
    def test_build_manifest(self):
        exp = Experiment(
            strategy_id="OBI-momentum",
            strategy_version="v3",
            dataset_id="BTC-L2-2026",
            dataset_hash="abc123",
            result_hash="def456",
        )
        manifest = build_manifest(exp)
        assert manifest["strategy_id"] == "OBI-momentum"
        assert manifest["dataset_hash"] == "abc123"
        assert manifest["result_hash"] == "def456"

    def test_sign_and_verify(self):
        manifest = {
            "experiment_id": "QS-TEST",
            "dataset_hash": "abc",
            "result_hash": "def",
        }
        signature = sign_manifest(manifest)
        assert signature  # non-empty

        # Verify the signature
        valid = verify_manifest_signature(manifest, signature)
        assert valid is True

    def test_verify_tampered_manifest_fails(self):
        manifest = {
            "experiment_id": "QS-TEST",
            "dataset_hash": "abc",
        }
        signature = sign_manifest(manifest)

        # Tamper with the manifest
        manifest["dataset_hash"] = "TAMPERED"
        valid = verify_manifest_signature(manifest, signature)
        assert valid is False


# ---------------------------------------------------------------------------
# Validation Gates
# ---------------------------------------------------------------------------

class TestValidationGates:
    def test_all_gates_pass(self):
        exp = Experiment(
            strategy_id="test",
            results={
                "oos_sharpe": 1.5,
                "survives_deflation": True,
                "avg_daily_turnover": 0.5,
                "max_position_size": 1000,
                "max_drawdown_limit": -0.1,
            },
            manifest_signature="some-sig",
        )
        result = validate_for_deployment(exp)
        assert result["all_gates_passed"] is True
        assert result["eligible_for_paper_deployment"] is True

    def test_low_sharpe_fails(self):
        exp = Experiment(
            strategy_id="test",
            results={
                "oos_sharpe": 0.2,
                "survives_deflation": True,
                "avg_daily_turnover": 0.5,
                "max_position_size": 1000,
            },
            manifest_signature="sig",
        )
        result = validate_for_deployment(exp)
        assert result["gates"]["walk_forward"]["passed"] is False
        assert result["all_gates_passed"] is False

    def test_missing_signature_fails(self):
        exp = Experiment(
            strategy_id="test",
            results={
                "oos_sharpe": 1.5,
                "survives_deflation": True,
                "avg_daily_turnover": 0.5,
                "max_position_size": 1000,
            },
        )
        result = validate_for_deployment(exp)
        assert result["gates"]["manifest_signed"]["passed"] is False

    def test_high_turnover_fails(self):
        exp = Experiment(
            strategy_id="test",
            results={
                "oos_sharpe": 1.5,
                "survives_deflation": True,
                "avg_daily_turnover": 5.0,
                "max_position_size": 1000,
            },
            manifest_signature="sig",
        )
        result = validate_for_deployment(exp)
        assert result["gates"]["turnover"]["passed"] is False


# ---------------------------------------------------------------------------
# Experiment Registry
# ---------------------------------------------------------------------------

class TestExperimentRegistry:

    def test_persistent_registry_survives_a_new_service_instance(self, db):
        user = models.User(email="research-owner@example.com", password_hash="x")
        db.add(user)
        db.commit()

        created = PersistentExperimentRegistry(db, user.id).create(
            strategy_id="OBI-v3", strategy_version="3.1", dataset_id="synthetic-AAPL",
            dataset=[{"timestamp": 1, "price": 100.0}], parameters={"threshold": 0.3},
            random_seed=7,
        )
        restored = PersistentExperimentRegistry(db, user.id).get(created.experiment_id)

        assert restored is not None
        assert restored.experiment_id == created.experiment_id
        assert restored.dataset_hash == created.dataset_hash
        assert restored.parameter_hash == created.parameter_hash
        assert restored.random_seed == 7

    def test_experiment_endpoints_use_the_durable_user_owned_registry(self, db):
        """Create → retrieve → replay works across independently built registries."""
        from backend import main

        owner = models.User(email="research-endpoint-owner@example.com", password_hash="x")
        other_user = models.User(email="research-endpoint-other@example.com", password_hash="x")
        db.add_all([owner, other_user])
        db.commit()
        request = {
            "strategy_id": "OBI-v3", "strategy_version": "3.1",
            "dataset_id": "synthetic-AAPL",
            "dataset": [{"timestamp": 1, "price": 100.0}],
            "parameters": {"threshold": 0.3}, "random_seed": 7,
        }

        created = main.experiment_create(request, owner, db)
        restored = main.experiment_get(created["experiment_id"], owner, db)
        replay = main.experiment_replay(created["experiment_id"], {}, owner, db)

        assert restored["dataset_hash"] == created["dataset_hash"]
        assert replay["matches_experiment"] is True
        with pytest.raises(HTTPException) as exc:
            main.experiment_get(created["experiment_id"], other_user, db)
        assert exc.value.status_code == 404

    def test_create_experiment(self):
        registry = ExperimentRegistry()
        exp = registry.create(
            strategy_id="OBI-v3",
            strategy_version="3.1",
            dataset_id="BTC-L2",
            dataset=[1, 2, 3],
            parameters={"lookback": 20},
            random_seed=42,
        )
        assert exp.experiment_id.startswith("QS-")
        assert exp.dataset_hash  # non-empty
        assert exp.parameter_hash  # non-empty
        assert exp.status == ExperimentStatus.CREATED

    def test_full_lifecycle(self):
        registry = ExperimentRegistry()
        exp = registry.create(
            strategy_id="test-strategy",
            strategy_version="1.0",
            dataset_id="test-data",
        )

        # Start
        registry.start(exp.experiment_id)
        assert exp.status == ExperimentStatus.RUNNING

        # Complete with results
        results = {
            "oos_sharpe": 1.8,
            "survives_deflation": True,
            "avg_daily_turnover": 0.3,
            "max_position_size": 500,
            "max_drawdown_limit": -0.1,
        }
        registry.complete(exp.experiment_id, results)
        assert exp.status == ExperimentStatus.COMPLETED
        assert exp.result_hash  # non-empty
        assert exp.manifest_signature  # signed

        # Validate
        validation = registry.validate(exp.experiment_id)
        assert validation is not None
        assert validation["all_gates_passed"] is True
        assert exp.status == ExperimentStatus.VALIDATED

        # Approve
        registry.approve(exp.experiment_id)
        assert exp.status == ExperimentStatus.APPROVED

    def test_list_experiments(self):
        registry = ExperimentRegistry()
        registry.create(strategy_id="s1", strategy_version="1", dataset_id="d1")
        registry.create(strategy_id="s2", strategy_version="1", dataset_id="d2")
        assert len(registry.list_all()) == 2

    def test_get_manifest(self):
        registry = ExperimentRegistry()
        exp = registry.create(strategy_id="test", strategy_version="1", dataset_id="d")
        registry.complete(exp.experiment_id, {"sharpe": 1.0})

        manifest = registry.get_manifest(exp.experiment_id)
        assert manifest is not None
        assert "signature" in manifest
        assert manifest["signature_valid"] is True

    def test_deterministic_replay_guarantee(self):
        """Same dataset + params + seed should produce same hashes."""
        data = [{"price": 100.0}, {"price": 101.0}]
        params = {"lookback": 20}

        registry = ExperimentRegistry()
        e1 = registry.create(strategy_id="s", strategy_version="1",
                              dataset_id="d", dataset=data, parameters=params, random_seed=42)
        e2 = registry.create(strategy_id="s", strategy_version="1",
                              dataset_id="d", dataset=data, parameters=params, random_seed=42)

        assert e1.dataset_hash == e2.dataset_hash
        assert e1.parameter_hash == e2.parameter_hash
        assert e1.random_seed == e2.random_seed
