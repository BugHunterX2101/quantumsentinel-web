"""The dataset registry on PostgreSQL: rows derived from verified blobs, tied
to their id by CHECKs, immutable by triggers, listed only to their holders."""
import dataclasses
import datetime as dt

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.orm import Session

from backend import models
from backend.services import dataset_registry
from backend.services.dataset_format import (BARS_V1, DatasetIdentity, DatasetIntegrityError, Dataset, Table,
                                             canonical_json)
from backend.services.dataset_registry import RegistryError
from backend.services.dataset_store import DatasetStore
from dataset_helpers import NOW, snapshot

REQUEST = {"symbols": ["AAPL"], "start": "2024-01-01", "end": "2024-01-11"}


@pytest.fixture
def db(make_engine):
    engine = make_engine(migrated=True)
    with engine.begin() as conn:
        for uid in ("u1", "u2"):
            conn.execute(text("INSERT INTO users (id, email, password_hash) VALUES (:id, :email, 'h')"),
                         {"id": uid, "email": f"{uid}@example.com"})
    with Session(engine) as session:
        yield session


@pytest.fixture
def stored(tmp_path):
    """A provider snapshot's equity dataset, stored and read back verified."""
    store = DatasetStore(tmp_path)
    dataset = snapshot(("AAPL", "VOD.L")).datasets[0]
    blob = store.put(dataset)
    return store.get(blob.dataset_id, blob.sha256), blob


def register(db, stored, user_id="u1", **kwargs):
    dataset, blob = stored
    values = dict(user_id=user_id, request=REQUEST, provider_client="yfinance test", retrieved_at=NOW,
                  job_id="job-1")
    values.update(kwargs)
    version = dataset_registry.register(db, dataset, blob, **values)
    db.commit()
    return version


def test_a_verified_dataset_is_registered_with_everything_derived_from_it(db, stored):
    dataset, blob = stored
    version = register(db, stored)
    assert version.dataset_id == dataset.dataset_id == blob.dataset_id
    assert {name: getattr(version, name) for name in dataset_registry.IDENTITY_FIELDS} \
        == dataset.identity.as_dict()
    assert (version.personal_use_only, version.redistribution_allowed, version.point_in_time) \
        == (True, False, False)
    assert (version.coverage_start, version.coverage_end) == (dt.date(2024, 1, 2), dt.date(2024, 1, 9))
    assert version.symbols == ["AAPL", "VOD.L"]
    assert bytes(version.manifest_canonical) == canonical_json(dataset.manifest())
    assert version.tables_json["bars"] == dataset.table("bars").manifest()
    assert (version.blob_format, version.blob_sha256, version.blob_size) \
        == ("qs-blob-v1", blob.sha256, blob.size)
    assert version.retrieved_at == NOW and version.request_json == REQUEST
    holder = db.get(models.UserDataset, ("u1", dataset.dataset_id))
    assert holder.job_id == "job-1"


def test_registering_again_keeps_the_first_row_and_adds_the_holder(db, stored):
    register(db, stored)
    later = NOW + dt.timedelta(days=3)
    register(db, stored, user_id="u2", retrieved_at=later, job_id="job-2")
    register(db, stored, user_id="u2", retrieved_at=later, job_id="job-3")
    db.expire_all()
    assert db.execute(text("SELECT count(*) FROM dataset_versions")).scalar() == 1
    assert db.get(models.DatasetVersion, stored[0].dataset_id).retrieved_at == NOW
    holders = db.execute(text("SELECT user_id, job_id FROM user_datasets ORDER BY user_id")).all()
    assert [tuple(h) for h in holders] == [("u1", "job-1"), ("u2", "job-2")]


def test_a_different_blob_under_a_registered_id_is_refused(db, stored):
    register(db, stored)
    dataset, blob = stored
    other = dataclasses.replace(blob, sha256="0" * 64)
    with pytest.raises(DatasetIntegrityError, match="different blob"):
        dataset_registry.register(db, dataset, other, user_id="u2", request=REQUEST,
                                  provider_client="x", retrieved_at=NOW)
    db.rollback()
    assert db.execute(text("SELECT count(*) FROM user_datasets")).scalar() == 1


def test_datasets_are_listed_only_to_their_holders(db, stored):
    register(db, stored)
    dataset_id = stored[0].dataset_id
    assert [v.dataset_id for v in dataset_registry.list_for_user(db, "u1")] == [dataset_id]
    assert dataset_registry.list_for_user(db, "u2") == []
    assert dataset_registry.get_for_user(db, "u1", dataset_id).dataset_id == dataset_id
    assert dataset_registry.get_for_user(db, "u2", dataset_id) is None


def test_the_view_reports_identity_licence_coverage_and_blob(db, stored):
    view = dataset_registry.view(register(db, stored))
    assert view["identity"]["adjustment"] == "split" and view["identity"]["provider"] == "yahoo"
    assert view["licence"] == {"personal_use_only": True, "redistribution_allowed": False}
    assert view["coverage"] == {"start": "2024-01-02", "end": "2024-01-09"}
    assert view["tables"] == {"bars": {"rows": 11}, "corporate_actions": {"rows": 2},
                              "instruments": {"rows": 2}}
    assert view["blob"]["sha256"] == stored[1].sha256
    assert view["retrieved_at"] == "2024-01-10T15:00:00+00:00"


# ── the database's own guarantees ─────────────────────────────────────────────

def _row(db, stored, **changes):
    dataset, blob = stored
    row = dataset_registry.describe(dataset, blob, REQUEST, "yfinance test", NOW)
    row.update(changes, created_at=NOW)
    db.execute(models.DatasetVersion.__table__.insert().values(**row))
    db.commit()


def _refused(db, stored, **changes):
    with pytest.raises(IntegrityError) as caught:
        _row(db, stored, **changes)
    db.rollback()
    return caught.value.orig.diag.constraint_name


def test_an_id_that_is_not_the_manifest_hash_is_refused(db, stored):
    assert _refused(db, stored, dataset_id="qs-ds-sha256-v1:" + "0" * 64) \
        == "ck_dataset_versions_id_is_manifest_hash"


def test_identity_columns_that_disagree_with_the_manifest_are_refused(db, stored):
    assert _refused(db, stored, adjustment="raw") == "ck_dataset_versions_identity_is_manifest"
    assert _refused(db, stored, licence_id="databento") == "ck_dataset_versions_identity_is_manifest"
    # A manifest rewritten to match the forged columns no longer hashes to the id.
    forged = stored[0].manifest()
    forged["identity"]["adjustment"] = "raw"
    assert _refused(db, stored, adjustment="raw", manifest_canonical=canonical_json(forged)) \
        == "ck_dataset_versions_id_is_manifest_hash"


@pytest.mark.parametrize("changes, constraint", [
    ({"blob_sha256": "XYZ"}, "ck_dataset_versions_blob_sha256"),
    ({"blob_size": 0}, "ck_dataset_versions_blob_size"),
    ({"coverage_start": dt.date(2025, 1, 1)}, "ck_dataset_versions_coverage"),
])
def test_blob_and_coverage_checks(db, stored, changes, constraint):
    assert _refused(db, stored, **changes) == constraint


@pytest.mark.parametrize("statement", [
    "UPDATE dataset_versions SET blob_size = blob_size + 1",
    "UPDATE dataset_versions SET request_json = '{}'",
    "DELETE FROM dataset_versions",
    "TRUNCATE dataset_versions CASCADE",
])
def test_registered_rows_never_change_even_for_the_owner(db, stored, statement):
    register(db, stored)
    with pytest.raises(DBAPIError, match="immutable") as caught:
        db.execute(text(statement))
    assert caught.value.orig.sqlstate == "23001"  # restrict_violation
    db.rollback()
    assert db.execute(text("SELECT count(*) FROM dataset_versions")).scalar() == 1


# ── what describe() refuses ───────────────────────────────────────────────────

def test_describe_refuses_what_it_cannot_vouch_for(stored):
    dataset, blob = stored
    with pytest.raises(RegistryError, match="not this dataset"):
        dataset_registry.describe(dataset, dataclasses.replace(blob, dataset_id="qs-ds-sha256-v1:" + "1" * 64),
                                  REQUEST, "x", NOW)
    with pytest.raises(RegistryError, match="timezone-aware"):
        dataset_registry.describe(dataset, blob, REQUEST, "x", NOW.replace(tzinfo=None))
    identity = dataclasses.replace(dataset.identity, licence_id="unknown-terms")
    unknown = Dataset(identity, dataset.tables)
    with pytest.raises(RegistryError, match="unknown licence"):
        dataset_registry.describe(unknown, dataclasses.replace(blob, dataset_id=unknown.dataset_id),
                                  REQUEST, "x", NOW)
    no_bars = Dataset(dataset.identity, (dataset.table("instruments"),))
    with pytest.raises(RegistryError, match="bars table"):
        dataset_registry.describe(no_bars, dataclasses.replace(blob, dataset_id=no_bars.dataset_id),
                                  REQUEST, "x", NOW)
    empty = Dataset(dataset.identity, (Table.build(BARS_V1, {c.name: [] for c in BARS_V1.columns}),))
    with pytest.raises(RegistryError, match="at least one bar"):
        dataset_registry.describe(empty, dataclasses.replace(blob, dataset_id=empty.dataset_id),
                                  REQUEST, "x", NOW)


def test_point_in_time_needs_raw_prices_and_a_point_in_time_universe(stored):
    dataset, blob = stored
    for adjustment, survivorship, expected in (("raw", "point_in_time", True), ("split", "point_in_time", False),
                                               ("raw", "current_listings", False)):
        identity = DatasetIdentity(**{**dataset.identity.as_dict(), "adjustment": adjustment,
                                      "survivorship": survivorship})
        other = Dataset(identity, dataset.tables)
        row = dataset_registry.describe(other, dataclasses.replace(blob, dataset_id=other.dataset_id),
                                        REQUEST, "x", NOW)
        assert row["point_in_time"] is expected
