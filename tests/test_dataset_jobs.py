"""Dataset snapshots as research jobs: the API queues, the job process fetches
and stores, the worker re-verifies every blob and registers it."""
import datetime as dt
import json
import os

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import select, text
from sqlalchemy.orm import sessionmaker

from backend import config, main, models, schemas, worker
from backend.services import dataset_providers, dataset_registry, research_jobs, research_tasks
from backend.services.dataset_store import DatasetStore
from dataset_helpers import FakeYahoo, frame
from research_job_helpers import run_queued

START, END = dt.date(2024, 1, 1), dt.date(2024, 1, 11)


@pytest.fixture
def Session(make_engine):
    return sessionmaker(bind=make_engine(migrated=True))


@pytest.fixture
def db(Session):
    session = Session()
    yield session
    session.close()


@pytest.fixture
def yahoo(monkeypatch, tmp_path):
    """Yahoo-shaped data instead of the network, and a dataset root of our own."""
    fake = FakeYahoo()
    monkeypatch.setattr(dataset_providers, "_yahoo_fetch", fake)
    monkeypatch.setattr(dataset_providers, "_yahoo_client", lambda: "yfinance test")
    monkeypatch.setattr(config, "DATASET_ROOT", tmp_path / "datasets")
    return fake


def make_user(db, email="researcher@example.com"):
    user = models.User(email=email, password_hash="x", role="user")
    db.add(user)
    db.commit()
    return user


def request(symbols=("AAPL", "BTC-USD"), start=START, end=END):
    return schemas.DatasetSnapshotRequest(symbols=list(symbols), start=start, end=end)


def snapshot(db, user, **kwargs):
    return run_queued(db, main.snapshot_dataset(request(**kwargs), user=user, db=db))


def test_the_api_queues_a_snapshot_and_fetches_nothing(db, monkeypatch):
    monkeypatch.setattr(research_tasks, "dataset_snapshot",
                        lambda _params: pytest.fail("the API process must not fetch data"))
    user = make_user(db)
    response = main.snapshot_dataset(request(symbols=(" aapl", "BTC-USD")), user=user, db=db)
    payload = json.loads(response.body)
    assert response.status_code == 202 and payload["kind"] == "dataset_snapshot"
    job = db.get(models.ResearchJob, payload["job_id"])
    assert job.params_json == {"symbols": ["AAPL", "BTC-USD"], "start": "2024-01-01", "end": "2024-01-11"}
    # The same request while the first is queued returns the first job.
    again = json.loads(main.snapshot_dataset(request(symbols=("AAPL", "BTC-USD")), user=user, db=db).body)
    assert again["job_id"] == payload["job_id"]


def test_a_snapshot_job_stores_verifies_and_registers_each_family(db, yahoo):
    user = make_user(db)
    view = snapshot(db, user, symbols=("AAPL", "BTC-USD", "NOPE"))
    assert view["status"] == "succeeded", view
    result = view["result"]
    assert result["unavailable"] == {"NOPE": "no data returned"}
    families = {d["identity"]["family"]: d for d in result["datasets"]}
    assert set(families) == {"EQUITY_EOD", "CRYPTO_SPOT_EOD"}
    equity = families["EQUITY_EOD"]
    assert equity["identity"]["adjustment"] == "split" and equity["licence"]["personal_use_only"] is True
    assert equity["provider_client"] == "yfinance test"
    assert equity["request"] == {"symbols": ["AAPL", "BTC-USD", "NOPE"], "start": "2024-01-01",
                                 "end": "2024-01-11"}
    store = DatasetStore(config.DATASET_ROOT)
    for dataset in result["datasets"]:
        stored = store.get(dataset["dataset_id"], dataset["blob"]["sha256"])
        assert stored.dataset_id == dataset["dataset_id"]
    rows = db.execute(text("SELECT dataset_id, user_id, job_id FROM user_datasets ORDER BY dataset_id")).all()
    assert sorted(r.dataset_id for r in rows) == sorted(families[f]["dataset_id"] for f in families)
    assert {(r.user_id, r.job_id) for r in rows} == {(user.id, view["job_id"])}
    audit = db.execute(select(models.AuditLog).where(models.AuditLog.action == "DATASET_SNAPSHOT")).scalar_one()
    assert audit.resource_type == "dataset" and audit.user_id == user.id


def test_the_dataset_endpoints_show_only_the_callers_datasets(db, yahoo):
    owner, other = make_user(db), make_user(db, "other@example.com")
    result = snapshot(db, owner, symbols=("AAPL",))["result"]
    dataset_id = result["datasets"][0]["dataset_id"]
    assert [d["dataset_id"] for d in main.list_datasets(limit=50, user=owner, db=db)] == [dataset_id]
    assert main.get_dataset(dataset_id, user=owner, db=db)["dataset_id"] == dataset_id
    assert main.list_datasets(limit=50, user=other, db=db) == []
    with pytest.raises(HTTPException) as caught:
        main.get_dataset(dataset_id, user=other, db=db)
    assert caught.value.status_code == 404


def test_the_same_data_fetched_again_is_one_dataset_held_by_both_users(db, yahoo):
    first, second = make_user(db), make_user(db, "second@example.com")
    a = snapshot(db, first, symbols=("AAPL",))["result"]["datasets"][0]
    b = snapshot(db, second, symbols=("AAPL",))["result"]["datasets"][0]
    assert a["dataset_id"] == b["dataset_id"]
    assert db.execute(text("SELECT count(*) FROM dataset_versions")).scalar() == 1
    assert db.execute(text("SELECT count(*) FROM user_datasets")).scalar() == 2


def test_a_blob_altered_after_the_job_stored_it_is_never_registered(db, yahoo, monkeypatch):
    real = research_tasks.dataset_snapshot

    def tamper(params):
        output = real(params)
        item = output.result["stored"][0]
        path = DatasetStore(config.DATASET_ROOT).path_for(item["dataset_id"])
        os.chmod(path, 0o600)
        data = bytearray(path.read_bytes())
        data[-1] ^= 0x01
        path.write_bytes(bytes(data))
        return output

    monkeypatch.setattr(research_tasks, "dataset_snapshot", tamper)
    view = snapshot(db, make_user(db), symbols=("AAPL",))
    assert view["status"] == "failed"
    assert view["error"] == {"status_code": 500,
                             "detail": "The snapshot could not be verified after it was stored"}
    assert db.execute(text("SELECT count(*) FROM dataset_versions")).scalar() == 0
    assert db.execute(text("SELECT count(*) FROM user_datasets")).scalar() == 0


def test_a_reported_hash_that_does_not_match_the_file_is_refused(db, yahoo, monkeypatch):
    real = research_tasks.dataset_snapshot

    def misreport(params):
        output = real(params)
        output.result["stored"][0]["sha256"] = "0" * 64
        return output

    monkeypatch.setattr(research_tasks, "dataset_snapshot", misreport)
    view = snapshot(db, make_user(db), symbols=("AAPL",))
    assert view["status"] == "failed" and view["error"]["status_code"] == 500
    assert db.execute(text("SELECT count(*) FROM dataset_versions")).scalar() == 0


def test_a_reported_id_outside_the_store_is_refused(db, yahoo, monkeypatch):
    real = research_tasks.dataset_snapshot

    def escape(params):
        output = real(params)
        output.result["stored"][0]["dataset_id"] = "../../etc/passwd"
        return output

    monkeypatch.setattr(research_tasks, "dataset_snapshot", escape)
    view = snapshot(db, make_user(db), symbols=("AAPL",))
    assert view["status"] == "failed" and view["error"]["status_code"] == 500


def test_no_usable_symbol_fails_the_job_with_the_providers_reason(db, yahoo):
    view = snapshot(db, make_user(db), symbols=("NOPE",))
    assert view["status"] == "failed"
    assert view["error"] == {"status_code": 422,
                             "detail": "no data for any requested symbol: NOPE: no data returned"}


@pytest.mark.parametrize("kwargs, message", [
    ({"symbols": []}, "at least 1 item"),
    ({"symbols": [f"S{i}" for i in range(21)]}, "at most 20 items"),
    ({"symbols": ["AAPL", "aapl"]}, "distinct"),
    ({"symbols": ["../X"]}, "not a ticker symbol"),
    ({"symbols": ["AAPL; DROP"]}, "not a ticker symbol"),
    ({"start": END, "end": START}, "start must be before end"),
    ({"start": dt.date(1949, 12, 31)}, "on or after 1950-01-01"),
    ({"end": dt.date.today() + dt.timedelta(days=30)}, "end must be on or before"),
])
def test_snapshot_requests_are_validated_before_queueing(kwargs, message):
    values = {"symbols": ["AAPL"], "start": START, "end": END, **kwargs}
    with pytest.raises(ValidationError, match=message):
        schemas.DatasetSnapshotRequest(**values)


def test_yahoo_symbol_forms_are_accepted_by_the_api():
    req = schemas.DatasetSnapshotRequest(symbols=["^gspc", "eurusd=x", "gc=f", "reliance.ns"], start=START, end=END)
    assert req.symbols == ["^GSPC", "EURUSD=X", "GC=F", "RELIANCE.NS"]


def test_a_job_process_is_told_where_datasets_live():
    env = worker.job_environment({"PATH": "/bin", "DATASET_ROOT": "/data/datasets", "DATABASE_URL": "x"})
    assert env["DATASET_ROOT"] == "/data/datasets" and "DATABASE_URL" not in env


def test_the_snapshot_kind_is_registered_with_its_finalizer():
    kind = research_jobs.KINDS["dataset_snapshot"]
    assert kind.schema is schemas.DatasetSnapshotRequest and kind.audit_action == "DATASET_SNAPSHOT"
    assert worker._resolve(kind.finalize) is dataset_registry.finalize_snapshot_job
    assert worker._resolve(kind.target) is research_tasks.dataset_snapshot


def test_a_symbol_whose_frame_is_unusable_is_reported_by_the_job(db, yahoo):
    yahoo.frames["SPY"] = frame("SPY", ["2024-01-02", "2024-01-02"])
    view = snapshot(db, make_user(db), symbols=("AAPL", "SPY"))
    assert view["status"] == "succeeded"
    assert "unusable data" in view["result"]["unavailable"]["SPY"]


def test_the_real_job_process_inherits_the_dataset_root(monkeypatch, tmp_path):
    monkeypatch.setenv("DATASET_ROOT", str(tmp_path))
    process = worker.JobProcess()
    try:
        status, result, _, _ = process.run("research_job_helpers:environment_task", {}, 120, lambda: None)
    finally:
        process.stop()
    assert status == "ok" and result["environ"]["DATASET_ROOT"] == str(tmp_path)
