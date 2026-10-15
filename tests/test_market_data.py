"""Research tasks read prices only from frozen datasets (services/market_data.py):
yf.download's layout and total-return adjustment rebuilt from split-adjusted
bars and dividends, a trailing window that a rerun on the same dataset ids
reproduces exactly, and jobs whose results name registered datasets."""
import datetime as dt
import stat
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from backend import config, main, models, schemas
from backend.services import market_data, research_jobs, research_tasks
from backend.services.dataset_format import DatasetIdentity
from backend.services.dataset_providers import SnapshotRequest
from backend.services.market_data import MarketDataError, frame_from, period_start, total_return_factor
from dataset_helpers import frame, provider
from market_helpers import fake_market
from research_job_helpers import run_queued

DAYS = [d.date().isoformat() for d in pd.bdate_range("2024-01-02", periods=8)]


def snapshot_datasets(frames, now=dt.datetime(2024, 2, 1, tzinfo=dt.timezone.utc)):
    snap = provider(frames=frames, now=now).snapshot(
        SnapshotRequest(tuple(frames), dt.date(2023, 1, 1), dt.date(2024, 2, 1)))
    return [(d.dataset_id, d) for d in snap.datasets]


# ── the window and the adjustment ───────────────────────────────────────────

def test_periods_count_back_calendar_months_from_the_end():
    assert period_start(dt.date(2026, 10, 12), "2y") == dt.date(2024, 10, 12)
    assert period_start(dt.date(2024, 2, 29), "1y") == dt.date(2023, 2, 28)
    assert period_start(dt.date(2024, 3, 31), "6mo") == dt.date(2023, 9, 30)
    assert period_start(dt.date(2024, 1, 15), "10y") == dt.date(2014, 1, 15)
    for bad in ("2", "0y", "1w", "100y", "", None):
        with pytest.raises(MarketDataError):
            period_start(dt.date(2024, 1, 1), bad)


def test_the_dividend_factor_is_yahoos_adj_close():
    """Yahoo's Adj Close: each close times prod(1 - dividend / previous close)
    over every later ex-date (verified live against Yahoo; see the module)."""
    sessions = np.array(DAYS, dtype="datetime64[D]")
    close = np.array([100.0, 101.0, 99.0, 102.0, 103.0, 101.0, 104.0, 105.0])
    dividends = {3: 0.5, 6: 0.25}
    expected, multiplier = np.empty(len(close)), 1.0
    for i in range(len(close) - 1, -1, -1):
        expected[i] = close[i] * multiplier
        if i in dividends:
            multiplier *= 1 - dividends[i] / close[i - 1]
    factor = total_return_factor(sessions, close, sessions[list(dividends)], np.array(list(dividends.values())))
    np.testing.assert_allclose(close * factor, expected, rtol=0, atol=1e-12)
    # A dividend on the first session has no earlier price to adjust.
    assert (total_return_factor(sessions, close, sessions[:1], np.array([1.0])) == 1.0).all()


@pytest.mark.parametrize("amount", [0.0, -0.1, 101.0])
def test_an_impossible_dividend_is_refused(amount):
    sessions = np.array(DAYS[:3], dtype="datetime64[D]")
    with pytest.raises(MarketDataError, match="dividend"):
        total_return_factor(sessions, np.array([100.0, 101.0, 102.0]), sessions[2:], np.array([amount]))


def test_frames_have_yf_downloads_layout_and_total_return_prices():
    crypto_days = [d.date().isoformat() for d in pd.date_range("2024-01-02", periods=11)]
    datasets = snapshot_datasets({
        "AAPL": frame("AAPL", DAYS, Dividends=[0, 0, 0, 0.5, 0, 0, 0, 0]),
        "BTC-USD": frame("BTC-USD", crypto_days, closes=[40000.0 + i for i in range(11)]),
    })
    data, used, missing = frame_from(datasets, ["BTC-USD", "AAPL"], "1y")
    assert list(data.columns.names) == ["Price", "Ticker"]
    assert list(data.columns) == [(f, t) for f in ("Close", "High", "Low", "Open", "Volume")
                                  for t in ("AAPL", "BTC-USD")]
    assert data.index.name == "Date" and data.index.tz is None
    assert list(data.index) == list(pd.DatetimeIndex(crypto_days))          # the union of calendars
    assert data["Close"]["AAPL"].isna().sum() == 3                           # a weekend and the 12th
    close = 100.0 + np.arange(8)
    factor = np.where(np.arange(8) < 3, 1 - 0.5 / close[2], 1.0)
    aapl = data.xs("AAPL", axis=1, level="Ticker").dropna()
    np.testing.assert_allclose(aapl["Close"], close * factor)
    np.testing.assert_allclose(aapl["Open"], (close - 0.5) * factor)       # one ratio for O/H/L/C
    np.testing.assert_allclose(aapl["High"], (close + 1.0) * factor)
    assert (aapl["Volume"] == 1_000_000.0).all()                            # volume is not adjusted
    assert used == sorted(i for i, _ in datasets) and missing == {}


def test_the_window_ends_at_the_datas_last_session():
    days = [d.date().isoformat() for d in pd.bdate_range("2023-01-02", "2024-01-31")]
    datasets = snapshot_datasets({"AAPL": frame("AAPL", days)}, now=dt.datetime(2024, 2, 2, tzinfo=dt.timezone.utc))
    data, _, _ = frame_from(datasets, ["AAPL"], "6mo")
    assert data.index[-1] == pd.Timestamp("2024-01-31")
    assert data.index[0] == pd.Timestamp("2023-08-01")    # 6 months before 2024-02-01, the day after
    # With several tickers it ends at the latest one's last session.
    earlier = [d for d in days if d <= "2023-12-29"]
    datasets = snapshot_datasets({"AAPL": frame("AAPL", days), "SPY": frame("SPY", earlier)},
                                 now=dt.datetime(2024, 2, 2, tzinfo=dt.timezone.utc))
    data, _, _ = frame_from(datasets, ["SPY", "AAPL"], "6mo")
    assert data["Close"]["SPY"].first_valid_index() == data.index[0] == pd.Timestamp("2023-08-01")


def test_missing_tickers_are_reported_and_none_at_all_is_an_error():
    datasets = snapshot_datasets({"AAPL": frame("AAPL", DAYS)})
    _, _, missing = frame_from(datasets, ["AAPL", "MSFT"], "1y")
    assert missing == {"MSFT": "no data"}
    with pytest.raises(MarketDataError, match="no data for any requested symbol: MSFT: no data"):
        frame_from(datasets, ["MSFT"], "1y")


def test_overlapping_datasets_merge_only_when_they_agree():
    first = snapshot_datasets({"AAPL": frame("AAPL", DAYS[:6])})
    second = snapshot_datasets({"AAPL": frame("AAPL", DAYS[3:], closes=[103.0 + i for i in range(5)])})
    data, used, _ = frame_from(first + second, ["AAPL"], "1y")
    assert list(data["Close"]["AAPL"]) == [100.0 + i for i in range(8)]
    assert used == sorted({first[0][0], second[0][0]})
    restated = snapshot_datasets({"AAPL": frame("AAPL", DAYS[3:], closes=[50.0 + i for i in range(5)])})
    with pytest.raises(MarketDataError, match="disagree on AAPL's bars"):
        frame_from(first + restated, ["AAPL"], "1y")
    paid = snapshot_datasets({"AAPL": frame("AAPL", DAYS[:6], Dividends=[0, 0, 0, 0.4, 0, 0])})
    other = snapshot_datasets({"AAPL": frame("AAPL", DAYS[:6], Dividends=[0, 0, 0, 0.3, 0, 0])})
    with pytest.raises(MarketDataError, match="disagree on AAPL's dividends"):
        frame_from(paid + other, ["AAPL"], "1y")


def test_identities_must_match_and_adjustments_must_be_understood():
    (dataset_id, dataset), = snapshot_datasets({"AAPL": frame("AAPL", DAYS)})
    other = type(dataset)(DatasetIdentity(**{**dataset.identity.as_dict(), "provider": "other"}),
                          dataset.tables)
    with pytest.raises(MarketDataError, match="different identities"):
        frame_from([(dataset_id, dataset), (other.dataset_id, other)], ["AAPL"], "1y")
    raw = type(dataset)(DatasetIdentity(**{**dataset.identity.as_dict(), "adjustment": "raw"}), dataset.tables)
    with pytest.raises(MarketDataError, match="'raw' are not supported"):
        frame_from([(raw.dataset_id, raw)], ["AAPL"], "1y")


def test_download_needs_a_session():
    with pytest.raises(RuntimeError, match="session"):
        market_data.download(["AAPL"], "1y")


def test_a_rerun_on_the_snapshots_datasets_sees_the_same_frame(monkeypatch, tmp_path):
    days = pd.bdate_range("2023-01-02", "2024-03-29")
    yahoo = fake_market(monkeypatch, tmp_path, pd.DataFrame(
        {("Close", "AAPL"): np.linspace(100, 150, len(days)), ("Volume", "AAPL"): 5e5}, index=days))
    yahoo.frames["AAPL"]["Dividends"] = np.where(np.arange(len(days)) == 200, 0.75, 0.0)
    with market_data.session() as first:
        original = market_data.download(["AAPL", "NOPE", "bad symbol"], "1y")
    record = first.record()
    assert record["source"] == "snapshot" and len(record["snapshots"]) == 1
    assert record["unavailable"] == {"BAD SYMBOL": "not a ticker symbol", "NOPE": "no data returned"}
    assert record["datasets"] == [s["dataset_id"] for s in record["snapshots"][0]["stored"]]
    # The snapshot asked for a week more than the period, ending tomorrow (now is 2024-03-31).
    assert yahoo.calls[0] == ("AAPL", dt.date(2023, 3, 25), dt.date(2024, 4, 1))
    raw = pd.Series(np.linspace(100, 150, len(days)), days)     # the dividend is added back before day 200
    assert original["Close"]["AAPL"].iloc[0] == pytest.approx(
        raw[original.index[0]] * (1 - 0.75 / raw.iloc[199]), rel=1e-12)
    calls = len(yahoo.calls)
    with market_data.session(record["datasets"]) as again:
        rerun = market_data.download(["AAPL"], "1y")
    pd.testing.assert_frame_equal(rerun, original)
    assert len(yahoo.calls) == calls                       # read from the store, not fetched
    assert again.record() == {"source": "datasets", "datasets": record["datasets"], "unavailable": {},
                              "snapshots": []}


def test_a_dataset_missing_from_the_store_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATASET_ROOT", tmp_path)
    with market_data.session(["qs-ds-sha256-v1:" + "0" * 64]):
        with pytest.raises(MarketDataError, match="could not be read from the store"):
            market_data.download(["AAPL"], "1y")


# ── research jobs ───────────────────────────────────────────────────────────

ASSETS = ["AAA", "BBB", "CCC", "DDD"]


@pytest.fixture
def Session(make_engine):
    return sessionmaker(bind=make_engine(migrated=True))


@pytest.fixture
def db(Session):
    session = Session()
    yield session
    session.close()


@pytest.fixture
def market(monkeypatch, tmp_path):
    days = pd.bdate_range("2023-01-02", periods=300)
    rng = np.random.default_rng(7)
    data = {}
    for i, t in enumerate(ASSETS):
        data[("Close", t)] = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days))))
        data[("Volume", t)] = 1e6 + i
    return fake_market(monkeypatch, tmp_path, pd.DataFrame(data, index=days))


def make_user(db, email="researcher@example.com"):
    user = models.User(email=email, password_hash="x", role="user")
    db.add(user)
    db.commit()
    return user


def correlation(db, user, **fields):
    return run_queued(db, main.correlation_endpoint(
        schemas.CorrelationRequest(assets=ASSETS, period="1y", **fields), user=user, db=db))


def test_every_kind_that_reads_prices_runs_in_a_market_data_session():
    for name, kind in research_jobs.KINDS.items():
        task = getattr(research_tasks, kind.target.partition(":")[2])
        assert kind.market_data == getattr(task, "uses_market_data", False), name
        assert ("dataset_ids" in kind.schema.model_fields) == kind.market_data, name
    assert sum(kind.market_data for kind in research_jobs.KINDS.values()) == 13


def test_a_job_registers_its_snapshot_and_names_it_in_the_result_and_audit(db, market):
    user = make_user(db)
    view = correlation(db, user)
    assert view["status"] == "succeeded", view
    used = view["result"]["market_data"]
    assert used["source"] == "snapshot" and used["unavailable"] == {}
    (dataset_id,) = used["datasets"]
    version = db.get(models.DatasetVersion, dataset_id)
    assert version.symbols == ASSETS and version.family == "EQUITY_EOD"
    holder = db.execute(select(models.UserDataset).where(models.UserDataset.dataset_id == dataset_id)).scalar_one()
    assert holder.user_id == user.id and holder.job_id == view["job_id"]
    audit = db.execute(select(models.AuditLog).where(
        models.AuditLog.action == "CORRELATION_ANALYSIS")).scalar_one()
    assert audit.metadata_json["datasets"] == [dataset_id]
    assert "snapshots" not in used


def test_a_rerun_on_the_dataset_ids_reproduces_the_result_without_fetching(db, market):
    user = make_user(db)
    first = correlation(db, user)["result"]
    calls = len(market.calls)
    second = correlation(db, user, dataset_ids=first["market_data"]["datasets"])
    assert second["status"] == "succeeded", second
    rerun = second["result"]
    assert len(market.calls) == calls
    assert rerun["market_data"] == {**first["market_data"], "source": "datasets"}
    assert {k: v for k, v in rerun.items() if k != "market_data"} == \
        {k: v for k, v in first.items() if k != "market_data"}


def test_a_request_can_name_only_datasets_its_user_holds(db, market):
    alice, bob = make_user(db, "alice@example.com"), make_user(db, "bob@example.com")
    (dataset_id,) = correlation(db, alice)["result"]["market_data"]["datasets"]
    with pytest.raises(HTTPException) as refused:
        main.correlation_endpoint(schemas.CorrelationRequest(assets=ASSETS, dataset_ids=[dataset_id]),
                                  user=bob, db=db)
    assert refused.value.status_code == 404
    assert db.execute(select(models.ResearchJob).where(models.ResearchJob.user_id == bob.id)).first() is None


@pytest.mark.parametrize("ids, message", [
    (["qs-ds-sha256-v1:" + "A" * 64], "not a dataset id"),
    (["../../etc/passwd"], "not a dataset id"),
    (["qs-ds-sha256-v1:" + "a" * 64 + "/../x"], "not a dataset id"),
    (["qs-ds-sha256-v1:" + "a" * 64] * 2, "must be distinct"),
    ([], "at least 1"),
    ([f"qs-ds-sha256-v1:{i:064x}" for i in range(11)], "at most 10"),
])
def test_dataset_ids_are_validated(ids, message):
    with pytest.raises(ValidationError, match=message):
        schemas.CorrelationRequest(dataset_ids=ids)


def test_a_snapshot_blob_that_changed_fails_the_job_and_registers_nothing(db, market, monkeypatch):
    original = market_data.MarketData.record

    def misreported(self):
        record = original(self)
        record["snapshots"][0]["stored"][0]["sha256"] = "0" * 64
        return record

    monkeypatch.setattr(market_data.MarketData, "record", misreported)
    view = correlation(db, make_user(db))
    assert view["status"] == "failed"
    assert view["error"] == {"status_code": 500,
                             "detail": "The snapshot could not be verified after it was stored"}
    assert db.execute(select(models.DatasetVersion)).first() is None


def test_a_result_computed_from_data_its_user_does_not_hold_is_refused(db, market, monkeypatch):
    original = market_data.MarketData.record

    def foreign(self):
        record = original(self)
        record["datasets"].append("qs-ds-sha256-v1:" + "f" * 64)
        return record

    monkeypatch.setattr(market_data.MarketData, "record", foreign)
    view = correlation(db, make_user(db))
    assert view["error"] == {"status_code": 500,
                             "detail": "The research job used market data that is not registered"}
    assert db.execute(select(models.DatasetVersion)).first() is None


def test_a_task_that_does_not_report_its_data_fails(db, market, monkeypatch):
    monkeypatch.setattr(research_tasks, "correlation", lambda params: research_tasks.TaskOutput({}, {}))
    view = correlation(db, make_user(db))
    assert view["error"] == {"status_code": 500,
                             "detail": "The research job did not report the market data it used"}


def test_the_data_a_trial_ran_on_is_not_part_of_its_configuration(db, market):
    user = make_user(db)
    request = dict(assets=["AAA", "BBB"], benchmark="CCC", period="1y", slow_window=30, fast_window=10)
    first = run_queued(db, main.advanced_backtest(schemas.AdvancedBacktestRequest(**request), user=user, db=db))
    assert first["status"] == "succeeded", first
    ids = first["result"]["market_data"]["datasets"]
    again = run_queued(db, main.advanced_backtest(
        schemas.AdvancedBacktestRequest(**request, dataset_ids=ids), user=user, db=db))
    assert again["status"] == "succeeded", again
    assert first["result"]["trials_in_family"] == again["result"]["trials_in_family"] == 1


def test_a_backtest_on_a_dataset_changed_on_disk_fails_with_the_reason(db, market):
    user = make_user(db)
    request = dict(assets=["AAA", "BBB"], benchmark="CCC", period="1y", slow_window=30, fast_window=10)
    first = run_queued(db, main.advanced_backtest(schemas.AdvancedBacktestRequest(**request), user=user, db=db))
    (dataset_id,) = first["result"]["market_data"]["datasets"]
    digest = dataset_id.partition(":")[2]
    (blob,) = Path(config.DATASET_ROOT).rglob(digest)
    blob.chmod(stat.S_IREAD | stat.S_IWRITE)
    blob.write_bytes(blob.read_bytes()[:-1])
    again = run_queued(db, main.advanced_backtest(
        schemas.AdvancedBacktestRequest(**request, dataset_ids=[dataset_id]), user=user, db=db))
    assert again["status"] == "failed"
    assert again["error"] == {"status_code": 422,
                              "detail": f"dataset {dataset_id} could not be read from the store"}
