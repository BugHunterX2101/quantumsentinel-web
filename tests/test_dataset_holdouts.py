"""Sealed holdouts (services/dataset_holdouts.py, market_data, worker): a
family's data from its holdout start on never reaches a research task, a task
computes exactly what it would on datasets physically cut at the start, the
start can only move later, and a start set while a job runs fails the job."""
import datetime as dt
import threading
import time

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import sessionmaker

from backend import main, manage, models, schemas
from backend.services import dataset_holdouts, market_data
from backend.services.dataset_holdouts import HoldoutError
from backend.services.dataset_providers import FAMILIES, SnapshotRequest
from backend.services.market_data import MarketDataError, frame_from
from dataset_helpers import frame, provider
from market_helpers import fake_market
from research_job_helpers import InlineJobProcess, run_queued

EQUITY_DAYS = [d.date().isoformat() for d in pd.bdate_range("2023-01-02", "2024-01-31")]
CRYPTO_DAYS = [d.date().isoformat() for d in pd.date_range("2023-01-01", "2024-01-31")]
START = dt.date(2024, 1, 15)  # a Monday: the last equity session before it is Friday 2024-01-12


def datasets(frames):
    snap = provider(frames=frames, now=dt.datetime(2024, 2, 2, tzinfo=dt.timezone.utc)).snapshot(
        SnapshotRequest(tuple(frames), dt.date(2022, 12, 1), dt.date(2024, 2, 2)))
    return [(d.dataset_id, d) for d in snap.datasets]


def equity(days, dividend_days=()):
    return frame("AAPL", days, closes=[100.0 + i / 10 for i in range(len(days))],
                 Dividends=[0.5 if d in dividend_days else 0.0 for d in days])


def crypto(days):
    return frame("BTC-USD", days, closes=[40000.0 + i for i in range(len(days))])


# A dividend before the start and one after it: the one after must not rescale
# anything a task sees.
DIVIDENDS = ("2023-11-09", "2024-01-25")


def before(days, start=START):
    return [d for d in days if d < start.isoformat()]


# ── the cut ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("period", ["1mo", "6mo", "1y"])
def test_a_task_sees_exactly_what_it_would_on_data_cut_at_the_start(period):
    full = datasets({"AAPL": equity(EQUITY_DAYS, DIVIDENDS)})
    cut = datasets({"AAPL": equity(before(EQUITY_DAYS), DIVIDENDS[:1])})
    sealed, used, missing = frame_from(full, ["AAPL"], period, {"EQUITY_EOD": START})
    expected, _, _ = frame_from(cut, ["AAPL"], period)
    pd.testing.assert_frame_equal(sealed, expected)
    assert sealed.index[-1] == pd.Timestamp("2024-01-12")
    assert used == [full[0][0]] and missing == {}
    # Without the cut the window, and the prices the later dividend rescales, differ.
    unsealed, _, _ = frame_from(full, ["AAPL"], period)
    assert unsealed.index[-1] == pd.Timestamp("2024-01-31")
    assert unsealed["Close"]["AAPL"].loc["2024-01-12"] != sealed["Close"]["AAPL"].loc["2024-01-12"]


def test_each_family_is_cut_at_its_own_start():
    """The trailing window ends at the latest session any ticker still has, as
    it would on the cut datasets: here the crypto family has no holdout."""
    full = datasets({"AAPL": equity(EQUITY_DAYS, DIVIDENDS), "BTC-USD": crypto(CRYPTO_DAYS)})
    cut = datasets({"AAPL": equity(before(EQUITY_DAYS), DIVIDENDS[:1]), "BTC-USD": crypto(CRYPTO_DAYS)})
    sealed, _, _ = frame_from(full, ["AAPL", "BTC-USD"], "6mo", {"EQUITY_EOD": START})
    expected, _, _ = frame_from(cut, ["AAPL", "BTC-USD"], "6mo")
    pd.testing.assert_frame_equal(sealed, expected)
    assert sealed["Close"]["AAPL"].last_valid_index() == pd.Timestamp("2024-01-12")
    assert sealed["Close"]["BTC-USD"].last_valid_index() == pd.Timestamp("2024-01-31")
    # Both cut: the window ends before the earlier start.
    both, _, _ = frame_from(full, ["AAPL", "BTC-USD"], "6mo",
                            {"EQUITY_EOD": START, "CRYPTO_SPOT_EOD": dt.date(2024, 1, 10)})
    assert both.index[-1] == pd.Timestamp("2024-01-12")
    assert both["Close"]["BTC-USD"].last_valid_index() == pd.Timestamp("2024-01-09")


def test_no_holdout_or_another_familys_changes_nothing():
    full = datasets({"AAPL": equity(EQUITY_DAYS, DIVIDENDS)})
    plain = frame_from(full, ["AAPL"], "1y")
    for holdouts in (None, {}, {"CRYPTO_SPOT_EOD": START}):
        again = frame_from(full, ["AAPL"], "1y", holdouts)
        pd.testing.assert_frame_equal(again[0], plain[0])
        assert again[1:] == plain[1:]


def test_a_ticker_with_only_held_back_data_is_reported_missing():
    full = datasets({"AAPL": equity(EQUITY_DAYS), "BTC-USD": crypto(CRYPTO_DAYS)})
    early = {"EQUITY_EOD": dt.date(2022, 6, 1)}
    data, _, missing = frame_from(full, ["AAPL", "BTC-USD"], "1y", early)
    assert missing == {"AAPL": "only data from the holdout start on"}
    assert list(data.columns.get_level_values("Ticker").unique()) == ["BTC-USD"]
    with pytest.raises(MarketDataError, match="AAPL: only data from the holdout start on"):
        frame_from(full, ["AAPL"], "1y", early)


def test_a_dataset_whose_data_is_all_held_back_is_not_named_as_used():
    """The result names the datasets it was computed from: a later snapshot
    that holds only sessions from the start on contributed nothing."""
    early = datasets({"AAPL": equity(before(EQUITY_DAYS))})
    late = datasets({"AAPL": equity([d for d in EQUITY_DAYS if d >= START.isoformat()])})
    sealed, used, missing = frame_from(early + late, ["AAPL"], "1y", {"EQUITY_EOD": START})
    assert used == [early[0][0]] and missing == {}
    assert sealed.index[-1] == pd.Timestamp("2024-01-12")
    _, unsealed_used, _ = frame_from(early + late, ["AAPL"], "1y")
    assert sorted(unsealed_used) == sorted([early[0][0], late[0][0]])


@pytest.mark.parametrize("holdouts", [
    {"EQUITY_EOD": "2024-01-15"},
    {"EQUITY_EOD": dt.datetime(2024, 1, 15)},
    {1: dt.date(2024, 1, 15)},
    [("EQUITY_EOD", dt.date(2024, 1, 15))],
])
def test_a_malformed_holdout_is_refused_rather_than_ignored(holdouts):
    with pytest.raises(TypeError, match="holdouts"):
        market_data.MarketData(holdouts=holdouts)
    with pytest.raises(TypeError, match="holdouts"):
        with market_data.holdouts(holdouts):
            pass


# ── sessions ────────────────────────────────────────────────────────────────

@pytest.fixture
def market(monkeypatch, tmp_path):
    days = pd.bdate_range("2023-01-02", "2024-03-29")
    rng = np.random.default_rng(11)
    data = {}
    for i, t in enumerate(("AAA", "BBB", "CCC", "DDD")):
        data[("Close", t)] = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days))))
        data[("Volume", t)] = 1e6 + i
    return fake_market(monkeypatch, tmp_path, pd.DataFrame(data, index=days))


def test_sessions_opened_inside_holdouts_apply_them_and_say_so(market):
    starts = {"EQUITY_EOD": dt.date(2024, 1, 2), "CRYPTO_SPOT_EOD": dt.date(2023, 6, 1)}
    with market_data.holdouts(starts):
        with market_data.session() as data:
            frame_ = market_data.download(["AAA"], "6mo")
    assert frame_.index[-1] < pd.Timestamp("2024-01-02")
    # Only the families of the datasets used are named.
    assert data.record()["holdouts"] == {"EQUITY_EOD": "2024-01-02"}
    # The snapshot reaches back a period before the earliest start, and still to today.
    (symbol, start, end), = market.calls
    assert start == dt.date(2022, 12, 1) - dt.timedelta(days=7) and end == dt.date(2024, 4, 1)
    # Outside the context nothing is cut.
    with market_data.session() as data:
        assert market_data.download(["AAA"], "6mo").index[-1] == pd.Timestamp("2024-03-29")
    assert data.record()["holdouts"] == {}


def test_a_rerun_on_the_snapshot_with_the_holdout_sees_the_same_frame(market):
    starts = {"EQUITY_EOD": dt.date(2024, 1, 2)}
    with market_data.holdouts(starts):
        with market_data.session() as first:
            original = market_data.download(["AAA", "BBB"], "6mo")
        calls = len(market.calls)
        with market_data.session(first.record()["datasets"]) as again:
            rerun = market_data.download(["AAA", "BBB"], "6mo")
    pd.testing.assert_frame_equal(rerun, original)
    assert len(market.calls) == calls
    assert again.record()["holdouts"] == first.record()["holdouts"] == {"EQUITY_EOD": "2024-01-02"}


# ── the registry of starts ────────────────────────────────────────────────────

@pytest.fixture
def Session(make_engine):
    return sessionmaker(bind=make_engine(migrated=True))


@pytest.fixture
def db(Session):
    session = Session()
    yield session
    session.close()


def test_a_start_is_set_audited_and_only_moves_later(db):
    assert dataset_holdouts.current(db) == {}
    dataset_holdouts.set_start(db, "EQUITY_EOD", dt.date(2024, 1, 15), set_by="ops", reason="final test")
    dataset_holdouts.set_start(db, "EQUITY_EOD", dt.date(2025, 1, 15), set_by="ops", reason="rolled forward")
    dataset_holdouts.set_start(db, "CRYPTO_SPOT_EOD", dt.date(2024, 6, 1), set_by="ops", reason="crypto")
    assert dataset_holdouts.current(db) == {"EQUITY_EOD": dt.date(2025, 1, 15),
                                            "CRYPTO_SPOT_EOD": dt.date(2024, 6, 1)}
    for earlier in (dt.date(2025, 1, 15), dt.date(2024, 12, 31)):
        with pytest.raises(HoldoutError, match="starts on 2025-01-15; it can only move later"):
            dataset_holdouts.set_start(db, "EQUITY_EOD", earlier, set_by="ops", reason="oops")
    audits = {(a.resource_id, a.metadata_json["holdout_start"]): a.metadata_json
              for a in db.execute(select(models.AuditLog).where(
                  models.AuditLog.action == dataset_holdouts.AUDIT_ACTION)).scalars()}
    assert set(audits) == {("EQUITY_EOD", "2024-01-15"), ("EQUITY_EOD", "2025-01-15"),
                           ("CRYPTO_SPOT_EOD", "2024-06-01")}  # the refused ones are not audited
    assert audits[("EQUITY_EOD", "2024-01-15")]["previous_start"] is None
    assert audits[("EQUITY_EOD", "2025-01-15")] == {
        "holdout_start": "2025-01-15", "previous_start": "2024-01-15", "set_by": "ops", "reason": "rolled forward"}


def test_a_start_another_writer_overtook_is_refused_by_the_trigger(db, monkeypatch):
    """set_start's own check read the starts before a concurrent writer
    committed a later one; the trigger then refuses, and set_start says why."""
    dataset_holdouts.set_start(db, "EQUITY_EOD", dt.date(2025, 1, 15), set_by="ops", reason="first")
    monkeypatch.setattr(dataset_holdouts, "current", lambda db: {})  # the stale read
    with pytest.raises(HoldoutError, match="the holdout of EQUITY_EOD can only move later"):
        dataset_holdouts.set_start(db, "EQUITY_EOD", dt.date(2024, 6, 1), set_by="ops", reason="stale")
    monkeypatch.undo()
    assert dataset_holdouts.current(db) == {"EQUITY_EOD": dt.date(2025, 1, 15)}
    assert db.execute(select(models.AuditLog).where(models.AuditLog.action == dataset_holdouts.AUDIT_ACTION,
                                                    models.AuditLog.metadata_json["reason"].as_string() == "stale")
                      ).first() is None


@pytest.mark.parametrize("family, start, set_by, reason, message", [
    ("NOT_A_FAMILY", dt.date(2024, 1, 1), "ops", "x", "unknown dataset family"),
    ("EQUITY_EOD", "2024-01-01", "ops", "x", "must be a date"),
    ("EQUITY_EOD", dt.datetime(2024, 1, 1), "ops", "x", "must be a date"),
    ("EQUITY_EOD", dt.date(2024, 1, 1), " ", "x", "who sets the holdout and why"),
    ("EQUITY_EOD", dt.date(2024, 1, 1), "ops", "", "who sets the holdout and why"),
])
def test_what_set_start_refuses(db, family, start, set_by, reason, message):
    with pytest.raises(HoldoutError, match=message):
        dataset_holdouts.set_start(db, family, start, set_by=set_by, reason=reason)
    assert dataset_holdouts.current(db) == {}


def test_every_family_a_provider_produces_can_be_held_out():
    assert FAMILIES == {"EQUITY_EOD", "FUND_NAV_EOD", "INDEX_EOD", "CRYPTO_SPOT_EOD", "FX_SPOT_EOD",
                        "FUTURES_CONTINUOUS_EOD"}


def _insert(db, start, family="EQUITY_EOD"):
    db.execute(text("INSERT INTO dataset_holdouts (family, holdout_start, set_by, reason, created_at) "
                    "VALUES (:f, :s, 'ops', 'direct', now())"), {"f": family, "s": start})


@pytest.mark.parametrize("start", [dt.date(2024, 1, 15), dt.date(2023, 1, 1)])
def test_the_database_refuses_a_start_that_does_not_move_later_for_every_writer(db, start):
    _insert(db, dt.date(2024, 1, 15))
    db.commit()
    with pytest.raises(DBAPIError, match="it can only move later") as caught:
        _insert(db, start)
    assert caught.value.orig.sqlstate == "23001"  # restrict_violation
    db.rollback()
    assert dataset_holdouts.current(db) == {"EQUITY_EOD": dt.date(2024, 1, 15)}


@pytest.mark.parametrize("statement", [
    "UPDATE dataset_holdouts SET holdout_start = holdout_start - 365",
    "UPDATE dataset_holdouts SET reason = 'rewritten'",
    "DELETE FROM dataset_holdouts",
    "TRUNCATE dataset_holdouts",
])
def test_recorded_starts_never_change_even_for_the_owner(db, statement):
    _insert(db, dt.date(2024, 1, 15))
    db.commit()
    with pytest.raises(DBAPIError, match="immutable") as caught:
        db.execute(text(statement))
    assert caught.value.orig.sqlstate == "23001"
    db.rollback()
    assert db.execute(text("SELECT count(*) FROM dataset_holdouts")).scalar() == 1


def test_concurrent_inserts_cannot_both_pass_the_check(Session):
    """Two writers insert different starts for one family at once. Without the
    trigger's lock both would read the same latest start and both pass, so an
    earlier start could land after a later one; with it the second waits for
    the first to commit, then sees its start and is refused."""
    first, second, watcher = Session(), Session(), Session()
    try:
        # Connect all three first, so that only the insert can keep the thread waiting.
        app, = {s.execute(text("SELECT current_setting('application_name')")).scalar()
                for s in (first, second, watcher)}
        second.commit()
        _insert(first, dt.date(2025, 6, 1))  # uncommitted: holds the family's lock
        outcome = {}

        def insert_earlier():
            try:
                _insert(second, dt.date(2025, 3, 1))
                second.commit()
                outcome["inserted"] = True
            except DBAPIError as exc:
                second.rollback()
                outcome["error"] = exc

        thread = threading.Thread(target=insert_earlier)
        thread.start()
        deadline = time.monotonic() + 30
        while not watcher.execute(text(
                "SELECT count(*) FROM pg_stat_activity WHERE application_name = :app "
                "AND wait_event_type = 'Lock' AND wait_event = 'advisory'"), {"app": app}).scalar():
            watcher.commit()  # pg_stat_activity is a snapshot taken once per transaction
            assert thread.is_alive(), f"the second insert did not wait for the first: {outcome}"
            assert time.monotonic() < deadline, "the second insert never waited on the family's lock"
            time.sleep(0.05)
        first.commit()
        thread.join(30)
        assert not thread.is_alive()
        assert "error" in outcome, outcome
        assert outcome["error"].orig.sqlstate == "23001"
        assert "starts on 2025-06-01; it can only move later" in str(outcome["error"])
    finally:
        for s in (first, second, watcher):
            s.close()
    with Session() as check:
        assert check.execute(text("SELECT holdout_start FROM dataset_holdouts")).scalars().all() == \
            [dt.date(2025, 6, 1)]


def test_the_family_must_look_like_one(db):
    with pytest.raises(DBAPIError, match="ck_dataset_holdouts_family"):
        _insert(db, dt.date(2024, 1, 15), family="equity_eod")
    db.rollback()


# ── research jobs ─────────────────────────────────────────────────────────────

def make_user(db):
    user = models.User(email="researcher@example.com", password_hash="x", role="user")
    db.add(user)
    db.commit()
    return user


def correlation(db, user, job_process=None, **fields):
    return run_queued(db, main.correlation_endpoint(
        schemas.CorrelationRequest(assets=["AAA", "BBB", "CCC", "DDD"], period="1y", **fields),
        user=user, db=db), job_process)


class SeesFrames(InlineJobProcess):
    """Records the last session of every frame the task downloads."""

    def __init__(self, monkeypatch):
        self.last_sessions = []
        original = market_data.MarketData.download

        def download(data, tickers, period):
            frame_ = original(data, tickers, period)
            self.last_sessions.append(frame_.index[-1])
            return frame_

        monkeypatch.setattr(market_data.MarketData, "download", download)


def test_a_job_sees_only_data_before_the_start_and_its_result_and_audit_name_it(db, market, monkeypatch):
    user = make_user(db)
    dataset_holdouts.set_start(db, "EQUITY_EOD", dt.date(2024, 1, 2), set_by="ops", reason="final test")
    process = SeesFrames(monkeypatch)
    view = correlation(db, user, process)
    assert view["status"] == "succeeded", view
    assert process.last_sessions and max(process.last_sessions) < pd.Timestamp("2024-01-02")
    used = view["result"]["market_data"]
    assert used["holdouts"] == {"EQUITY_EOD": "2024-01-02"}
    audit = db.execute(select(models.AuditLog).where(
        models.AuditLog.action == "CORRELATION_ANALYSIS")).scalar_one()
    assert audit.metadata_json["holdouts"] == {"EQUITY_EOD": "2024-01-02"}
    # A rerun on the datasets the snapshot produced (which reach today) is cut the same way.
    again = correlation(db, user, process, dataset_ids=used["datasets"])
    assert again["status"] == "succeeded", again
    assert {k: v for k, v in again["result"].items() if k != "market_data"} == \
        {k: v for k, v in view["result"].items() if k != "market_data"}


def test_a_holdout_of_a_family_the_job_did_not_use_is_not_required_of_it(db, market):
    user = make_user(db)
    dataset_holdouts.set_start(db, "EQUITY_EOD", dt.date(2024, 1, 2), set_by="ops", reason="final test")
    dataset_holdouts.set_start(db, "CRYPTO_SPOT_EOD", dt.date(2023, 6, 1), set_by="ops", reason="crypto")
    view = correlation(db, user)
    assert view["status"] == "succeeded", view
    assert view["result"]["market_data"]["holdouts"] == {"EQUITY_EOD": "2024-01-02"}


def test_the_real_job_process_applies_the_starts_it_is_given():
    """The worker hands the starts to a separate child process over a pipe."""
    from backend import worker
    process = worker.JobProcess()
    try:
        starts = {"EQUITY_EOD": dt.date(2024, 1, 2), "CRYPTO_SPOT_EOD": dt.date(2023, 6, 1)}
        status, result, _, _ = process.run("research_job_helpers:holdouts_task", {}, 120, lambda: None,
                                           holdouts=starts)
        assert status == "ok" and result["holdouts"] == {"EQUITY_EOD": "2024-01-02",
                                                         "CRYPTO_SPOT_EOD": "2023-06-01"}
        # The next job in the same child gets only its own starts.
        status, result, _, _ = process.run("research_job_helpers:holdouts_task", {}, 120, lambda: None)
        assert status == "ok" and result["holdouts"] == {}
    finally:
        process.stop()


def test_without_a_holdout_a_job_is_unchanged(db, market, monkeypatch):
    process = SeesFrames(monkeypatch)
    view = correlation(db, make_user(db), process)
    assert view["status"] == "succeeded", view
    assert process.last_sessions == [pd.Timestamp("2024-03-29")]
    assert view["result"]["market_data"]["holdouts"] == {}


class SetsAHoldoutMidJob(InlineJobProcess):
    """An operator sets a holdout while the task runs."""

    def __init__(self, Session, start):
        self.Session, self.start = Session, start

    def run(self, *args, **kwargs):
        with self.Session() as other:
            dataset_holdouts.set_start(other, "EQUITY_EOD", self.start, set_by="ops", reason="mid-job")
        return super().run(*args, **kwargs)


@pytest.mark.parametrize("first", [None, dt.date(2023, 12, 1)])
def test_a_holdout_set_while_a_job_runs_fails_it(db, Session, market, first):
    user = make_user(db)
    if first is not None:
        dataset_holdouts.set_start(db, "EQUITY_EOD", first, set_by="ops", reason="before the job")
    view = correlation(db, user, SetsAHoldoutMidJob(Session, dt.date(2024, 1, 2)))
    assert view["status"] == "failed"
    assert view["error"] == {"status_code": 409,
                             "detail": "A dataset holdout was set while the research job ran; run it again"}
    assert db.execute(select(models.AuditLog).where(
        models.AuditLog.action == "CORRELATION_ANALYSIS")).first() is None


def test_a_task_that_ignores_the_holdout_is_refused(db, market, monkeypatch):
    """The registry does not take the task's word for which starts applied."""
    user = make_user(db)
    dataset_holdouts.set_start(db, "EQUITY_EOD", dt.date(2024, 1, 2), set_by="ops", reason="final test")
    original = market_data.MarketData.record
    monkeypatch.setattr(market_data.MarketData, "record", lambda self: {**original(self), "holdouts": {}})
    view = correlation(db, user)
    assert view["error"]["status_code"] == 409


# ── the operator CLI ─────────────────────────────────────────────────────────

def test_set_holdout_and_holdouts_commands(Session, capsys):
    assert manage._print_holdouts(Session) == 0
    assert "no dataset family has a holdout" in capsys.readouterr().out
    assert manage._set_holdout(Session, "EQUITY_EOD", dt.date(2024, 1, 15), "ops", "final test") == 0
    assert "EQUITY_EOD: research sees data before 2024-01-15" in capsys.readouterr().out
    assert manage._set_holdout(Session, "EQUITY_EOD", dt.date(2024, 1, 1), "ops", "earlier") == 1
    assert "it can only move later" in capsys.readouterr().err
    assert manage._print_holdouts(Session) == 0
    assert capsys.readouterr().out.split() == ["EQUITY_EOD", "2024-01-15"]


def test_set_holdout_needs_a_date_a_name_and_a_reason(capsys):
    for argv in (["set-holdout", "EQUITY_EOD", "15/01/2024", "--by", "ops", "--reason", "x"],
                 ["set-holdout", "EQUITY_EOD", "2024-01-15", "--reason", "x"],
                 ["set-holdout", "EQUITY_EOD", "2024-01-15", "--by", "ops"]):
        with pytest.raises(SystemExit) as exited:
            manage.main(argv)
        assert exited.value.code == 2
    assert "not a date (YYYY-MM-DD)" in capsys.readouterr().err
