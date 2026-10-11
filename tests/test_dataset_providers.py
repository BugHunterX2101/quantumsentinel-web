"""The yfinance provider: families, session dates, adjustment identity and
licence, from Yahoo-shaped frames (tests/dataset_helpers.py), no network."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from backend.services import dataset_providers
from backend.services.dataset_format import BARS_V1, Table
from backend.services.dataset_providers import (LICENCES, ProviderError, SnapshotRequest, YahooProvider)
from dataset_helpers import META, NOW, WEEK, FakeYahoo, frame, provider, snapshot


def dates(table, column="session_date"):
    return [str(d) for d in table.columns[column]]


def by_family(snap):
    return {d.identity.family: d for d in snap.datasets}


def test_symbols_of_different_families_become_separate_datasets():
    snap = snapshot(("AAPL", "BTC-USD", "EURUSD=X", "VOD.L"))
    datasets = by_family(snap)
    assert set(datasets) == {"EQUITY_EOD", "CRYPTO_SPOT_EOD", "FX_SPOT_EOD"}
    assert sorted(set(datasets["EQUITY_EOD"].table("bars").columns["symbol"])) == ["AAPL", "VOD.L"]
    assert datasets["CRYPTO_SPOT_EOD"].identity.session_boundary == "utc-00:00"
    assert datasets["EQUITY_EOD"].identity.session_boundary == "exchange-local"
    assert datasets["FX_SPOT_EOD"].identity.session_boundary == "exchange-local"
    assert len({d.dataset_id for d in snap.datasets}) == 3


def test_yahoo_identity_states_split_adjustment_and_the_personal_use_licence():
    identity = by_family(snapshot(("AAPL",))).popitem()[1].identity
    assert identity.as_dict() == {
        "family": "EQUITY_EOD", "provider": "yahoo", "licence_id": "yahoo-personal-use",
        "frequency": "1d", "session_boundary": "exchange-local", "adjustment": "split",
        "survivorship": "current_listings"}
    licence = LICENCES["yahoo-personal-use"]
    assert licence.personal_use_only and not licence.redistribution_allowed


def test_session_dates_are_the_local_trading_day_not_a_utc_shift():
    # New York midnight is 05:00 UTC; Tokyo midnight is 15:00 UTC the day
    # before. The session date is the local date either way.
    tokyo = {"instrumentType": "EQUITY", "exchangeName": "JPX", "exchangeTimezoneName": "Asia/Tokyo",
             "currency": "JPY"}
    snap = snapshot(("7203.T", "AAPL"), frames={"7203.T": frame("7203.T", ["2024-01-04", "2024-01-05"],
                                                                  tz="Asia/Tokyo"),
                                                "AAPL": frame("AAPL", ["2024-01-04", "2024-01-05"])},
                    meta={**META, "7203.T": tokyo})
    bars = snap.datasets[0].table("bars")
    assert dates(bars) == ["2024-01-04", "2024-01-05", "2024-01-04", "2024-01-05"]
    assert list(bars.columns["symbol"]) == ["7203.T", "7203.T", "AAPL", "AAPL"]


def test_the_session_still_open_at_retrieval_is_left_out():
    # NOW is 2024-01-10 10:00 in New York and 15:00 in London and UTC.
    for symbol in ("AAPL", "BTC-USD", "VOD.L"):
        bars = snapshot((symbol,)).datasets[0].table("bars")
        assert dates(bars)[-1] == "2024-01-09", symbol
    # A day later the 10th is complete everywhere.
    later = snapshot(("AAPL",), now=NOW + dt.timedelta(days=1)).datasets[0].table("bars")
    assert dates(later)[-1] == "2024-01-10"


def test_the_local_date_decides_what_is_complete():
    # 2024-01-10 03:00 UTC is still the 9th in New York: the 9th is open there.
    early = dt.datetime(2024, 1, 10, 3, 0, tzinfo=dt.timezone.utc)
    assert dates(snapshot(("AAPL",), now=early).datasets[0].table("bars"))[-1] == "2024-01-08"
    assert dates(snapshot(("BTC-USD",), now=early).datasets[0].table("bars"))[-1] == "2024-01-09"


def test_a_row_without_a_close_is_dropped_and_missing_prices_are_null():
    frames = {"VOD.L": frame("VOD.L", WEEK[:4], closes=[70.1, np.nan, 70.3, 70.4],
                             Open=[np.nan, 1.0, 2.0, 3.0])}
    bars = snapshot(("VOD.L",), frames=frames).datasets[0].table("bars")
    assert dates(bars) == ["2024-01-02", "2024-01-04", "2024-01-05"]
    assert np.isnan(bars.columns["open"][0]) and bars.columns["open"][1] == 2.0


def test_dividends_and_splits_become_dated_corporate_actions():
    actions = snapshot(("AAPL",)).datasets[0].table("corporate_actions")
    assert dates(actions, "ex_date") == ["2024-01-04", "2024-01-08"]
    assert list(actions.columns["action"]) == ["dividend", "split"]
    assert list(actions.columns["value"]) == [0.24, 4.0]
    assert list(actions.columns["source"]) == ["yahoo", "yahoo"]
    assert np.isnat(actions.columns["announced_at"]).all()


def test_capital_gains_are_recorded_when_yahoo_reports_them():
    frames = {"SPY": frame("SPY", WEEK[:3], **{"Capital Gains": [0.0, 0.5, 0.0]})}
    actions = snapshot(("SPY",), frames=frames).datasets[0].table("corporate_actions")
    assert list(actions.columns["action"]) == ["capital_gain"] and list(actions.columns["value"]) == [0.5]


def test_fx_volume_is_null_not_zero():
    bars = by_family(snapshot(("EURUSD=X",)))["FX_SPOT_EOD"].table("bars")
    assert np.isnan(bars.columns["volume"]).all()


def test_the_instruments_table_records_where_each_symbol_trades():
    instruments = by_family(snapshot(("AAPL", "VOD.L")))["EQUITY_EOD"].table("instruments")
    rows = list(zip(*(instruments.columns[c] for c in ("symbol", "instrument_type", "exchange",
                                                       "timezone", "currency"))))
    assert rows == [("AAPL", "equity", "NMS", "America/New_York", "USD"),
                    ("VOD.L", "equity", "LSE", "Europe/London", "GBp")]


@pytest.mark.parametrize("meta, frames, reason", [
    ({"instrumentType": "OPTION", "exchangeName": "OPR", "exchangeTimezoneName": "America/New_York",
      "currency": "USD"}, None, "unsupported instrument type"),
    ({"instrumentType": "EQUITY", "exchangeName": "NMS", "exchangeTimezoneName": "Mars/Base",
      "currency": "USD"}, None, "unknown timezone"),
    ({"instrumentType": "EQUITY", "exchangeName": "NMS", "currency": "USD"}, None, "where the symbol trades"),
    ({"instrumentType": "CRYPTOCURRENCY", "exchangeName": "CCC", "exchangeTimezoneName": "America/New_York",
      "currency": "USD"}, None, "expected UTC days"),
    (None, pd.DataFrame(), "no data returned"),
])
def test_a_symbol_the_provider_cannot_describe_is_reported_not_guessed(meta, frames, reason):
    fetched = {"AAPL": frame("AAPL", WEEK), "XYZ": frame("AAPL", WEEK) if frames is None else frames}
    snap = snapshot(("AAPL", "XYZ"), frames=fetched, meta={**META, "XYZ": meta or {}})
    assert reason in snap.unavailable["XYZ"]
    assert [d.identity.family for d in snap.datasets] == ["EQUITY_EOD"]
    assert set(snap.datasets[0].table("bars").columns["symbol"]) == {"AAPL"}


def test_timestamps_that_are_not_local_midnights_are_refused():
    shifted = frame("AAPL", WEEK)
    shifted.index = shifted.index.tz_convert("UTC").tz_localize(None).tz_localize("America/New_York")
    snap = snapshot(("AAPL", "SPY"), frames={"AAPL": shifted, "SPY": frame("SPY", WEEK)})
    assert "not local midnights" in snap.unavailable["AAPL"]
    naive = frame("AAPL", WEEK)
    naive.index = naive.index.tz_localize(None)
    snap = snapshot(("AAPL", "SPY"), frames={"AAPL": naive, "SPY": frame("SPY", WEEK)})
    assert "no timezone" in snap.unavailable["AAPL"]


def test_duplicate_sessions_fail_only_their_symbol():
    doubled = frame("AAPL", ["2024-01-02", "2024-01-02"])
    snap = snapshot(("AAPL", "SPY"), frames={"AAPL": doubled, "SPY": frame("SPY", WEEK)})
    assert "unusable data" in snap.unavailable["AAPL"]


def test_no_usable_symbol_at_all_is_an_error():
    with pytest.raises(ProviderError, match="no data for any requested symbol"):
        snapshot(("AAPL",), frames={})
    with pytest.raises(ProviderError, match="no completed sessions"):
        snapshot(("AAPL",), frames={"AAPL": frame("AAPL", ["2024-01-10"])})


def test_the_same_data_gets_the_same_id_whatever_the_request_order():
    first = snapshot(("BTC-USD", "VOD.L", "AAPL"))
    second = snapshot(("AAPL", "BTC-USD", "VOD.L"))
    assert [d.dataset_id for d in first.datasets] == [d.dataset_id for d in second.datasets]
    assert first.request.symbols == ("AAPL", "BTC-USD", "VOD.L")


def test_a_changed_price_changes_the_id():
    closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0]
    first = snapshot(("AAPL",), frames={"AAPL": frame("AAPL", WEEK, closes=closes)})
    closes[5] = 105.5
    second = snapshot(("AAPL",), frames={"AAPL": frame("AAPL", WEEK, closes=closes)})
    assert first.datasets[0].dataset_id != second.datasets[0].dataset_id


def test_the_snapshot_records_its_provenance():
    fake = FakeYahoo()
    snap = YahooProvider(fetch=fake, client="yfinance 9.9", clock=lambda: NOW).snapshot(
        SnapshotRequest(("AAPL",), dt.date(2024, 1, 1), dt.date(2024, 1, 11)))
    assert (snap.provider, snap.client, snap.retrieved_at) == ("yahoo", "yfinance 9.9", NOW)
    assert snap.request.as_dict() == {"symbols": ["AAPL"], "start": "2024-01-01", "end": "2024-01-11"}
    assert fake.calls == [("AAPL", dt.date(2024, 1, 1), dt.date(2024, 1, 11))]


@pytest.mark.parametrize("symbols, start, end, message", [
    ((), dt.date(2024, 1, 1), dt.date(2024, 2, 1), "1 to 50 symbols"),
    (tuple(f"S{i}" for i in range(51)), dt.date(2024, 1, 1), dt.date(2024, 2, 1), "1 to 50 symbols"),
    (("AAPL", "AAPL"), dt.date(2024, 1, 1), dt.date(2024, 2, 1), "distinct"),
    (("aapl",), dt.date(2024, 1, 1), dt.date(2024, 2, 1), "not a symbol"),
    (("../X",), dt.date(2024, 1, 1), dt.date(2024, 2, 1), "not a symbol"),
    (("A" * 21,), dt.date(2024, 1, 1), dt.date(2024, 2, 1), "not a symbol"),
    (("AAPL",), dt.date(2024, 2, 1), dt.date(2024, 2, 1), "before end"),
    (("AAPL",), dt.datetime(2024, 1, 1), dt.date(2024, 2, 1), "must be dates"),
    (("AAPL",), "2024-01-01", dt.date(2024, 2, 1), "must be dates"),
])
def test_requests_are_validated(symbols, start, end, message):
    with pytest.raises(ValueError, match=message):
        SnapshotRequest(symbols, start, end)


def test_yahoo_symbol_forms_are_accepted():
    request = SnapshotRequest(("^GSPC", "EURUSD=X", "GC=F", "RELIANCE.NS", "BRK-B", "7203.T"),
                              dt.date(2024, 1, 1), dt.date(2024, 2, 1))
    assert request.symbols == ("7203.T", "BRK-B", "EURUSD=X", "GC=F", "RELIANCE.NS", "^GSPC")


def test_every_yahoo_instrument_type_has_a_family_and_merged_bars_are_bars():
    for instrument_type, family in dataset_providers._YAHOO_TYPES.values():
        assert family.endswith("_EOD") and instrument_type.islower()
    bars = by_family(snapshot(("AAPL", "VOD.L")))["EQUITY_EOD"].table("bars")
    assert isinstance(bars, Table) and bars.schema is BARS_V1


class _Ticker:
    """yfinance.Ticker as it behaves for an unknown symbol (1.5.2): an empty
    history, then a KeyError from get_history_metadata."""
    history_result = pd.DataFrame()
    history_error = None

    def __init__(self, symbol):
        self.symbol = symbol

    def history(self, **kwargs):
        if self.history_error:
            raise self.history_error
        return self.history_result

    def get_history_metadata(self):
        raise KeyError("exchangeTimezoneName")


def test_an_unknown_yahoo_symbol_is_reported_without_asking_for_metadata(monkeypatch):
    import yfinance

    monkeypatch.setattr(yfinance, "Ticker", _Ticker)
    frame_, meta = dataset_providers._yahoo_fetch("NOPE", dt.date(2024, 1, 1), dt.date(2024, 2, 1))
    assert frame_.empty and meta == {}


def test_a_yfinance_failure_fails_only_its_symbol(monkeypatch):
    import yfinance

    class Failing(_Ticker):
        history_error = ConnectionError("rate limited")

    monkeypatch.setattr(yfinance, "Ticker", Failing)
    with pytest.raises(ProviderError, match=r"the provider failed \(ConnectionError\)"):
        dataset_providers._yahoo_fetch("AAPL", dt.date(2024, 1, 1), dt.date(2024, 2, 1))

    def fetch(symbol, start, end):
        if symbol == "AAPL":
            return dataset_providers._yahoo_fetch(symbol, start, end)
        return frame("SPY", WEEK), META["SPY"]

    snap = YahooProvider(fetch=fetch, client="t", clock=lambda: NOW).snapshot(
        SnapshotRequest(("AAPL", "SPY"), dt.date(2024, 1, 1), dt.date(2024, 1, 11)))
    assert snap.unavailable == {"AAPL": "the provider failed (ConnectionError)"}
