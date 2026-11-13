"""QuantumSentinel — market data for research tasks, from frozen datasets only.

A research task never downloads prices itself. Inside the task's session
(``session()``), ``download(tickers, period)`` returns what
``yf.download(tickers, period=period, auto_adjust=True)`` used to return —
open/high/low/close adjusted for splits and dividends, volume as traded, one
(field, ticker) column each on the union of the tickers' session dates — but
built from datasets (dataset_format) that the task's result then names:

* with ``dataset_ids`` (the request's; the API checked that the user holds
  them), from those stored datasets, each rehashed against its id on read;
* otherwise from a snapshot the provider takes now, stored under
  DATASET_ROOT. The worker reads each blob back, verifies it and registers it
  before it stores the result (dataset_registry.record_market_data).

``period`` ("6mo", "2y", ...) is a trailing window that ends at the latest
session the data has for the requested tickers, so a rerun on the dataset ids
a snapshot produced sees exactly the bars the first run saw. Sessions on or
after the day a snapshot is taken are never included (dataset_providers).

Holdouts. A dataset family with a holdout start (dataset_holdouts) is read as
if its data ended the day before: bars from the start on and dividends going
ex from it on are dropped before anything else, so the trailing window ends at
the last session before the start and no later dividend rescales an earlier
price. A task therefore computes exactly what it would on datasets cut at the
start. The research worker passes the starts in (``holdouts()``) and the
result names those that applied to the datasets it used. Splits are
different: the provider serves prices already adjusted for every split up to
the day of the snapshot, so a split after the start still scales earlier
price levels (returns are unchanged). Only point-in-time data (adjustment
"raw") would avoid that.

Total return. The provider's bars are split-adjusted only (identity
adjustment "split"). Dividends are added back with the backward factor
prod(1 - dividend / previous close) over later ex-dates, which is how Yahoo
derives its Adj Close: the two matched within 1e-6 over ten years of AAPL,
MSFT, SPY, KO, T and twelve mutual funds, and exactly at every ex-date of
SAP.DE, 7203.T and RY.TO (2026-10-11). Yahoo reports fund distributions as
dividends; the capital-gain amounts it returned were zero for all of them, so
capital_gain actions are not used. For London listings quoted in pence (GBp)
Yahoo's Adj Close divides the dividend by 100 times the close, so it adds
back a hundredth of each dividend; this factor does not (VOD.L's 3.86p
half-year dividend on a 78.48p close is 4.9%, as Vodafone paid, not 0.05%).
"""
from __future__ import annotations

import calendar
import contextlib
import contextvars
import datetime as dt
import logging
import re
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .dataset_format import Dataset, DatasetError, DatasetIntegrityError
from .dataset_providers import MAX_SYMBOLS, SYMBOL, ProviderError, SnapshotRequest, YahooProvider
from .dataset_store import DatasetStore

log = logging.getLogger(__name__)

PERIOD = re.compile(r"([1-9][0-9]?)(mo|y)")
FIELDS = ("Close", "High", "Low", "Open", "Volume")  # yf.download's column order
_PRICES = ("Open", "High", "Low", "Close")
_FETCH_MARGIN_DAYS = 7
# Adjustments whose prices this module can complete to total return.
_ADJUSTMENTS = ("split", "split_dividend")


class MarketDataError(ValueError):
    """The requested data cannot be provided; the message is safe to show."""


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def default_provider():
    return YahooProvider(clock=lambda: utcnow())


def period_start(end: dt.date, period: str) -> dt.date:
    """The first day of the ``period`` that ends before ``end`` (exclusive)."""
    match = PERIOD.fullmatch(period) if isinstance(period, str) else None
    if match is None:
        raise MarketDataError(f"unsupported period {period!r}")
    months = int(match.group(1)) * (12 if match.group(2) == "y" else 1)
    year, month = divmod(end.year * 12 + end.month - 1 - months, 12)
    month += 1
    return dt.date(year, month, min(end.day, calendar.monthrange(year, month)[1]))


def total_return_factor(sessions: np.ndarray, close: np.ndarray, ex_dates: np.ndarray,
                        amounts: np.ndarray, symbol: str = "") -> np.ndarray:
    """Per session, the product of (1 - dividend / previous close) over every
    later ex-date: the factor that adds dividends back into split-adjusted
    prices. ``sessions`` (datetime64[D]) is sorted; a dividend with no earlier
    session in the data changes nothing in it."""
    factor = np.ones(len(close))
    for ex_date, amount in sorted(zip(ex_dates.tolist(), amounts.tolist())):
        first = int(np.searchsorted(sessions, np.datetime64(ex_date, "D")))
        if first == 0:
            continue
        previous = float(close[first - 1])
        if not 0.0 < amount < previous:
            raise MarketDataError(f"{symbol}: the dividend of {amount} on {ex_date} is not "
                                  f"between 0 and the previous close {previous}")
        factor[:first] *= 1.0 - amount / previous
    return factor


@dataclass(frozen=True)
class _Series:
    """One symbol's daily bars and dividends."""
    identity: dict
    bars: pd.DataFrame  # open/high/low/close/volume on a DatetimeIndex of sessions
    dividends: dict     # ex_date (datetime.date) -> amount


def _series(dataset: Dataset, symbol: str, before: dt.date | None = None) -> _Series | None:
    """``symbol``'s bars and dividends in ``dataset``, or None if it has none.
    With ``before`` (a holdout start), only the bars of sessions and the
    dividends going ex before it; the bars may then be empty."""
    bars = dataset.table("bars").columns
    rows = bars["symbol"] == symbol
    if not rows.any():
        return None
    if before is not None:
        rows = rows & (bars["session_date"] < np.datetime64(before, "D"))
    sessions = pd.DatetimeIndex(bars["session_date"][rows].astype("datetime64[ns]"))
    frame = pd.DataFrame({name: bars[name][rows] for name in ("open", "high", "low", "close", "volume")},
                         index=sessions)
    dividends = {}
    try:
        actions = dataset.table("corporate_actions").columns
    except KeyError:
        actions = None
    if actions is not None:
        mine = (actions["symbol"] == symbol) & (actions["action"] == "dividend")
        if before is not None:
            mine = mine & (actions["ex_date"] < np.datetime64(before, "D"))
        dividends = dict(zip(actions["ex_date"][mine].tolist(), actions["value"][mine].tolist()))
    return _Series(dataset.identity.as_dict(), frame, dividends)


def _merged(symbol: str, parts: list[_Series]) -> _Series:
    """One series from every dataset that holds ``symbol``. They must describe
    the same data: one identity, and equal values wherever they overlap."""
    if len(parts) == 1:
        return parts[0]
    if any(p.identity != parts[0].identity for p in parts[1:]):
        raise MarketDataError(f"{symbol} is in datasets with different identities")
    bars = pd.concat([p.bars for p in parts])
    overlap = bars.index.duplicated(keep=False)
    if overlap.any() and (bars[overlap].groupby(level=0).nunique(dropna=False) > 1).to_numpy().any():
        raise MarketDataError(f"the datasets disagree on {symbol}'s bars")
    dividends = {}
    for part in parts:
        for ex_date, amount in part.dividends.items():
            if dividends.setdefault(ex_date, amount) != amount:
                raise MarketDataError(f"the datasets disagree on {symbol}'s dividends")
    bars = bars[~bars.index.duplicated()].sort_index()
    return _Series(parts[0].identity, bars, dividends)


def _adjusted(symbol: str, series: _Series) -> pd.DataFrame:
    """Open/High/Low/Close with splits and dividends included; Volume as is."""
    adjustment = series.identity["adjustment"]
    if adjustment not in _ADJUSTMENTS:
        raise MarketDataError(f"{symbol}: prices adjusted as {adjustment!r} are not supported")
    bars = series.bars
    factor = np.ones(len(bars))
    if adjustment == "split" and series.dividends:
        factor = total_return_factor(bars.index.to_numpy().astype("datetime64[D]"),
                                     bars["close"].to_numpy(dtype=float),
                                     np.array(list(series.dividends), dtype="datetime64[D]"),
                                     np.array(list(series.dividends.values()), dtype=float), symbol)
    out = {field: bars[field.lower()].to_numpy(dtype=float) * factor for field in _PRICES}
    out["Volume"] = bars["volume"].to_numpy(dtype=float)
    return pd.DataFrame(out, index=bars.index)


def frame_from(datasets: Iterable[tuple[str, Dataset]], tickers: list[str], period: str,
               holdouts: dict[str, dt.date] | None = None) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """``tickers`` from ``datasets`` ((id, dataset) pairs) over the trailing
    ``period``, in yf.download's layout, each dataset cut at its family's
    holdout start in ``holdouts``. Returns the frame, the ids of the datasets
    that supplied it and the tickers it could not supply, with why."""
    datasets = list(datasets)
    holdouts = holdouts or {}
    found, missing = {}, {}
    for ticker in tickers:
        parts = [(dataset_id, s) for dataset_id, dataset in datasets
                 if (s := _series(dataset, ticker, holdouts.get(dataset.identity.family))) is not None]
        if not parts:
            missing[ticker] = "no data"
            continue
        series = _merged(ticker, [s for _, s in parts])
        if series.bars.empty:
            missing[ticker] = "only data from the holdout start on"
        else:
            found[ticker] = (series, {i for i, s in parts if not s.bars.empty})
    columns, used = {}, set()
    if found:
        anchor = max(series.bars.index.max() for series, _ in found.values()) + pd.Timedelta(days=1)
        start = pd.Timestamp(period_start(anchor.date(), period))
        for ticker, (series, ids) in found.items():
            adjusted = _adjusted(ticker, series)
            adjusted = adjusted[adjusted.index >= start]
            if adjusted.empty:
                missing[ticker] = "no sessions in the period"
                continue
            used |= ids
            for field in FIELDS:
                columns[(field, ticker)] = adjusted[field]
    if not columns:
        raise MarketDataError("no data for any requested symbol: "
                              + "; ".join(f"{t}: {why}" for t, why in missing.items()))
    order = sorted(columns, key=lambda key: (FIELDS.index(key[0]), key[1]))
    frame = pd.concat([columns[key] for key in order], axis=1, sort=True)  # the union of calendars
    frame.columns = pd.MultiIndex.from_tuples(order, names=["Price", "Ticker"])
    frame.index.name = "Date"
    return frame, sorted(used), missing


def _checked_holdouts(holdouts) -> dict[str, dt.date]:
    """``holdouts`` as {family: date}; anything else is refused, since a
    boundary that silently fails to apply would unseal the holdout."""
    if holdouts is None:
        return {}
    if not isinstance(holdouts, dict):
        raise TypeError("holdouts must map a dataset family to a date")
    for family, start in holdouts.items():
        if not isinstance(family, str) or not isinstance(start, dt.date) or isinstance(start, dt.datetime):
            raise TypeError("holdouts must map a dataset family to a date")
    return dict(holdouts)


class MarketData:
    """The market data one research task used (see the module docstring)."""

    def __init__(self, dataset_ids: list[str] | None = None, *, provider=None,
                 store: DatasetStore | None = None, clock: Callable[[], dt.datetime] | None = None,
                 holdouts: dict[str, dt.date] | None = None):
        from ..config import DATASET_ROOT

        self.dataset_ids = list(dataset_ids) if dataset_ids else None
        self._provider = provider
        self._store = store or DatasetStore(DATASET_ROOT)
        self._clock = clock or utcnow
        self.holdouts = _checked_holdouts(holdouts)
        self._loaded: dict[str, Dataset] = {}
        self._families: dict[str, str] = {}  # dataset id -> its family
        self.snapshots: list[dict] = []
        self.used: set[str] = set()
        self.unavailable: dict[str, str] = {}

    def download(self, tickers: list[str] | str, period: str) -> pd.DataFrame:
        if isinstance(tickers, str):
            tickers = [tickers]
        symbols = list(dict.fromkeys(str(t).strip().upper() for t in tickers))
        if self.dataset_ids is not None:
            sources = [(dataset_id, self._load(dataset_id)) for dataset_id in self.dataset_ids]
        else:
            sources = self._snapshot(symbols, period)
        for dataset_id, dataset in sources:
            self._families[dataset_id] = dataset.identity.family
        frame, used, missing = frame_from(sources, symbols, period, self.holdouts)
        self.used.update(used)
        for symbol, why in missing.items():
            self.unavailable.setdefault(symbol, why)
        return frame

    def _load(self, dataset_id: str) -> Dataset:
        if dataset_id not in self._loaded:
            try:
                self._loaded[dataset_id] = self._store.get(dataset_id)
            except (DatasetError, DatasetIntegrityError) as exc:
                log.error("dataset %s could not be read: %s", dataset_id, exc)
                raise MarketDataError(f"dataset {dataset_id} could not be read from the store") from exc
        return self._loaded[dataset_id]

    def _snapshot(self, symbols: list[str], period: str) -> list[tuple[str, Dataset]]:
        valid = [s for s in symbols if SYMBOL.fullmatch(s)]
        for symbol in symbols:
            if symbol not in valid:
                self.unavailable.setdefault(symbol, "not a ticker symbol")
        if not valid:
            raise MarketDataError("no valid ticker symbols")
        if len(valid) > MAX_SYMBOLS:
            raise MarketDataError(f"at most {MAX_SYMBOLS} symbols can be fetched at once")
        end = self._clock().astimezone(dt.timezone.utc).date() + dt.timedelta(days=1)
        # A week more than the period, so the trailing window (which ends at
        # the data's last session, not at today) is always covered. The window
        # ends at a holdout start if one comes first: the provider's families
        # are only known once it answers, so the earliest start is used, and the
        # snapshot still reaches today so that it stays reusable once the
        # holdout is rolled forward.
        horizon = min([end, *self.holdouts.values()])
        start = period_start(horizon, period) - dt.timedelta(days=_FETCH_MARGIN_DAYS)
        request = SnapshotRequest(tuple(valid), start, end)
        try:
            snapshot = (self._provider or default_provider()).snapshot(request)
        except ProviderError as exc:
            raise MarketDataError(str(exc)) from exc
        stored = [self._store.put(dataset) for dataset in snapshot.datasets]
        self.snapshots.append({
            "provider": snapshot.provider, "provider_client": snapshot.client,
            "retrieved_at": snapshot.retrieved_at.isoformat(), "request": request.as_dict(),
            "stored": [{"dataset_id": s.dataset_id, "sha256": s.sha256} for s in stored],
            "unavailable": dict(snapshot.unavailable)})
        for symbol, why in snapshot.unavailable.items():
            self.unavailable.setdefault(symbol, why)
        return [(s.dataset_id, d) for s, d in zip(stored, snapshot.datasets)]

    def record(self) -> dict:
        """What the task's result says about its data, before the worker
        registers ``snapshots`` (dataset_registry.record_market_data).
        ``holdouts`` names the start of each holdout that cut a dataset used."""
        families = {self._families[dataset_id] for dataset_id in self.used}
        return {"source": "datasets" if self.dataset_ids is not None else "snapshot",
                "datasets": sorted(self.used), "unavailable": dict(sorted(self.unavailable.items())),
                "snapshots": self.snapshots,
                "holdouts": {family: start.isoformat() for family, start in sorted(self.holdouts.items())
                             if family in families}}


_active: contextvars.ContextVar[MarketData | None] = contextvars.ContextVar("qs_market_data", default=None)
_sealed: contextvars.ContextVar[dict | None] = contextvars.ContextVar("qs_holdouts", default=None)


@contextlib.contextmanager
def holdouts(starts: dict[str, dt.date] | None):
    """Apply the holdout ``starts`` ({family: date}) to every ``session()``
    opened inside; the research worker wraps each job in this."""
    token = _sealed.set(_checked_holdouts(starts))
    try:
        yield
    finally:
        _sealed.reset(token)


@contextlib.contextmanager
def session(dataset_ids: list[str] | None = None, **options):
    """Make ``download`` serve one research task; yields its MarketData."""
    options.setdefault("holdouts", _sealed.get())
    data = MarketData(dataset_ids, **options)
    token = _active.set(data)
    try:
        yield data
    finally:
        _active.reset(token)


def download(tickers: list[str] | str, period: str) -> pd.DataFrame:
    """yf.download(tickers, period=period, auto_adjust=True), from datasets."""
    data = _active.get()
    if data is None:
        raise RuntimeError("market data is only available inside market_data.session()")
    return data.download(tickers, period)
