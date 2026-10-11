"""QuantumSentinel — market-data providers that produce frozen datasets.

A provider turns a request (symbols and a date range) into datasets
(dataset_format.Dataset) whose identity states who supplied the data, under
which licence, with which adjustment policy and session boundary. Symbols of
different families (equities, crypto, ...) become separate datasets: their
sessions are defined differently, and the plan's rule is one family per
dataset.

``YahooProvider`` (yfinance) is the only provider for now. It is a
personal-use prototype source: the yfinance README says Yahoo's API "is
intended for personal use only", and Yahoo's terms prohibit automated
collection and commercial reuse. Its datasets carry licence
``yahoo-personal-use`` so they can be found and quarantined when a licensed
vendor replaces it.

What Yahoo serves, measured with yfinance 1.5.2 (2026-10-11):

* No raw prices. "Open"/"High"/"Low"/"Close"/"Volume" are split-adjusted as
  of the fetch (AAPL's 2020-08-28 close is 124.8075, the $499.23 print / 4),
  and so are its dividends. The identity says ``adjustment="split"``; a split
  announced later changes the data and therefore the dataset id.
* "Adj Close" also adjusts for dividends. It is not stored: it is
  reproducible from the closes and dividends here (the backward factor
  prod(1 - dividend / previous close) matched it within 7e-7 relative over
  ten years of AAPL, MSFT, SPY, KO and T).
* Daily bars are dated at local midnight of the exchange's timezone; crypto
  bars cover UTC days (each close equals the next open, and they match
  UTC-day aggregates of hourly bars). Crypto prices are CoinMarketCap's
  cross-venue composite (Yahoo's exchange code CCC), not one venue's.
* Continuous futures (GC=F) roll between contracts by Yahoo's own,
  unpublished rule, so their family says CONTINUOUS.
"""
from __future__ import annotations

import datetime as dt
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Callable, Mapping, Protocol
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np

from .dataset_format import (BARS_V1, CORPORATE_ACTIONS_V1, INSTRUMENTS_V1, Dataset, DatasetError,
                             DatasetIdentity, Table)

log = logging.getLogger(__name__)

SYMBOL = re.compile(r"[A-Z0-9^][A-Z0-9.=^-]{0,19}")
MAX_SYMBOLS = 50


class ProviderError(Exception):
    """The provider could not produce any dataset for a request."""


@dataclass(frozen=True)
class Licence:
    licence_id: str
    personal_use_only: bool
    redistribution_allowed: bool
    terms_url: str


LICENCES: dict[str, Licence] = {
    "yahoo-personal-use": Licence(
        "yahoo-personal-use", personal_use_only=True, redistribution_allowed=False,
        terms_url="https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html"),
}


@dataclass(frozen=True)
class SnapshotRequest:
    """Daily bars for ``symbols`` from ``start`` (inclusive) to ``end``
    (exclusive). Sessions on or after the retrieval date are never included:
    the provider may still be revising them."""
    symbols: tuple[str, ...]
    start: dt.date
    end: dt.date

    def __post_init__(self):
        symbols = tuple(self.symbols)
        if not symbols or len(symbols) > MAX_SYMBOLS:
            raise ValueError(f"a snapshot needs 1 to {MAX_SYMBOLS} symbols")
        if len(set(symbols)) != len(symbols):
            raise ValueError("symbols must be distinct")
        bad = [s for s in symbols if not isinstance(s, str) or not SYMBOL.fullmatch(s)]
        if bad:
            raise ValueError(f"not a symbol: {bad[0]!r}")
        if not (isinstance(self.start, dt.date) and isinstance(self.end, dt.date)) \
                or isinstance(self.start, dt.datetime) or isinstance(self.end, dt.datetime):
            raise ValueError("start and end must be dates")
        if self.start >= self.end:
            raise ValueError("start must be before end")
        object.__setattr__(self, "symbols", tuple(sorted(symbols)))

    def as_dict(self) -> dict:
        return {"symbols": list(self.symbols), "start": self.start.isoformat(),
                "end": self.end.isoformat()}


@dataclass(frozen=True)
class Snapshot:
    request: SnapshotRequest
    provider: str
    client: str                      # the library and version that fetched it
    retrieved_at: dt.datetime        # UTC
    datasets: tuple[Dataset, ...]
    unavailable: Mapping[str, str] = field(default_factory=dict)  # symbol -> reason


class Provider(Protocol):
    name: str
    licence: Licence

    def snapshot(self, request: SnapshotRequest) -> Snapshot: ...


# --------------------------------------------------------------------------
# Yahoo (yfinance)
# --------------------------------------------------------------------------

# Yahoo instrumentType -> (instruments.instrument_type, dataset family)
_YAHOO_TYPES = {
    "EQUITY": ("equity", "EQUITY_EOD"),
    "ETF": ("etf", "EQUITY_EOD"),
    "MUTUALFUND": ("fund", "FUND_NAV_EOD"),
    "INDEX": ("index", "INDEX_EOD"),
    "CRYPTOCURRENCY": ("crypto", "CRYPTO_SPOT_EOD"),
    "CURRENCY": ("fx", "FX_SPOT_EOD"),
    "FUTURE": ("future", "FUTURES_CONTINUOUS_EOD"),
}
_ACTIONS = (("Dividends", "dividend"), ("Stock Splits", "split"), ("Capital Gains", "capital_gain"))

Fetch = Callable[[str, dt.date, dt.date], tuple[object, Mapping]]


def _yahoo_fetch(symbol: str, start: dt.date, end: dt.date):
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    try:
        frame = ticker.history(start=start.isoformat(), end=end.isoformat(), interval="1d",
                               auto_adjust=False, actions=True)
        if frame is None or frame.empty:
            # Asking for metadata now refetches and, for an unknown symbol,
            # raises KeyError inside yfinance 1.5.2.
            return frame, {}
        return frame, ticker.get_history_metadata()
    except Exception as exc:
        # One symbol's failure (network, rate limit, a yfinance bug) leaves
        # the others usable; the reason stays in the worker log.
        log.warning("yahoo fetch of %s failed: %r", symbol, exc)
        raise ProviderError(f"the provider failed ({type(exc).__name__})") from exc


def _yahoo_client() -> str:
    import yfinance as yf

    return f"yfinance {yf.__version__}"


@dataclass(frozen=True)
class _Symbol:
    symbol: str
    family: str
    session_boundary: str
    instrument: dict
    bars: dict
    actions: dict


class YahooProvider:
    name = "yahoo"
    licence = LICENCES["yahoo-personal-use"]

    def __init__(self, fetch: Fetch | None = None, client: str | None = None,
                 clock: Callable[[], dt.datetime] | None = None):
        self._fetch = fetch or _yahoo_fetch
        self._client = client
        self._clock = clock or (lambda: dt.datetime.now(dt.timezone.utc))

    def snapshot(self, request: SnapshotRequest) -> Snapshot:
        retrieved_at = self._clock().astimezone(dt.timezone.utc)
        parsed, unavailable = [], {}
        for symbol in request.symbols:
            try:
                frame, metadata = self._fetch(symbol, request.start, request.end)
                parsed.append(_parse_yahoo(symbol, frame, metadata, retrieved_at))
            except ProviderError as exc:
                unavailable[symbol] = str(exc)
        if not parsed:
            raise ProviderError("no data for any requested symbol: "
                                + "; ".join(f"{s}: {r}" for s, r in unavailable.items()))
        groups: dict[tuple[str, str], list[_Symbol]] = {}
        for item in parsed:
            groups.setdefault((item.family, item.session_boundary), []).append(item)
        datasets = tuple(self._dataset(family, boundary, items)
                         for (family, boundary), items in sorted(groups.items()))
        return Snapshot(request, self.name, self._client or _yahoo_client(), retrieved_at,
                        datasets, unavailable)

    def _dataset(self, family: str, boundary: str, items: list[_Symbol]) -> Dataset:
        identity = DatasetIdentity(family=family, provider=self.name,
                                   licence_id=self.licence.licence_id, frequency="1d",
                                   session_boundary=boundary, adjustment="split",
                                   survivorship="current_listings")

        def merged(part: str, schema) -> Table:
            data = {c.name: [] for c in schema.columns}
            for item in items:
                for name, values in getattr(item, part).items():
                    data[name].extend(values)
            return Table.build(schema, data)

        return Dataset(identity, (merged("bars", BARS_V1), merged("actions", CORPORATE_ACTIONS_V1),
                                  merged("instrument", INSTRUMENTS_V1)))


def _parse_yahoo(symbol: str, frame, metadata: Mapping, retrieved_at: dt.datetime) -> _Symbol:
    """One symbol's bars, corporate actions and instrument row."""
    if frame is None or len(frame) == 0:
        raise ProviderError("no data returned")
    metadata = metadata or {}
    yahoo_type = metadata.get("instrumentType")
    if yahoo_type not in _YAHOO_TYPES:
        raise ProviderError(f"unsupported instrument type {yahoo_type!r}")
    instrument_type, family = _YAHOO_TYPES[yahoo_type]
    timezone, exchange, currency = (metadata.get("exchangeTimezoneName"), metadata.get("exchangeName"),
                                    metadata.get("currency"))
    if not all(isinstance(v, str) and v for v in (timezone, exchange, currency)):
        raise ProviderError("the provider did not say where the symbol trades")
    try:
        zone = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ProviderError(f"unknown timezone {timezone!r}") from exc
    if instrument_type == "crypto":
        if timezone != "UTC":
            raise ProviderError(f"crypto bars in {timezone}, expected UTC days")
        boundary = "utc-00:00"
    else:
        boundary = "exchange-local"

    index = getattr(frame, "index", None)
    if getattr(index, "tz", None) is None:
        raise ProviderError("bar timestamps carry no timezone")
    local = index.tz_convert(zone)
    if (local.hour != 0).any() or (local.minute != 0).any() or (local.second != 0).any() \
            or (local.nanosecond != 0).any() or (local.microsecond != 0).any():
        raise ProviderError("bar timestamps are not local midnights")
    sessions = local.tz_localize(None).to_numpy().astype("datetime64[D]")
    today = np.datetime64(retrieved_at.astimezone(zone).date(), "D")
    close = _column(frame, "Close")
    # A session still trading (or not yet settled) on the retrieval date is
    # left out, and so is a row without a close.
    keep = (sessions < today) & np.isfinite(close)
    if not keep.any():
        raise ProviderError("no completed sessions in the requested range")
    sessions = sessions[keep]

    def prices(name: str) -> list:
        return [None if not math.isfinite(v) else float(v) for v in _column(frame, name)[keep]]

    volume = [None] * int(keep.sum()) if instrument_type == "fx" else prices("Volume")  # Yahoo has no FX volume
    bars = {"symbol": [symbol] * len(sessions), "session_date": sessions, "open": prices("Open"),
            "high": prices("High"), "low": prices("Low"), "close": close[keep].tolist(), "volume": volume}
    actions = {c.name: [] for c in CORPORATE_ACTIONS_V1.columns}
    for column, action in _ACTIONS:
        if column not in frame.columns:
            continue
        values = _column(frame, column)[keep]
        for session, value in zip(sessions, values):
            if math.isfinite(value) and value != 0.0:
                actions["symbol"].append(symbol)
                actions["ex_date"].append(session)
                actions["action"].append(action)
                actions["value"].append(float(value))
                actions["announced_at"].append(None)  # Yahoo does not say
                actions["source"].append("yahoo")
    actions["ex_date"] = np.array(actions["ex_date"], dtype="datetime64[D]")
    instrument = {"symbol": [symbol], "instrument_type": [instrument_type], "exchange": [exchange],
                  "timezone": [timezone], "currency": [currency]}
    try:  # validate this symbol alone, so one bad symbol fails only itself
        Table.build(BARS_V1, bars)
        Table.build(CORPORATE_ACTIONS_V1, actions)
        Table.build(INSTRUMENTS_V1, instrument)
    except DatasetError as exc:
        raise ProviderError(f"unusable data: {exc}") from exc
    return _Symbol(symbol, family, boundary, instrument, bars, actions)


def _column(frame, name: str) -> np.ndarray:
    if name not in frame.columns:
        raise ProviderError(f"no {name} column")
    try:
        return frame[name].to_numpy(dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ProviderError(f"{name} is not numeric") from exc
