"""QuantumSentinel — dataset tables, their logical hash and their file format.

A research result can only be reproduced if the data it used is frozen and
named by its content. This module defines both:

* A ``Table`` is typed, column-oriented data sorted by its primary key.
  ``BARS_V1`` (daily OHLCV keyed by symbol and session date),
  ``CORPORATE_ACTIONS_V1`` and ``INSTRUMENTS_V1`` are the canonical
  market-data schemas.
* ``qs-col-sha256-v1`` hashes a table from its values, never from a file, so
  the same data written by another library version, or later by the Rust
  core, keeps its name. tests/fixtures/qs_col_sha256_v1.json holds golden
  vectors for other implementations.
* A ``Dataset`` is a set of tables plus the identity facts that change what
  its numbers mean (provider, licence, adjustment policy, session boundary).
  Its id ``qs-ds-sha256-v1:<hex>`` hashes both, so snapshots that differ
  only in adjustment policy get different ids.
* ``encode_blob``/``decode_blob`` store a dataset as one file. Decoding
  rehashes the content and refuses a file that does not match its id.

The encoding of each row (integers little-endian):

* ``float64``: IEEE 754 binary64, with -0.0 written as 0.0. NaN marks a null
  in a nullable column and is an error elsewhere; infinities are errors.
* ``date32``: int32 days since 1970-01-01.
* ``timestamp_ns``: int64 nanoseconds since 1970-01-01 UTC.
* ``utf8``: u64 byte length, then the UTF-8 bytes.
* A nullable column prefixes every row with a validity byte: 0x01 and the
  value, or 0x00 and the type's zero value (zero bytes; a zero length for
  utf8). arrow-digest writes a null as a lone zero byte, which is not
  injective: eight nulls then 0.0 encode like 0.0 then eight nulls.

A column's digest is the SHA-256 of its encoded rows. A table's hash is the
SHA-256 of its canonical JSON manifest ``{"columns": [{"name", "nullable",
"sha256", "type"}, ...], "key": [...], "rows": n, "scheme":
"qs-col-sha256-v1"}``, columns in schema order. A dataset's id is the SHA-256
of ``{"identity": {...}, "scheme": "qs-ds-sha256-v1", "tables": {name: table
hash}}``. Canonical JSON is RFC 8785 (JCS) restricted to the values on which
Python's json module emits the same bytes: ASCII keys, strings, integers
within +/-(2**53 - 1), booleans, null, lists and objects, and no floats.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import struct
import zlib
from dataclasses import dataclass, fields
from typing import Iterator, Mapping, Sequence

import numpy as np

TABLE_SCHEME = "qs-col-sha256-v1"
DATASET_SCHEME = "qs-ds-sha256-v1"
BLOB_FORMAT = "qs-blob-v1"
DATASET_ID = re.compile(rf"{DATASET_SCHEME}:[0-9a-f]{{64}}")
TYPES = ("utf8", "date32", "timestamp_ns", "float64")

_MAX_SAFE_INTEGER = 2**53 - 1
_NAME = re.compile(r"[a-z][a-z0-9_]{0,62}")
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+-]{0,63}")
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_U64 = struct.Struct("<Q")
_INT32 = (-(2**31), 2**31 - 1)
_UTF8_BATCH = 65_536
# Fixed-width encodings: the value, and the record of a nullable column.
_WIDE = {"float64": np.dtype("<f8"), "date32": np.dtype("<i4"), "timestamp_ns": np.dtype("<i8")}
_NULLABLE = {t: np.dtype([("valid", "u1"), ("value", d)]) for t, d in _WIDE.items()}

_BLOB_MAGIC = b"QSBLOB\x00\x01"
_MAX_HEADER = 1 << 20
DEFAULT_MAX_BYTES = 2 << 30


class DatasetError(ValueError):
    """Values that cannot form a table or dataset."""


class DatasetIntegrityError(Exception):
    """A stored dataset whose bytes do not match its id."""


def canonical_json(value) -> bytes:
    """RFC 8785 bytes of ``value``, restricted as the module docstring says."""
    _check_canonical(value)
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except UnicodeEncodeError as exc:  # a lone surrogate
        raise DatasetError("canonical JSON strings must be valid Unicode") from exc


def _check_canonical(value) -> None:
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, int):
        if abs(value) > _MAX_SAFE_INTEGER:
            raise DatasetError("canonical JSON integers must be within +/-(2**53 - 1)")
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _check_canonical(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or not key.isascii():
                raise DatasetError("canonical JSON keys must be ASCII strings")
            _check_canonical(item)
        return
    raise DatasetError(f"canonical JSON cannot hold {type(value).__name__} values")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Column:
    name: str
    type: str
    nullable: bool = False


@dataclass(frozen=True)
class TableSchema:
    name: str
    columns: tuple[Column, ...]
    key: tuple[str, ...]

    def __post_init__(self):
        names = [c.name for c in self.columns]
        if not _NAME.fullmatch(self.name) or not all(_NAME.fullmatch(n) for n in names):
            raise DatasetError("table and column names must match [a-z][a-z0-9_]*")
        if len(set(names)) != len(names) or not names:
            raise DatasetError(f"table {self.name} needs distinct column names")
        if any(c.type not in TYPES for c in self.columns):
            raise DatasetError(f"column types must be one of {TYPES}")
        nullable = {c.name for c in self.columns if c.nullable}
        if not self.key or len(set(self.key)) != len(self.key) \
                or any(k not in names or k in nullable for k in self.key):
            raise DatasetError(f"table {self.name} needs a key of distinct non-nullable columns")

    def column(self, name: str) -> Column:
        return next(c for c in self.columns if c.name == name)


# Daily bars, adjusted as the dataset identity's ``adjustment`` says ("raw"
# for the prints as they traded). session_date is the exchange-local trading
# day, or for 24/7 markets the day ending at the dataset's declared
# session_boundary; a bar exists only for a session with a close.
BARS_V1 = TableSchema("bars", (
    Column("symbol", "utf8"),
    Column("session_date", "date32"),
    Column("open", "float64", nullable=True),
    Column("high", "float64", nullable=True),
    Column("low", "float64", nullable=True),
    Column("close", "float64"),
    Column("volume", "float64", nullable=True),
), key=("symbol", "session_date"))

# Splits and cash dividends. value is the split ratio (new shares per old
# share) or the cash amount per share; announced_at is when the action
# became public, where the provider knows it.
CORPORATE_ACTIONS_V1 = TableSchema("corporate_actions", (
    Column("symbol", "utf8"),
    Column("ex_date", "date32"),
    Column("action", "utf8"),
    Column("value", "float64"),
    Column("announced_at", "timestamp_ns", nullable=True),
    Column("source", "utf8"),
), key=("symbol", "ex_date", "action"))

# What each symbol is and where it trades: the provider's instrument type,
# exchange code, IANA timezone (the one session_date is local to) and the
# currency its prices are quoted in, exactly as quoted (GBp is pence).
INSTRUMENTS_V1 = TableSchema("instruments", (
    Column("symbol", "utf8"),
    Column("instrument_type", "utf8"),
    Column("exchange", "utf8"),
    Column("timezone", "utf8"),
    Column("currency", "utf8"),
), key=("symbol",))


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

def _is_number(value) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) \
        and not isinstance(value, (bool, np.bool_))


def _float64(column: Column, values) -> np.ndarray:
    # numpy would parse "1.5" and True as numbers; a provider bug should not.
    # A plain sequence is checked item by item: np.asarray([1.0, True])
    # is already a float array.
    raw = np.asarray(values) if hasattr(values, "dtype") else np.asarray(values, dtype=object)
    if raw.dtype.kind == "O":
        if not all(v is None or _is_number(v) for v in raw.reshape(-1)):
            raise DatasetError(f"{column.name}: values must be numbers")
    elif raw.dtype.kind not in "iuf":
        raise DatasetError(f"{column.name}: values must be numbers")
    try:
        array = np.array(raw, dtype=float).reshape(-1)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DatasetError(f"{column.name}: values must be numbers") from exc
    if np.isinf(array).any():
        raise DatasetError(f"{column.name}: values must be finite")
    if not column.nullable and np.isnan(array).any():
        raise DatasetError(f"{column.name}: missing values in a non-nullable column")
    return np.where(array == 0.0, 0.0, array)  # -0.0 -> 0.0


def _date32(column: Column, values) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind == "M":
        days = array.astype("datetime64[D]")
        timed = ~np.isnat(array) & (days.astype(array.dtype) != array)
        if timed.any():
            # A timezone-aware midnight converted to UTC lands here, on the
            # previous day's evening: refuse it rather than drop the time.
            raise DatasetError(f"{column.name}: dates must be whole days, not times")
    elif array.dtype.kind in "OU" or array.size == 0:
        days = np.array([_one_date(column, v) for v in array.reshape(-1)], dtype="datetime64[D]")
    else:
        raise DatasetError(f"{column.name}: dates must be datetime64, datetime.date or YYYY-MM-DD")
    days = days.reshape(-1)
    present = ~np.isnat(days)
    numbers = days[present].astype("int64")
    if numbers.size and (numbers.min() < _INT32[0] or numbers.max() > _INT32[1]):
        raise DatasetError(f"{column.name}: date out of range")
    return days


def _one_date(column: Column, value) -> np.datetime64:
    if value is None:
        return np.datetime64("NaT", "D")
    if isinstance(value, dt.datetime):
        raise DatasetError(f"{column.name}: pass a date, not a datetime")
    if isinstance(value, dt.date):
        return np.datetime64(value, "D")
    if isinstance(value, str) and _ISO_DATE.fullmatch(value):
        try:
            return np.datetime64(dt.date.fromisoformat(value), "D")
        except ValueError:
            pass
    raise DatasetError(f"{column.name}: {value!r} is not a date")


def _timestamp_ns(column: Column, values) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind == "M":
        # Outside 1677..2262 the cast to ns raises (numpy >= 2) or wraps.
        try:
            stamps = array.astype("datetime64[ns]")
            wrapped = (~np.isnat(array) & (stamps.astype(array.dtype) != array)).any()
        except OverflowError:
            wrapped = True
        if wrapped:
            raise DatasetError(f"{column.name}: timestamp out of range")
    elif array.dtype.kind == "O" or array.size == 0:
        stamps = np.array([_one_timestamp(column, v) for v in array.reshape(-1)],
                          dtype="datetime64[ns]")
    else:
        raise DatasetError(f"{column.name}: timestamps must be datetime64 or timezone-aware datetimes")
    return stamps.reshape(-1)


def _one_timestamp(column: Column, value) -> np.datetime64:
    if value is None:
        return np.datetime64("NaT", "ns")
    if not isinstance(value, dt.datetime) or value.utcoffset() is None:
        raise DatasetError(f"{column.name}: timestamps must be timezone-aware datetimes")
    delta = value - dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
    ns = (delta.days * 86_400 + delta.seconds) * 10**9 + delta.microseconds * 1000
    ns += getattr(value, "nanosecond", 0)  # pandas.Timestamp
    if not -(2**63) < ns < 2**63:
        raise DatasetError(f"{column.name}: timestamp out of range")
    return np.datetime64(ns, "ns")


def _utf8(column: Column, values) -> np.ndarray:
    items = list(np.asarray(values, dtype=object).reshape(-1))
    for item in items:
        if item is None and column.nullable:
            continue
        if not isinstance(item, str):
            raise DatasetError(f"{column.name}: values must be strings")
        try:
            item.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise DatasetError(f"{column.name}: strings must be valid Unicode") from exc
    array = np.empty(len(items), dtype=object)
    array[:] = items
    return array


_NORMALISE = {"float64": _float64, "date32": _date32, "timestamp_ns": _timestamp_ns, "utf8": _utf8}


def _missing(column: Column, array: np.ndarray) -> np.ndarray:
    if column.type == "float64":
        return np.isnan(array)
    if column.type == "utf8":
        return np.array([v is None for v in array], dtype=bool)
    return np.isnat(array)


def _sort_codes(column: Column, array: np.ndarray) -> np.ndarray:
    if column.type == "utf8":
        # Python orders str by code point, which is UTF-8 byte order.
        return np.unique(array, return_inverse=True)[1].reshape(-1)
    return array.view("int64")


@dataclass(frozen=True, eq=False)
class Table:
    """Rows of ``schema`` in key order; build one with ``Table.build``."""
    schema: TableSchema
    columns: Mapping[str, np.ndarray]

    @classmethod
    def build(cls, schema: TableSchema, data: Mapping[str, Sequence]) -> "Table":
        names = [c.name for c in schema.columns]
        if set(data) != set(names):
            raise DatasetError(f"table {schema.name} needs exactly the columns {names}")
        arrays = {c.name: _NORMALISE[c.type](c, data[c.name]) for c in schema.columns}
        lengths = {len(a) for a in arrays.values()}
        if len(lengths) != 1:
            raise DatasetError(f"table {schema.name}: columns have different lengths")
        for c in schema.columns:
            if not c.nullable and _missing(c, arrays[c.name]).any():
                raise DatasetError(f"{c.name}: missing values in a non-nullable column")
        keys = [_sort_codes(schema.column(k), arrays[k]) for k in schema.key]
        order = np.lexsort(keys[::-1]) if lengths != {0} else np.arange(0)
        keys = [k[order] for k in keys]
        if len(order) > 1:
            same = np.ones(len(order) - 1, dtype=bool)
            for k in keys:
                same &= k[1:] == k[:-1]
            if same.any():
                raise DatasetError(f"table {schema.name}: duplicate key {list(schema.key)}")
        columns = {}
        for name, array in arrays.items():
            array = array[order]
            array.setflags(write=False)
            columns[name] = array
        return cls(schema, columns)

    def __len__(self) -> int:
        return len(next(iter(self.columns.values())))

    def encoded(self, name: str) -> Iterator[bytes]:
        """The ``qs-col-sha256-v1`` encoding of column ``name``, in chunks."""
        column = self.schema.column(name)
        array = self.columns[name]
        missing = _missing(column, array)
        if column.type == "utf8":
            flag = b"\x01" if column.nullable else b""
            for start in range(0, len(array), _UTF8_BATCH):
                rows = []
                for value in array[start:start + _UTF8_BATCH]:
                    if value is None:
                        rows.append(b"\x00" + _U64.pack(0))
                    else:
                        raw = value.encode("utf-8")
                        rows.append(flag + _U64.pack(len(raw)) + raw)
                yield b"".join(rows)
            return
        if column.type == "float64":
            values = np.where(missing, 0.0, array)
        else:
            values = np.where(missing, 0, array.view("int64"))
        values = values.astype(_WIDE[column.type])
        if not column.nullable:
            yield values.tobytes()
            return
        records = np.zeros(len(values), dtype=_NULLABLE[column.type])
        records["valid"] = ~missing
        records["value"] = values
        yield records.tobytes()

    def column_digest(self, name: str) -> str:
        digest = hashlib.sha256()
        for chunk in self.encoded(name):
            digest.update(chunk)
        return digest.hexdigest()

    def manifest(self) -> dict:
        return {"columns": [{"name": c.name, "nullable": c.nullable,
                             "sha256": self.column_digest(c.name), "type": c.type}
                            for c in self.schema.columns],
                "key": list(self.schema.key), "rows": len(self), "scheme": TABLE_SCHEME}

    def table_hash(self) -> str:
        return _sha256(canonical_json(self.manifest()))


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetIdentity:
    """Facts that change what a dataset's numbers mean; part of its id."""
    family: str            # e.g. US_EQUITY_EOD, CRYPTO_SPOT_OHLCV
    provider: str          # who supplied it; series from different providers never share an id
    licence_id: str        # the terms it was obtained under
    frequency: str         # bar length, e.g. 1d
    session_boundary: str  # exchange-local, or a declared cut-off such as utc-00:00
    adjustment: str        # raw, split or split_dividend
    survivorship: str      # current_listings (survivor-biased) or point_in_time

    def __post_init__(self):
        for f in fields(self):
            if not isinstance(getattr(self, f.name), str) or not _TOKEN.fullmatch(getattr(self, f.name)):
                raise DatasetError(f"identity {f.name} must match {_TOKEN.pattern}")

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(frozen=True, eq=False)
class Dataset:
    identity: DatasetIdentity
    tables: tuple[Table, ...]

    def __post_init__(self):
        names = [t.schema.name for t in self.tables]
        if not names or len(set(names)) != len(names):
            raise DatasetError("a dataset needs one or more tables with distinct names")

    def table(self, name: str) -> Table:
        for table in self.tables:
            if table.schema.name == name:
                return table
        raise KeyError(name)

    def manifest(self) -> dict:
        return {"identity": self.identity.as_dict(), "scheme": DATASET_SCHEME,
                "tables": {t.schema.name: t.table_hash() for t in self.tables}}

    @property
    def dataset_id(self) -> str:
        return f"{DATASET_SCHEME}:{_sha256(canonical_json(self.manifest()))}"


# --------------------------------------------------------------------------
# Blob format
# --------------------------------------------------------------------------
#
#   magic  b"QSBLOB\x00\x01"
#   u64    header length
#   header canonical JSON: format, dataset_id, identity, payload length and,
#          per table, its name, key, row count and columns with their
#          encoded lengths
#   zlib   the encoded columns, table after table in header order
#
# The payload is exactly the bytes the logical hash covers, so a decoded
# blob is rebuilt into a Table and rehashed: any byte that disagrees with
# the expected id fails the read.

def encode_blob(dataset: Dataset) -> bytes:
    compressor = zlib.compressobj(6)
    parts, tables, total = [], [], 0
    for table in dataset.tables:
        columns = []
        for c in table.schema.columns:
            length = 0
            for chunk in table.encoded(c.name):
                length += len(chunk)
                parts.append(compressor.compress(chunk))
            total += length
            columns.append({"length": length, "name": c.name, "nullable": c.nullable, "type": c.type})
        tables.append({"columns": columns, "key": list(table.schema.key),
                       "name": table.schema.name, "rows": len(table)})
    parts.append(compressor.flush())
    header = canonical_json({"dataset_id": dataset.dataset_id, "format": BLOB_FORMAT,
                             "identity": dataset.identity.as_dict(), "payload_length": total,
                             "tables": tables})
    return _BLOB_MAGIC + _U64.pack(len(header)) + header + b"".join(parts)


def decode_blob(blob: bytes, expected_id: str, max_bytes: int = DEFAULT_MAX_BYTES) -> Dataset:
    """The dataset in ``blob``, if and only if its content hashes to ``expected_id``."""
    try:
        dataset = _decode(blob, max_bytes)
    except DatasetIntegrityError:
        raise
    except (DatasetError, ValueError, KeyError, TypeError, IndexError, AttributeError,
            RecursionError, zlib.error, struct.error) as exc:
        raise DatasetIntegrityError(f"malformed dataset blob: {exc}") from exc
    if dataset.dataset_id != expected_id:
        raise DatasetIntegrityError("dataset blob content does not match its id")
    return dataset


def _decode(blob: bytes, max_bytes: int) -> Dataset:
    view = memoryview(blob)
    if bytes(view[:8]) != _BLOB_MAGIC or len(view) < 16:
        raise DatasetIntegrityError("not a qs-blob-v1 file")
    (header_length,) = _U64.unpack(view[8:16])
    if header_length > _MAX_HEADER or 16 + header_length > len(view):
        raise DatasetIntegrityError("dataset blob header is truncated or too large")
    header = json.loads(bytes(view[16:16 + header_length]))
    if header.get("format") != BLOB_FORMAT:
        raise DatasetIntegrityError("not a qs-blob-v1 file")
    total = header["payload_length"]
    if not isinstance(total, int) or not 0 <= total <= max_bytes:
        raise DatasetIntegrityError("dataset payload is larger than the read limit")
    inflater = zlib.decompressobj()
    payload = inflater.decompress(view[16 + header_length:], total + 1)
    if len(payload) != total or not inflater.eof or inflater.unconsumed_tail or inflater.unused_data:
        raise DatasetIntegrityError("dataset payload length does not match its header")
    offset, tables = 0, []
    for spec in header["tables"]:
        schema = TableSchema(spec["name"], tuple(Column(c["name"], c["type"], c["nullable"])
                                                 for c in spec["columns"]), tuple(spec["key"]))
        rows, data = spec["rows"], {}
        for c, column in zip(spec["columns"], schema.columns):
            length = c["length"]
            if not isinstance(length, int) or length < 0 or offset + length > total:
                raise DatasetIntegrityError("dataset column overruns the payload")
            data[column.name] = _parse(column, payload[offset:offset + length], rows)
            offset += length
        tables.append(Table.build(schema, data))
    if offset != total:
        raise DatasetIntegrityError("dataset payload has trailing bytes")
    return Dataset(DatasetIdentity(**header["identity"]), tuple(tables))


def _parse(column: Column, raw: bytes, rows: int):
    if not isinstance(rows, int) or rows < 0:
        raise DatasetIntegrityError("bad row count")
    if column.type == "utf8":
        return _parse_utf8(column, raw, rows)
    dtype = _NULLABLE[column.type] if column.nullable else _WIDE[column.type]
    if len(raw) != rows * dtype.itemsize:
        raise DatasetIntegrityError(f"{column.name}: length does not match its rows")
    records = np.frombuffer(raw, dtype=dtype)
    values = records["value"] if column.nullable else records
    missing = np.zeros(rows, dtype=bool)
    if column.nullable:
        if (records["valid"] > 1).any():
            raise DatasetIntegrityError(f"{column.name}: validity bytes must be 0 or 1")
        missing = records["valid"] == 0
    if column.type == "float64":
        return np.where(missing, np.nan, values.astype(float))
    unit = "D" if column.type == "date32" else "ns"
    stamps = values.astype("int64").view(f"datetime64[{unit}]").copy()
    stamps[missing] = np.datetime64("NaT", unit)
    return stamps


def _parse_utf8(column: Column, raw: bytes, rows: int) -> list:
    values, offset, view = [], 0, memoryview(raw)
    for _ in range(rows):
        valid = True
        if column.nullable:
            if view[offset] > 1:
                raise DatasetIntegrityError(f"{column.name}: validity bytes must be 0 or 1")
            valid = view[offset] == 1
            offset += 1
        (length,) = _U64.unpack(view[offset:offset + 8])
        offset += 8
        if offset + length > len(view):
            raise DatasetIntegrityError(f"{column.name}: string overruns the column")
        values.append(str(view[offset:offset + length], "utf-8") if valid else None)
        offset += length
    if offset != len(view):
        raise DatasetIntegrityError(f"{column.name}: length does not match its rows")
    return values
