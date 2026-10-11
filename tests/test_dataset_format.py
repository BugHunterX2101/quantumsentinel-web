"""Dataset tables, the qs-col-sha256-v1 logical hash and the qs-blob-v1 format.

The golden vectors in fixtures/qs_col_sha256_v1.json are checked against two
implementations: the production one (numpy) and ``reference_*`` below, which
follows the module docstring with nothing but struct, hashlib and
hand-written manifest strings. Another implementation (the planned Rust
core) is correct when it reproduces the same file.
"""
import datetime as dt
import hashlib
import json
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from backend.services import dataset_format as fmt
from backend.services.dataset_format import (BARS_V1, CORPORATE_ACTIONS_V1, Column, Dataset,
                                             DatasetError, DatasetIdentity, DatasetIntegrityError,
                                             Table, TableSchema)

FIXTURE = Path(__file__).parent / "fixtures" / "qs_col_sha256_v1.json"
EPOCH = dt.date(1970, 1, 1)


# --------------------------------------------------------------------------
# Reference implementation (fixture value notation: float64 as float.hex(),
# date32 as YYYY-MM-DD, timestamp_ns as an integer, utf8 as a string, null)
# --------------------------------------------------------------------------

_ZERO = {"float64": bytes(8), "date32": bytes(4), "timestamp_ns": bytes(8), "utf8": bytes(8)}


def _reference_value(kind, value):
    if kind == "float64":
        number = float.fromhex(value)
        return struct.pack("<d", 0.0 if number == 0 else number)
    if kind == "date32":
        return struct.pack("<i", (dt.date.fromisoformat(value) - EPOCH).days)
    if kind == "timestamp_ns":
        return struct.pack("<q", value)
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _reference_sort_key(kind, value):
    if kind == "utf8":
        return value.encode("utf-8")
    if kind == "date32":
        return (dt.date.fromisoformat(value) - EPOCH).days
    return value


def reference_table(schema, rows):
    """(column digests, table hash) of ``rows`` (dicts in fixture notation)."""
    columns = schema["columns"]
    kinds = {c["name"]: c["type"] for c in columns}
    rows = sorted(rows, key=lambda r: tuple(_reference_sort_key(kinds[k], r[k]) for k in schema["key"]))
    digests = {}
    for c in columns:
        stream = b""
        for row in rows:
            value = row[c["name"]]
            if c["nullable"]:
                stream += b"\x00" + _ZERO[c["type"]] if value is None \
                    else b"\x01" + _reference_value(c["type"], value)
            else:
                stream += _reference_value(c["type"], value)
        digests[c["name"]] = hashlib.sha256(stream).hexdigest()
    manifest = ('{"columns":['
                + ",".join('{"name":"%s","nullable":%s,"sha256":"%s","type":"%s"}'
                           % (c["name"], "true" if c["nullable"] else "false", digests[c["name"]], c["type"])
                           for c in columns)
                + '],"key":[' + ",".join('"%s"' % k for k in schema["key"]) + "]"
                + ',"rows":%d,"scheme":"qs-col-sha256-v1"}' % len(rows))
    return digests, hashlib.sha256(manifest.encode()).hexdigest()


def reference_dataset_id(identity, table_hashes):
    manifest = ('{"identity":{' + ",".join('"%s":"%s"' % (k, identity[k]) for k in sorted(identity))
                + '},"scheme":"qs-ds-sha256-v1","tables":{'
                + ",".join('"%s":"%s"' % (n, table_hashes[n]) for n in sorted(table_hashes)) + "}}")
    return "qs-ds-sha256-v1:" + hashlib.sha256(manifest.encode()).hexdigest()


# --------------------------------------------------------------------------
# Production implementation fed from fixture notation
# --------------------------------------------------------------------------

def _schema(spec) -> TableSchema:
    return TableSchema(spec["name"], tuple(Column(c["name"], c["type"], c["nullable"])
                                           for c in spec["columns"]), tuple(spec["key"]))


def _production_values(kind, values):
    if kind == "float64":
        return [None if v is None else float.fromhex(v) for v in values]
    if kind == "timestamp_ns":
        return np.array([np.datetime64("NaT", "ns") if v is None else np.datetime64(v, "ns")
                         for v in values], dtype="datetime64[ns]")
    return list(values)


def production_table(spec, rows) -> Table:
    return Table.build(_schema(spec), {c["name"]: _production_values(c["type"], [r[c["name"]] for r in rows])
                                       for c in spec["columns"]})


@pytest.fixture(scope="module")
def golden():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_golden_vectors_match_the_reference_implementation(golden):
    assert golden["scheme"] == "qs-col-sha256-v1"
    for case in golden["tables"]:
        digests, table_hash = reference_table(case["schema"], case["rows"])
        assert digests == case["column_sha256"], case["name"]
        assert table_hash == case["table_sha256"], case["name"]
    hashes = {c["name"]: c["table_sha256"] for c in golden["tables"]}
    for case in golden["datasets"]:
        assert reference_dataset_id(case["identity"], {t: hashes[c] for t, c in case["tables"].items()}) \
            == case["dataset_id"], case["name"]


def test_golden_vectors_match_the_production_implementation(golden):
    tables = {}
    for case in golden["tables"]:
        table = production_table(case["schema"], case["rows"])
        assert {c.name: table.column_digest(c.name) for c in table.schema.columns} == case["column_sha256"]
        assert table.table_hash() == case["table_sha256"], case["name"]
        tables[case["name"]] = table
    for case in golden["datasets"]:
        dataset = Dataset(DatasetIdentity(**case["identity"]),
                          tuple(tables[c] for c in case["tables"].values()))
        assert [t.schema.name for t in dataset.tables] == list(case["tables"])
        assert dataset.dataset_id == case["dataset_id"], case["name"]


def test_an_empty_column_hashes_the_empty_string(golden):
    """Checkable by hand: no rows means no bytes, for every column."""
    empty = next(c for c in golden["tables"] if c["name"] == "bars_empty")
    assert set(empty["column_sha256"].values()) == {hashlib.sha256(b"").hexdigest()}


def test_nulls_cannot_collide_with_values(golden):
    """[null x8, 0.0] and [0.0, null x8] hash differently (a lone null byte would not)."""
    a, b = (next(c for c in golden["tables"] if c["name"] == n)
            for n in ("nullable_collision_a", "nullable_collision_b"))
    assert a["column_sha256"]["x"] != b["column_sha256"]["x"]

    def lone_null_byte(values):  # arrow-digest's encoding
        return b"".join(b"\x00" if v is None else struct.pack("<d", v) for v in values)
    assert lone_null_byte([None] * 8 + [0.0]) == lone_null_byte([0.0] + [None] * 8)


def test_a_null_string_is_not_the_empty_string(golden):
    a, b = (next(c for c in golden["tables"] if c["name"] == n)
            for n in ("nullable_utf8", "nullable_utf8_empty"))
    assert a["column_sha256"]["s"] != b["column_sha256"]["s"]


def test_dataset_id_depends_on_adjustment_policy(golden):
    ids = {c["name"]: c["dataset_id"] for c in golden["datasets"]}
    assert ids["eod_raw"] != ids["eod_split_adjusted"]


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

IDENTITY = DatasetIdentity(family="US_EQUITY_EOD", provider="yfinance", licence_id="yahoo-personal-use",
                           frequency="1d", session_boundary="exchange-local", adjustment="raw",
                           survivorship="current_listings")


def bars(symbols=("AAPL", "AAPL", "MSFT"), dates=("2024-01-02", "2024-01-03", "2024-01-02"),
         close=(185.64, 184.25, 370.87), volume=(82488700.0, None, 25258600.0)) -> Table:
    n = len(symbols)
    return Table.build(BARS_V1, {"symbol": list(symbols), "session_date": list(dates),
                                 "open": [None] * n, "high": list(close), "low": list(close),
                                 "close": list(close), "volume": list(volume)})


def test_row_order_does_not_change_the_hash():
    forward = bars()
    backward = bars(symbols=("MSFT", "AAPL", "AAPL"), dates=("2024-01-02", "2024-01-03", "2024-01-02"),
                    close=(370.87, 184.25, 185.64), volume=(25258600.0, None, 82488700.0))
    assert forward.table_hash() == backward.table_hash()
    assert list(forward.columns["symbol"]) == ["AAPL", "AAPL", "MSFT"]
    assert list(forward.columns["session_date"].astype(str)) == ["2024-01-02", "2024-01-03", "2024-01-02"]


def test_every_value_changes_the_hash():
    base = bars().table_hash()
    assert bars(close=(185.64, 184.25, 370.88)).table_hash() != base
    assert bars(volume=(82488700.0, 0.0, 25258600.0)).table_hash() != base
    assert bars(dates=("2024-01-02", "2024-01-04", "2024-01-02")).table_hash() != base
    assert bars(symbols=("AAPL", "AAPL", "MSFT.")).table_hash() != base


def test_negative_zero_is_zero():
    assert bars(volume=(0.0, None, 1.0)).table_hash() == bars(volume=(-0.0, None, 1.0)).table_hash()


def test_dates_can_be_given_as_dates_strings_or_whole_day_datetime64():
    as_str = bars().table_hash()
    as_date = bars(dates=(dt.date(2024, 1, 2), dt.date(2024, 1, 3), dt.date(2024, 1, 2))).table_hash()
    as_ns = bars(dates=np.array(["2024-01-02", "2024-01-03", "2024-01-02"], dtype="datetime64[ns]")).table_hash()
    assert as_str == as_date == as_ns


def test_a_timezone_aware_midnight_is_refused_not_shifted():
    """The trap from the plan: midnight New York is 05:00 UTC, the previous session under naive conversion."""
    import pandas as pd
    local = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-02"]).tz_localize("America/New_York")
    with pytest.raises(DatasetError, match="date"):
        bars(dates=local)
    with pytest.raises(DatasetError, match="whole days"):
        bars(dates=local.tz_convert("UTC").tz_localize(None).to_numpy())
    with pytest.raises(DatasetError, match="not a datetime"):
        bars(dates=(dt.datetime(2024, 1, 2), dt.datetime(2024, 1, 3), dt.datetime(2024, 1, 2)))


@pytest.mark.parametrize("dates", [("2024-01-02", "2024-02-30", "2024-01-02"),
                                   ("2024-01-02", "20240103", "2024-01-02"),
                                   ("2024-01-02", None, "2024-01-02")])
def test_bad_or_missing_dates_are_refused(dates):
    with pytest.raises(DatasetError):
        bars(dates=dates)


@pytest.mark.parametrize("close", [(1.0, float("inf"), 2.0), (1.0, float("nan"), 2.0), (1.0, None, 2.0),
                                   (1.0, "2.5", 2.0), (1.0, True, 2.0)])
def test_bad_or_missing_closes_are_refused(close):
    with pytest.raises(DatasetError):
        bars(close=close)


def test_a_nullable_column_takes_nan_and_none_as_null():
    assert bars(volume=(1.0, None, 2.0)).table_hash() == bars(volume=(1.0, float("nan"), 2.0)).table_hash()


def test_duplicate_keys_are_refused():
    with pytest.raises(DatasetError, match="duplicate key"):
        bars(dates=("2024-01-02", "2024-01-02", "2024-01-02"))


def test_strings_sort_by_code_point_not_utf16():
    """U+FF21 sorts before U+1F600 by code point (and UTF-8 bytes), after it in UTF-16."""
    table = bars(symbols=("\U0001F600", "Ａ", "A"), dates=("2024-01-02",) * 3)
    assert list(table.columns["symbol"]) == ["A", "Ａ", "\U0001F600"]


def test_lone_surrogates_are_refused():
    with pytest.raises(DatasetError, match="Unicode"):
        bars(symbols=("AAPL", "\ud800", "MSFT"))


def test_timestamps_must_be_timezone_aware_or_datetime64():
    def actions(announced):
        return Table.build(CORPORATE_ACTIONS_V1, {
            "symbol": ["AAPL"], "ex_date": ["2020-08-31"], "action": ["split"], "value": [4.0],
            "announced_at": announced, "source": ["test"]})
    aware = actions([dt.datetime(2020, 7, 30, 20, 30, tzinfo=dt.timezone.utc)])
    as_ns = actions(np.array(["2020-07-30T20:30"], dtype="datetime64[ns]"))
    import pandas as pd
    as_pandas = actions([pd.Timestamp("2020-07-30 16:30", tz="America/New_York")])
    assert aware.table_hash() == as_ns.table_hash() == as_pandas.table_hash()
    with pytest.raises(DatasetError, match="timezone-aware"):
        actions([dt.datetime(2020, 7, 30, 20, 30)])
    with pytest.raises(DatasetError, match="out of range"):
        actions(np.array(["2300-01-01"], dtype="datetime64[D]"))


@pytest.mark.parametrize("columns, key", [
    ((Column("a", "utf8"), Column("a", "utf8")), ("a",)),
    ((Column("a", "utf8", nullable=True),), ("a",)),
    ((Column("a", "utf8"),), ("b",)),
    ((Column("a", "int8"),), ("a",)),
    ((Column("A", "utf8"),), ("A",)),
    ((Column("a", "utf8"),), ()),
])
def test_bad_schemas_are_refused(columns, key):
    with pytest.raises(DatasetError):
        TableSchema("t", columns, key)


def test_tables_need_exactly_their_columns_of_one_length():
    with pytest.raises(DatasetError, match="exactly the columns"):
        Table.build(BARS_V1, {"symbol": ["A"]})
    with pytest.raises(DatasetError, match="different lengths"):
        Table.build(BARS_V1, {"symbol": ["A", "B"], "session_date": ["2024-01-02"], "open": [1.0],
                              "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0]})


def test_table_columns_are_read_only():
    with pytest.raises(ValueError):
        bars().columns["close"][0] = 0.0


# --------------------------------------------------------------------------
# Datasets and canonical JSON
# --------------------------------------------------------------------------

def test_every_identity_field_is_part_of_the_dataset_id():
    table = bars()
    base = Dataset(IDENTITY, (table,)).dataset_id
    for field, value in IDENTITY.as_dict().items():
        changed = DatasetIdentity(**{**IDENTITY.as_dict(), field: value + "x"})
        assert Dataset(changed, (table,)).dataset_id != base, field


@pytest.mark.parametrize("field, value", [("provider", ""), ("adjustment", "split dividend"),
                                          ("family", "x" * 65), ("licence_id", "é")])
def test_identity_fields_are_short_tokens(field, value):
    with pytest.raises(DatasetError):
        DatasetIdentity(**{**IDENTITY.as_dict(), field: value})


def test_a_dataset_needs_distinct_tables():
    with pytest.raises(DatasetError):
        Dataset(IDENTITY, ())
    with pytest.raises(DatasetError):
        Dataset(IDENTITY, (bars(), bars()))


@pytest.mark.parametrize("value", [1.5, 2**53, {"é": 1}, {1: 2}, [float("nan")], object()])
def test_canonical_json_refuses_values_outside_its_domain(value):
    with pytest.raises(DatasetError):
        fmt.canonical_json(value)


def test_canonical_json_matches_rfc8785_on_its_domain():
    # Sorted keys, no whitespace, raw UTF-8, short escapes for \b \f \n \r \t,
    # \u00XX (lowercase hex) for other controls: RFC 8785 section 3.2.2.2.
    assert fmt.canonical_json({"b": [True, None, -(2**53 - 1)], "a": "é\n\x01\"\\ "}) \
        == '{"a":"é\\n\\u0001\\"\\\\ ","b":[true,null,-9007199254740991]}'.encode()


# --------------------------------------------------------------------------
# Blob format
# --------------------------------------------------------------------------

def actions_table() -> Table:
    return Table.build(CORPORATE_ACTIONS_V1, {
        "symbol": ["AAPL", "AAPL"], "ex_date": ["2020-08-31", "2024-02-09"],
        "action": ["split", "dividend"], "value": [4.0, 0.24],
        "announced_at": [dt.datetime(2020, 7, 30, 20, 30, tzinfo=dt.timezone.utc), None],
        "source": ["test", "test"]})


def dataset() -> Dataset:
    return Dataset(IDENTITY, (bars(), actions_table()))


def test_a_blob_round_trips():
    original = dataset()
    decoded = fmt.decode_blob(fmt.encode_blob(original), original.dataset_id)
    assert decoded.dataset_id == original.dataset_id
    assert decoded.identity == IDENTITY
    for name in ("bars", "corporate_actions"):
        a, b = original.table(name), decoded.table(name)
        assert a.schema == b.schema
        for column in a.schema.columns:
            left, right = a.columns[column.name], b.columns[column.name]
            if column.type in ("date32", "timestamp_ns"):  # NaT != NaT
                left, right = left.view("int64"), right.view("int64")
            assert np.array_equal(left, right, equal_nan=column.type == "float64"), column.name
            assert left.dtype == right.dtype, column.name


def test_encoding_is_deterministic():
    assert fmt.encode_blob(dataset()) == fmt.encode_blob(dataset())


def _split(blob: bytes):
    (length,) = struct.unpack("<Q", blob[8:16])
    header = json.loads(blob[16:16 + length])
    return header, zlib.decompress(blob[16 + length:])


def _join(header, payload: bytes) -> bytes:
    raw = fmt.canonical_json(header)
    return b"QSBLOB\x00\x01" + struct.pack("<Q", len(raw)) + raw + zlib.compress(payload)


def test_a_changed_value_is_refused():
    original = dataset()
    header, payload = _split(fmt.encode_blob(original))
    close_offset = sum(c["length"] for c in header["tables"][0]["columns"][:5])
    tampered = bytearray(payload)
    tampered[close_offset] ^= 0x01  # lowest mantissa bit of the first close
    with pytest.raises(DatasetIntegrityError, match="does not match its id"):
        fmt.decode_blob(_join(header, bytes(tampered)), original.dataset_id)


def test_a_changed_identity_is_refused():
    original = dataset()
    header, payload = _split(fmt.encode_blob(original))
    header["identity"]["adjustment"] = "split"
    with pytest.raises(DatasetIntegrityError, match="does not match its id"):
        fmt.decode_blob(_join(header, payload), original.dataset_id)


def test_another_datasets_blob_is_refused():
    other = Dataset(IDENTITY, (bars(close=(1.0, 2.0, 3.0)),))
    with pytest.raises(DatasetIntegrityError, match="does not match its id"):
        fmt.decode_blob(fmt.encode_blob(other), dataset().dataset_id)


def test_a_bad_validity_byte_is_refused():
    original = dataset()
    header, payload = _split(fmt.encode_blob(original))
    open_offset = header["tables"][0]["columns"][0]["length"] + header["tables"][0]["columns"][1]["length"]
    tampered = bytearray(payload)
    tampered[open_offset] = 2
    with pytest.raises(DatasetIntegrityError, match="validity"):
        fmt.decode_blob(_join(header, bytes(tampered)), original.dataset_id)


@pytest.mark.parametrize("mangle", [
    lambda b: b[:-1],
    lambda b: b + b"\x00",
    lambda b: b"XSBLOB\x00\x01" + b[8:],
    lambda b: b[:20],
    lambda b: b[:8] + struct.pack("<Q", 1 << 40) + b[16:],
    lambda b: b"",
])
def test_malformed_blobs_are_refused(mangle):
    original = dataset()
    with pytest.raises(DatasetIntegrityError):
        fmt.decode_blob(mangle(fmt.encode_blob(original)), original.dataset_id)


@pytest.mark.parametrize("edit", [
    lambda h, p: ({**h, "payload_length": len(p) - 1}, p),
    lambda h, p: ({**h, "payload_length": len(p) + 1}, p),
    lambda h, p: ({**h, "format": "qs-blob-v2"}, p),
    lambda h, p: ({**h, "identity": {**h["identity"], "extra": "x"}}, p),
    lambda h, p: (h, p + b"\x00"),
])
def test_header_and_payload_must_agree(edit):
    original = dataset()
    with pytest.raises(DatasetIntegrityError):
        fmt.decode_blob(_join(*edit(*_split(fmt.encode_blob(original)))), original.dataset_id)


def test_the_read_limit_stops_a_decompression_bomb():
    """A small file declaring a huge payload is refused before it is inflated."""
    original = dataset()
    header, _ = _split(fmt.encode_blob(original))
    bomb = _join({**header, "payload_length": 64 << 20}, bytes(64 << 20))
    assert len(bomb) < 128 << 10
    with pytest.raises(DatasetIntegrityError, match="read limit"):
        fmt.decode_blob(bomb, original.dataset_id, max_bytes=1 << 20)


def test_a_hostile_header_is_refused():
    original = dataset()
    deep = b"[" * 100_000 + b"]" * 100_000
    blob = b"QSBLOB\x00\x01" + struct.pack("<Q", len(deep)) + deep
    with pytest.raises(DatasetIntegrityError):
        fmt.decode_blob(blob, original.dataset_id)
    for header in (b"[]", b"null", b'"x"'):
        blob = b"QSBLOB\x00\x01" + struct.pack("<Q", len(header)) + header
        with pytest.raises(DatasetIntegrityError):
            fmt.decode_blob(blob, original.dataset_id)
