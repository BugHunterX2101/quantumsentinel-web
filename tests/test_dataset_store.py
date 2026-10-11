"""Content-addressed dataset storage: atomic, write-once, verified on every read."""
import os
import stat
import threading

import pytest

from backend.services.dataset_format import (BARS_V1, Dataset, DatasetError, DatasetIdentity,
                                             DatasetIntegrityError, Table)
from backend.services.dataset_store import DatasetStore

IDENTITY = DatasetIdentity(family="US_EQUITY_EOD", provider="yfinance", licence_id="yahoo-personal-use",
                           frequency="1d", session_boundary="exchange-local", adjustment="raw",
                           survivorship="current_listings")


def dataset(close=185.64) -> Dataset:
    return Dataset(IDENTITY, (Table.build(BARS_V1, {
        "symbol": ["AAPL", "MSFT"], "session_date": ["2024-01-02", "2024-01-02"], "open": [None, 1.0],
        "high": [close, 2.0], "low": [close, 0.5], "close": [close, 1.5], "volume": [1e6, None]}),))


def test_a_blob_is_stored_under_its_id(tmp_path):
    original = dataset()
    stored = DatasetStore(tmp_path).put(original)
    digest = original.dataset_id.split(":")[1]
    assert stored.path == tmp_path / "qs-ds-sha256-v1" / digest[:2] / digest
    assert stored.size == stored.path.stat().st_size
    assert not stored.path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    assert [p.name for p in stored.path.parent.iterdir()] == [digest]  # no temporary left
    assert DatasetStore(tmp_path).get(original.dataset_id, stored.sha256).dataset_id == original.dataset_id


def test_storing_the_same_content_again_keeps_the_first_file(tmp_path):
    store = DatasetStore(tmp_path)
    first = store.put(dataset())
    mtime = first.path.stat().st_mtime_ns
    second = store.put(dataset())
    assert (second.path, second.sha256, second.size) == (first.path, first.sha256, first.size)
    assert first.path.stat().st_mtime_ns == mtime


def test_concurrent_writers_of_one_dataset_leave_one_verified_file(tmp_path):
    store, results, errors = DatasetStore(tmp_path), [], []

    def write():
        try:
            results.append(store.put(dataset()))
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)
    threads = [threading.Thread(target=write) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert {r.sha256 for r in results} == {results[0].sha256}
    assert len(list(results[0].path.parent.iterdir())) == 1
    store.get(dataset().dataset_id, results[0].sha256)


def _rewrite(path, data: bytes):
    os.chmod(path, 0o600)
    path.write_bytes(data)
    os.chmod(path, 0o444)


def test_a_corrupted_file_is_refused(tmp_path):
    store = DatasetStore(tmp_path)
    stored = store.put(dataset())
    blob = bytearray(stored.path.read_bytes())
    blob[-5] ^= 0xFF
    _rewrite(stored.path, bytes(blob))
    with pytest.raises(DatasetIntegrityError):
        store.get(stored.dataset_id)
    with pytest.raises(DatasetIntegrityError):
        store.put(dataset())  # never trusts or overwrites a bad file silently


def test_a_substituted_file_is_refused(tmp_path):
    """Another dataset's valid blob, copied under this id, does not pass for it."""
    store = DatasetStore(tmp_path)
    stored = store.put(dataset())
    other = store.put(dataset(close=1.0))
    _rewrite(stored.path, other.path.read_bytes())
    with pytest.raises(DatasetIntegrityError, match="does not match its id"):
        store.get(stored.dataset_id)


def test_the_recorded_file_hash_is_checked(tmp_path):
    store = DatasetStore(tmp_path)
    stored = store.put(dataset())
    with pytest.raises(DatasetIntegrityError, match="recorded hash"):
        store.get(stored.dataset_id, "0" * 64)


def test_a_missing_blob_is_an_integrity_error(tmp_path):
    with pytest.raises(DatasetIntegrityError, match="no stored blob"):
        DatasetStore(tmp_path).get(dataset().dataset_id)


@pytest.mark.parametrize("bad", ["../../etc/passwd", "qs-ds-sha256-v1:" + "A" * 64,
                                 "qs-ds-sha256-v1:" + "0" * 63, "qs-ds-sha256-v1:" + "0" * 64 + "/..",
                                 "qs-col-sha256-v1:" + "0" * 64, None])
def test_only_dataset_ids_name_paths(tmp_path, bad):
    with pytest.raises(DatasetError):
        DatasetStore(tmp_path).get(bad)


def test_the_size_limit_applies_to_reads_and_writes(tmp_path):
    stored = DatasetStore(tmp_path).put(dataset())
    with pytest.raises(DatasetIntegrityError, match="read limit"):
        DatasetStore(tmp_path, max_bytes=stored.size - 1).get(stored.dataset_id)
    with pytest.raises(DatasetError, match="limit"):
        DatasetStore(tmp_path / "small", max_bytes=10).put(dataset())
    assert not (tmp_path / "small").exists()
