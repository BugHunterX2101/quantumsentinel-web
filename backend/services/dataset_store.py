"""QuantumSentinel — content-addressed storage for dataset blobs.

A dataset blob lives at ``<root>/qs-ds-sha256-v1/<first 2 hex>/<hex>``, named
by its dataset id, with only metadata in PostgreSQL: a bytea value is capped
at 1 GB and is fully WAL-logged. Blobs are written once to a temporary name
in the same directory, flushed, made read-only and renamed into place, so a
reader never sees a partial file. Every read decodes and rehashes the blob,
so a corrupted or substituted file is refused rather than used.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .dataset_format import (DATASET_ID, DATASET_SCHEME, DEFAULT_MAX_BYTES, Dataset,
                             DatasetError, DatasetIntegrityError, decode_blob, encode_blob)


@dataclass(frozen=True)
class StoredDataset:
    dataset_id: str
    path: Path
    sha256: str  # of the stored bytes, for integrity checks and backups
    size: int


class DatasetStore:
    def __init__(self, root: str | os.PathLike, max_bytes: int = DEFAULT_MAX_BYTES):
        self.root = Path(root)
        self.max_bytes = max_bytes

    def path_for(self, dataset_id: str) -> Path:
        if not isinstance(dataset_id, str) or not DATASET_ID.fullmatch(dataset_id):
            raise DatasetError("not a dataset id")
        digest = dataset_id.split(":", 1)[1]
        return self.root / DATASET_SCHEME / digest[:2] / digest

    def put(self, dataset: Dataset) -> StoredDataset:
        """Store ``dataset``; storing the same content again keeps the first file."""
        dataset_id = dataset.dataset_id
        path = self.path_for(dataset_id)
        if path.exists():
            return self._verified(dataset_id, path)
        blob = encode_blob(dataset)
        if len(blob) > self.max_bytes:
            raise DatasetError("dataset blob is larger than the store's limit")
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(blob)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o444)
            try:
                os.replace(temporary, path)
            except PermissionError:
                # Windows will not replace a read-only file: another writer
                # stored this id first, so keep theirs once it verifies.
                if not path.exists():
                    raise
            else:
                _sync_directory(path.parent)
                return StoredDataset(dataset_id, path, hashlib.sha256(blob).hexdigest(), len(blob))
        finally:
            if os.path.exists(temporary):
                os.chmod(temporary, 0o600)
                os.unlink(temporary)
        return self._verified(dataset_id, path)

    def get(self, dataset_id: str, sha256: str | None = None) -> Dataset:
        """The stored dataset, verified against its id (and file hash, if given)."""
        path = self.path_for(dataset_id)
        blob = self._read(path)
        if sha256 is not None and hashlib.sha256(blob).hexdigest() != sha256:
            raise DatasetIntegrityError(f"{dataset_id}: stored bytes do not match the recorded hash")
        return decode_blob(blob, dataset_id, self.max_bytes)

    def _read(self, path: Path) -> bytes:
        try:
            if path.stat().st_size > self.max_bytes:
                raise DatasetIntegrityError(f"{path.name}: blob is larger than the read limit")
            return path.read_bytes()
        except FileNotFoundError as exc:
            raise DatasetIntegrityError(f"{path.name}: no stored blob") from exc

    def _verified(self, dataset_id: str, path: Path) -> StoredDataset:
        blob = self._read(path)
        decode_blob(blob, dataset_id, self.max_bytes)
        return StoredDataset(dataset_id, path, hashlib.sha256(blob).hexdigest(), len(blob))


def _sync_directory(directory: Path) -> None:
    """Persist a rename (POSIX); Windows has no directory handles to sync."""
    if os.name != "posix":
        return
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
