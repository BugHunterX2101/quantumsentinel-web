"""QuantumSentinel — the registry of frozen datasets.

A dataset is registered only after its blob has been read back from the
store and verified against its id (DatasetStore.get), so a row always
describes bytes that exist and hash to its name. Everything the row says is
derived here from the verified dataset; nothing is taken on trust from the
process that fetched it except its provenance (request, client, time).

Rows in ``dataset_versions`` never change (migration 0004). Registering the
same content again, by another user or a later fetch, keeps the first row
and its provenance, and only records that this user holds it.
"""
from __future__ import annotations

import datetime as dt
import logging

import numpy as np
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from .. import models
from .dataset_format import BLOB_FORMAT, Dataset, DatasetError, DatasetIntegrityError, canonical_json
from .dataset_providers import LICENCES
from .dataset_store import DatasetStore, StoredDataset

log = logging.getLogger(__name__)

IDENTITY_FIELDS = ("family", "provider", "licence_id", "frequency", "session_boundary", "adjustment",
                   "survivorship")


class RegistryError(ValueError):
    """A dataset the registry refuses to describe."""


def describe(dataset: Dataset, stored: StoredDataset, request: dict, provider_client: str,
             retrieved_at: dt.datetime) -> dict:
    """The dataset_versions row for a verified dataset and its stored blob."""
    if stored.dataset_id != dataset.dataset_id:
        raise RegistryError("the stored blob is not this dataset")
    identity = dataset.identity
    licence = LICENCES.get(identity.licence_id)
    if licence is None:
        raise RegistryError(f"unknown licence {identity.licence_id!r}")
    try:
        bars = dataset.table("bars")
    except KeyError:
        raise RegistryError("a registered dataset needs a bars table") from None
    if not len(bars):
        raise RegistryError("a registered dataset needs at least one bar")
    if retrieved_at.tzinfo is None:
        raise RegistryError("retrieved_at must be timezone-aware")
    sessions = bars.columns["session_date"]
    return {
        "dataset_id": dataset.dataset_id,
        **identity.as_dict(),
        "personal_use_only": licence.personal_use_only,
        "redistribution_allowed": licence.redistribution_allowed,
        # Prices are as known at each date only if no later action restates
        # them (raw) and the universe is not today's survivors.
        "point_in_time": identity.adjustment == "raw" and identity.survivorship == "point_in_time",
        "coverage_start": sessions.min().item(),
        "coverage_end": sessions.max().item(),
        "symbols": sorted(str(s) for s in np.unique(bars.columns["symbol"])),
        "manifest_canonical": canonical_json(dataset.manifest()),
        "tables_json": {t.schema.name: t.manifest() for t in dataset.tables},
        "blob_format": BLOB_FORMAT,
        "blob_sha256": stored.sha256,
        "blob_size": stored.size,
        "provider_client": provider_client[:64],
        "request_json": request,
        "retrieved_at": retrieved_at.astimezone(dt.timezone.utc),
    }


def register(db: Session, dataset: Dataset, stored: StoredDataset, *, user_id: str, request: dict,
             provider_client: str, retrieved_at: dt.datetime,
             job_id: str | None = None) -> models.DatasetVersion:
    """Record ``dataset`` (verified, stored as ``stored``) and that
    ``user_id`` holds it, in the caller's transaction."""
    row = describe(dataset, stored, request, provider_client, retrieved_at)
    now = dt.datetime.now(dt.timezone.utc)
    db.execute(insert(models.DatasetVersion).values(**row, created_at=now)
               .on_conflict_do_nothing(index_elements=["dataset_id"]))
    version = db.execute(select(models.DatasetVersion).where(
        models.DatasetVersion.dataset_id == row["dataset_id"])).scalar_one()
    # Equal content encodes to equal bytes; a different blob under the same
    # id means one of the two writers is wrong.
    if version.blob_sha256 != row["blob_sha256"] or version.blob_size != row["blob_size"]:
        raise DatasetIntegrityError(f"{row['dataset_id']}: a different blob is already registered")
    db.execute(insert(models.UserDataset).values(user_id=user_id, dataset_id=row["dataset_id"],
                                                 job_id=job_id, created_at=now)
               .on_conflict_do_nothing(index_elements=["user_id", "dataset_id"]))
    return version


def finalize_snapshot_job(db: Session, job: models.ResearchJob, result: dict, audit: dict):
    """Register what a dataset_snapshot job stored (research_jobs.JobKind).

    The job process fetched and wrote the blobs; it is not trusted to have
    written what it reports. Each blob is read back here, rehashed against
    its id and the reported file hash, and only then registered, in the
    transaction that stores the job's result.
    """
    from ..config import DATASET_ROOT
    from .research_tasks import TaskError

    store = DatasetStore(DATASET_ROOT)
    retrieved_at = dt.datetime.fromisoformat(result["retrieved_at"])
    versions = []
    for item in result["stored"]:
        try:
            dataset, blob = store.verified(item["dataset_id"], item["sha256"])
        except (DatasetError, DatasetIntegrityError) as exc:
            log.error("dataset snapshot job %s: %s", job.id, exc)
            raise TaskError(500, "The snapshot could not be verified after it was stored") from exc
        versions.append(register(db, dataset, blob, user_id=job.user_id, request=result["request"],
                                 provider_client=result["provider_client"], retrieved_at=retrieved_at,
                                 job_id=job.id))
    outcome = {"datasets": [view(v) for v in versions], "unavailable": result["unavailable"]}
    return outcome, ("dataset", versions[0].dataset_id if versions else None), audit


def _held_by(user_id: str):
    return (select(models.DatasetVersion)
            .join(models.UserDataset, models.UserDataset.dataset_id == models.DatasetVersion.dataset_id)
            .where(models.UserDataset.user_id == user_id))


def list_for_user(db: Session, user_id: str, limit: int = 50) -> list[models.DatasetVersion]:
    return list(db.execute(_held_by(user_id).order_by(
        models.UserDataset.created_at.desc(), models.DatasetVersion.dataset_id).limit(limit)).scalars())


def get_for_user(db: Session, user_id: str, dataset_id: str) -> models.DatasetVersion | None:
    return db.execute(_held_by(user_id).where(
        models.DatasetVersion.dataset_id == dataset_id)).scalar_one_or_none()


def view(version: models.DatasetVersion) -> dict:
    """The dataset as the API reports it."""
    return {
        "dataset_id": version.dataset_id,
        "identity": {name: getattr(version, name) for name in IDENTITY_FIELDS},
        "licence": {"personal_use_only": version.personal_use_only,
                    "redistribution_allowed": version.redistribution_allowed},
        "point_in_time": version.point_in_time,
        "coverage": {"start": version.coverage_start.isoformat(), "end": version.coverage_end.isoformat()},
        "symbols": version.symbols,
        "tables": {name: {"rows": manifest["rows"]} for name, manifest in version.tables_json.items()},
        "blob": {"format": version.blob_format, "sha256": version.blob_sha256, "size": version.blob_size},
        "provider_client": version.provider_client,
        "request": version.request_json,
        "retrieved_at": version.retrieved_at.astimezone(dt.timezone.utc).isoformat(),
    }
