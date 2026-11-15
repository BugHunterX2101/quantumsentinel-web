"""QuantumSentinel — the registry of strategy versions.

A strategy version is a strategy spec (services/strategy_spec.py) stored as
its canonical JSON under its content hash. Rows never change (migration
0006): a changed strategy is a new version. Registering a spec that already
exists, by another user or again, keeps the first row and only records that
this user holds it, so one user never learns from the registry who else
wrote the same spec.

A tag is a user's own name for a version they hold, such as ``1.2.0``. It is
an alias for reading: it belongs to that user, always names the version it
was first given to, and means nothing to any gate.
"""
from __future__ import annotations

import datetime as dt
import json
import re
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .. import models
from . import strategy_spec
from .strategy_spec import STRATEGY_VERSION_ID, StrategySpec

LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


class StrategyRegistryError(ValueError):
    """A registry request that cannot be honoured."""


def register(db: Session, value: Any, *, user_id: str) -> models.StrategyVersion:
    """Validate a spec (parsed JSON), record it and that ``user_id`` holds it,
    in the caller's transaction. Raises pydantic's ValidationError for a spec
    that is not DSL v1."""
    spec = strategy_spec.parse(value)
    canonical = strategy_spec.canonical(spec)
    version_id = strategy_spec.version_id(spec)
    now = dt.datetime.now(dt.timezone.utc)
    db.execute(insert(models.StrategyVersion).values(
        version_id=version_id, dsl=spec.dsl, template=spec.signal.template, family=spec.universe.family,
        spec_canonical=canonical, created_at=now).on_conflict_do_nothing(index_elements=["version_id"]))
    version = db.execute(select(models.StrategyVersion).where(
        models.StrategyVersion.version_id == version_id)).scalar_one()
    # The database checks the stored bytes hash to the id, so equal ids hold
    # equal bytes; checked here too, as the registry promises the caller.
    if bytes(version.spec_canonical) != canonical:
        raise StrategyRegistryError(f"{version_id}: different bytes are already registered")
    db.execute(insert(models.UserStrategyVersion).values(user_id=user_id, version_id=version_id, created_at=now)
               .on_conflict_do_nothing(index_elements=["user_id", "version_id"]))
    return version


def spec_of(version: models.StrategyVersion) -> StrategySpec:
    """The spec a stored version holds, validated again by today's DSL code.

    A version written by an earlier release that today's validators refuse,
    or that no longer encodes to its own id, is refused rather than run as
    something other than what was registered."""
    raw = bytes(version.spec_canonical)
    try:
        spec = strategy_spec.parse(json.loads(raw))
    except ValueError as exc:  # pydantic's ValidationError and JSONDecodeError are ValueErrors
        raise StrategyRegistryError(f"{version.version_id}: the stored spec no longer validates: {exc}") from exc
    if strategy_spec.canonical(spec) != raw or strategy_spec.version_id(spec) != version.version_id:
        raise StrategyRegistryError(f"{version.version_id}: the stored spec no longer encodes to its id")
    return spec


def get_for_user(db: Session, user_id: str, version_id: str) -> models.StrategyVersion | None:
    """The version, if ``user_id`` holds it."""
    return db.execute(
        select(models.StrategyVersion)
        .join(models.UserStrategyVersion, models.UserStrategyVersion.version_id == models.StrategyVersion.version_id)
        .where(models.UserStrategyVersion.user_id == user_id,
               models.StrategyVersion.version_id == version_id)).scalar_one_or_none()


def list_for_user(db: Session, user_id: str) -> list[tuple[models.StrategyVersion, list[str]]]:
    """The versions ``user_id`` holds, first registered first, each with the
    user's tags for it in the order given."""
    rows = db.execute(
        select(models.StrategyVersion)
        .join(models.UserStrategyVersion, models.UserStrategyVersion.version_id == models.StrategyVersion.version_id)
        .where(models.UserStrategyVersion.user_id == user_id)
        .order_by(models.UserStrategyVersion.created_at, models.StrategyVersion.version_id)).scalars().all()
    tags: dict[str, list[str]] = {}
    for row in db.execute(select(models.UserStrategyTag).where(models.UserStrategyTag.user_id == user_id)
                          .order_by(models.UserStrategyTag.created_at, models.UserStrategyTag.label)).scalars():
        tags.setdefault(row.version_id, []).append(row.label)
    return [(version, tags.get(version.version_id, [])) for version in rows]


def tag(db: Session, user_id: str, version_id: str, label: str) -> models.UserStrategyTag:
    """Give ``label`` to a version ``user_id`` holds, in the caller's
    transaction. Giving a label again to the version it names is a no-op; a
    label that names another version is refused, as tags never move."""
    if not isinstance(label, str) or not LABEL.fullmatch(label):
        raise StrategyRegistryError("a tag is 1-64 letters, digits, '.', '_' or '-', starting with a letter or digit")
    if get_for_user(db, user_id, version_id) is None:
        raise StrategyRegistryError("no such strategy version")
    existing = db.get(models.UserStrategyTag, (user_id, label))
    if existing is None:
        row = models.UserStrategyTag(user_id=user_id, label=label, version_id=version_id,
                                     created_at=dt.datetime.now(dt.timezone.utc))
        try:
            with db.begin_nested():
                db.add(row)
            return row
        except IntegrityError:
            # A concurrent request gave the label first.
            existing = db.get(models.UserStrategyTag, (user_id, label))
    if existing is None or existing.version_id != version_id:
        raise StrategyRegistryError(f"the tag {label!r} already names another version")
    return existing


def resolve(db: Session, user_id: str, ref: str) -> models.StrategyVersion | None:
    """A version ``user_id`` holds, named by its id or by one of their tags."""
    if STRATEGY_VERSION_ID.fullmatch(ref):
        return get_for_user(db, user_id, ref)
    found = db.get(models.UserStrategyTag, (user_id, ref)) if LABEL.fullmatch(ref) else None
    return get_for_user(db, user_id, found.version_id) if found else None
