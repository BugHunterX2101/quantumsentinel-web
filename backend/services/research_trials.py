"""QuantumSentinel — server-side count of research trials.

The Deflated Sharpe test discounts a result by the number of strategy
variants tried before it was found. A count the client declares can be
anything; this ledger records every distinct configuration the server
actually evaluated for a user, grouped by research question ("family": the
same strategy on the same assets), so the test can use at least that many.
"""
from __future__ import annotations

import hashlib
import json

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .. import models

_RECORD_ATTEMPTS = 3


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     default=str).encode()).hexdigest()


def family_hash(strategy: str, assets: list[str]) -> str:
    """One research question: a strategy applied to a set of assets."""
    return _digest({"strategy": strategy,
                    "assets": sorted({a.strip().upper() for a in assets})})


def record(db: Session, user_id: str, family: str, configs: list[dict], source: str) -> int:
    """Record each distinct config as a trial; return the family's trial count."""
    hashes = list(dict.fromkeys(_digest(c) for c in configs))
    for _ in range(_RECORD_ATTEMPTS):
        existing = set(db.execute(
            select(models.ResearchTrial.config_hash).where(
                models.ResearchTrial.user_id == user_id,
                models.ResearchTrial.family_hash == family,
                models.ResearchTrial.config_hash.in_(hashes),
            )
        ).scalars()) if hashes else set()
        new = [h for h in hashes if h not in existing]
        if not new:
            break
        db.add_all(models.ResearchTrial(user_id=user_id, family_hash=family, config_hash=h,
                                        source=source) for h in new)
        try:
            db.commit()
            break
        except IntegrityError:
            # A concurrent request recorded one of these first: re-read and
            # insert only what is still missing.
            db.rollback()
    return count(db, user_id, family)


def count(db: Session, user_id: str, family: str) -> int:
    return int(db.execute(
        select(func.count()).select_from(models.ResearchTrial).where(
            models.ResearchTrial.user_id == user_id,
            models.ResearchTrial.family_hash == family,
        )
    ).scalar() or 0)
