"""QuantumSentinel — server-side count of research trials.

The Deflated Sharpe test discounts a result by the number of strategy
variants tried before it was found. A count the client declares can be
anything; this ledger records every distinct configuration the server
actually evaluated for a user, grouped by research question ("family": the
same strategy on the same assets), so the test can use at least that many.

A run of a strategy version also records what it measured
(``record_evaluation``): the Sharpe ratio, the number of returns and the
return series' id (``evaluation_metrics``), all from the one series of daily
returns its backtest produced, in full and unrounded.

The return-series id is ``qs-returns-v1:<hex>``: each return r becomes the
integer round(r * 10^12), written as a signed 64-bit big-endian integer, and
the id is the SHA-256 of those 8-byte words in order. Quantising makes the id
independent of how a language prints or stores a float, so another
implementation can recompute it from the returns; two computations that
differ by less than 10^-12 get the same id unless a return falls on either
side of a rounding boundary. tests/test_trial_evaluations.py holds a golden
vector.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .. import models

_RECORD_ATTEMPTS = 3
RETURNS_SCHEME = "qs-returns-v1"
RETURNS_ID = re.compile(rf"{RETURNS_SCHEME}:[0-9a-f]{{64}}")
_RETURNS_SCALE = 1e12
_INT64_MAX = 2 ** 63 - 1
PERIODS_PER_YEAR = 252


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     default=str).encode()).hexdigest()


def config_hash(config: dict) -> str:
    """The hash a trial's config is recorded under."""
    return _digest(config)


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


def quantise(returns) -> list[int]:
    """Each return r as the integer round(r * 10^12) the series' id hashes."""
    quantised = []
    for r in np.asarray(returns, dtype=float).reshape(-1):
        if not math.isfinite(r):
            raise ValueError("every return must be finite")
        quantised.append(round(float(r) * _RETURNS_SCALE))
    return quantised


def quantised_returns_id(quantised) -> str:
    """The id of a series already quantised (``quantise``): the SHA-256 of its
    integers as signed 64-bit big-endian words."""
    digest = hashlib.sha256()
    for q in quantised:
        if isinstance(q, bool) or not isinstance(q, int):
            raise ValueError("a quantised return is an integer")
        if not -_INT64_MAX - 1 <= q <= _INT64_MAX:
            raise ValueError("a return is too large to identify")
        digest.update(q.to_bytes(8, "big", signed=True))
    return f"{RETURNS_SCHEME}:{digest.hexdigest()}"


def returns_id(returns) -> str:
    """The id of a series of returns (see the module docstring)."""
    return quantised_returns_id(quantise(returns))


def evaluation_metrics(returns) -> dict:
    """What a backtest's daily returns measure, all from that one series: the
    annualised Sharpe ratio as the event engine computes it (mean times 252
    over the sample standard deviation times sqrt(252), at least 1e-9), the
    number of returns (the observations the Deflated Sharpe test takes) and
    the series' id."""
    series = np.asarray(returns, dtype=float).reshape(-1)
    if len(series) < 2:
        raise ValueError("at least two returns are needed")
    if not np.isfinite(series).all():
        raise ValueError("every return must be finite")
    annual_return = float(series.mean()) * PERIODS_PER_YEAR
    annual_volatility = float(series.std(ddof=1)) * math.sqrt(PERIODS_PER_YEAR)
    return {"sharpe": annual_return / max(annual_volatility, 1e-9), "n_obs": int(len(series)),
            "returns_sha256": returns_id(series)}


def record_evaluation(db: Session, *, job_id: str, user_id: str, family: str, config: dict,
                      strategy_version_id: str, metrics: dict, dataset_ids: list[str]
                      ) -> models.ResearchTrialEvaluation:
    """Record what an accepted run of a strategy version measured, in the
    caller's transaction (the one that stores the job's result). The run's
    trial, ``config`` in ``family``, must already be recorded (``record``)."""
    sharpe, n_obs, series_id = metrics["sharpe"], metrics["n_obs"], metrics["returns_sha256"]
    if not isinstance(sharpe, (int, float)) or isinstance(sharpe, bool) or not math.isfinite(sharpe):
        raise ValueError("the Sharpe ratio must be a finite number")
    if not isinstance(n_obs, int) or isinstance(n_obs, bool) or n_obs < 2:
        raise ValueError("an evaluation needs at least two returns")
    if not isinstance(series_id, str) or not RETURNS_ID.fullmatch(series_id):
        raise ValueError(f"not a {RETURNS_SCHEME} return-series id")
    if not dataset_ids:
        raise ValueError("an evaluation names the datasets it read")
    row = models.ResearchTrialEvaluation(
        job_id=job_id, user_id=user_id, family_hash=family, config_hash=_digest(config),
        strategy_version_id=strategy_version_id, sharpe=float(sharpe), n_obs=n_obs,
        returns_sha256=series_id, dataset_ids=sorted(dataset_ids),
        created_at=dt.datetime.now(dt.timezone.utc))
    db.add(row)
    db.flush()
    return row
