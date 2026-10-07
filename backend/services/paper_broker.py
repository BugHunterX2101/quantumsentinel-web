"""QuantumSentinel — internal paper broker: cash ledger, reservations, fills.

This module is the single authority over paper-account money. Every write
preserves the invariant ``0 <= reserved_micros <= cash_micros`` and is a
single conditional UPDATE, so concurrent requests (threads or separate
gunicorn workers) can never spend the same dollars twice. A sell fill also
locks the account row while it checks the holding, so concurrent sells can
never be credited for the same shares.

Order lifecycle (all transitions are compare-and-set on ``trades.status``)::

    PENDING --(not immediately marketable)--> ACCEPTED --(sweeper)--> FILLED
       |                                          |
       +--> FILLED / EXPIRED (IOC) / REJECTED     +--> CANCELLED / REJECTED
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP

from sqlalchemy import Float, and_, case, literal, not_, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .. import models
from ..config import PAPER_INITIAL_CASH
from . import portfolio_service, trading_service

MICROS = Decimal(1_000_000)
OPEN_STATUSES = ("PENDING", "ACCEPTED")
_Account = models.PaperAccount
_Trade = models.Trade


def to_micros(amount) -> int:
    return int((Decimal(str(amount)) * MICROS).to_integral_value(rounding=ROUND_HALF_UP))


def from_micros(micros: int) -> float:
    return float(Decimal(int(micros)) / MICROS)


def notional_micros(quantity, price) -> int:
    return to_micros(Decimal(str(quantity)) * Decimal(str(price)))


def reservation_micros(side: str, quantity, order_type: str, limit_price, stop_price,
                       market_price=None) -> int:
    """Cash a buy order must hold while open. Sells reserve no cash.

    Derived purely from the order's own fields (market orders excepted, which
    use the price observed at placement and never rest), so the amount
    released later is bit-identical to the amount reserved.
    """
    if side != "buy":
        return 0
    if order_type in ("limit", "stop_limit"):
        reference = limit_price
    elif order_type == "stop":
        reference = stop_price
    else:
        reference = market_price
    if reference is None:
        return 0
    return notional_micros(quantity, reference)


def trade_reservation_micros(trade: models.Trade) -> int:
    """Reservation held by a persisted open order (market orders never rest)."""
    if trade.order_type == "market":
        return 0
    return reservation_micros(trade.side, trade.quantity, trade.order_type,
                              trade.limit_price, trade.stop_price)


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


# ---------------------------------------------------------------------------
# Account
# ---------------------------------------------------------------------------

def get_or_create_account(db: Session, user_id: str) -> models.PaperAccount:
    """Return the user's ledger row, opening it on first use.

    Users who traded before the ledger existed are migrated by replaying
    their filled trades, so their balance is continuous with history.
    """
    account = db.get(_Account, user_id)
    if account is not None:
        return account
    trades = db.execute(select(_Trade).where(_Trade.user_id == user_id)).scalars().all()
    cash = to_micros(PAPER_INITIAL_CASH)
    reserved = 0
    for t in trades:
        if t.status == "FILLED" and t.filled_price is not None:
            amount = notional_micros(t.quantity, t.filled_price)
            cash += amount if t.side == "sell" else -amount
        elif t.status in OPEN_STATUSES:
            reserved += trade_reservation_micros(t)
    cash = max(0, cash)
    db.add(_Account(user_id=user_id, cash_micros=cash, reserved_micros=min(reserved, cash)))
    try:
        db.commit()
    except IntegrityError:
        db.rollback()  # a concurrent request opened it first
    account = db.get(_Account, user_id)
    db.refresh(account)
    return account


def try_reserve(db: Session, user_id: str, amount: int) -> bool:
    """Atomically move ``amount`` from available to reserved cash.

    Not committed: the caller commits it together with the order insert, so
    a failed insert rolls the reservation back with it.
    """
    if amount <= 0:
        return True
    result = db.execute(
        update(_Account)
        .where(_Account.user_id == user_id,
               _Account.cash_micros - _Account.reserved_micros >= amount)
        .values(reserved_micros=_Account.reserved_micros + amount, updated_at=_now())
        .execution_options(synchronize_session=False)
    )
    return result.rowcount == 1


def _released(amount: int):
    return case((_Account.reserved_micros >= amount, _Account.reserved_micros - amount), else_=0)


def _release_stmt(user_id: str, amount: int):
    return (update(_Account).where(_Account.user_id == user_id)
            .values(reserved_micros=_released(amount), updated_at=_now())
            .execution_options(synchronize_session=False))


# ---------------------------------------------------------------------------
# Order transitions
# ---------------------------------------------------------------------------

def _cas_status(db: Session, trade_id: str, to_status: str, **values) -> bool:
    result = db.execute(
        update(_Trade)
        .where(_Trade.id == trade_id, _Trade.status.in_(OPEN_STATUSES))
        .values(status=to_status, **values)
        .execution_options(synchronize_session=False)
    )
    return result.rowcount == 1


def close_order(db: Session, trade: models.Trade, to_status: str, reserved: int | None = None) -> bool:
    """Move an open order to a terminal non-fill status and release its cash."""
    if reserved is None:
        reserved = trade_reservation_micros(trade)
    if not _cas_status(db, trade.id, to_status):
        db.rollback()
        return False
    if reserved > 0:
        db.execute(_release_stmt(trade.user_id, reserved))
    db.commit()
    db.refresh(trade)
    return True


def accept_order(db: Session, trade: models.Trade) -> bool:
    """PENDING -> ACCEPTED: the order now rests and keeps its reservation."""
    result = db.execute(
        update(_Trade).where(_Trade.id == trade.id, _Trade.status == "PENDING")
        .values(status="ACCEPTED").execution_options(synchronize_session=False)
    )
    db.commit()
    db.refresh(trade)
    return result.rowcount == 1


@dataclass(frozen=True)
class FillOutcome:
    status: str          # FILLED | NOT_OPEN | REJECTED
    reason: str | None = None


def fill_order(db: Session, trade: models.Trade, fill_price: float,
               reserved: int | None = None) -> FillOutcome:
    """Atomically fill an open order and settle cash in one transaction.

    A buy whose actual cost exceeds what the account can pay (possible only
    when a stop order gaps through its trigger) is rejected rather than
    overdrawing: the paper account is a cash account with no margin. A sell
    larger than the current holding is rejected for the same reason, since
    crediting its full proceeds would mint cash from a phantom position.
    """
    if reserved is None:
        reserved = trade_reservation_micros(trade)
    cost = notional_micros(trade.quantity, fill_price)
    if not _cas_status(db, trade.id, "FILLED", filled_price=fill_price, filled_at=_now()):
        db.rollback()
        return FillOutcome("NOT_OPEN")

    if trade.side == "buy":
        new_reserved = _released(reserved)
        result = db.execute(
            update(_Account)
            .where(_Account.user_id == trade.user_id,
                   _Account.cash_micros - cost >= new_reserved)
            .values(cash_micros=_Account.cash_micros - cost, reserved_micros=new_reserved,
                    updated_at=_now())
            .execution_options(synchronize_session=False)
        )
        if result.rowcount != 1:
            db.rollback()
            close_order(db, trade, "REJECTED", reserved)
            return FillOutcome("REJECTED", "insufficient available cash at fill")
    else:
        # Concurrent sells of one holding must not each be credited. Locking
        # the ledger row serialises this user's fills across every process,
        # and the holding is then read from the filled trades themselves
        # (committed fills included), not from the positions projection,
        # which is only rebuilt after a fill commits.
        db.execute(select(_Account.user_id).where(_Account.user_id == trade.user_id).with_for_update())
        held = held_quantity(db, trade.user_id, trade.asset, exclude_trade_id=trade.id)
        if Decimal(str(trade.quantity)) > held:
            db.rollback()
            close_order(db, trade, "REJECTED", reserved)
            return FillOutcome("REJECTED", "sell quantity exceeds the current position")
        db.execute(
            update(_Account).where(_Account.user_id == trade.user_id)
            .values(cash_micros=_Account.cash_micros + cost, updated_at=_now())
            .execution_options(synchronize_session=False)
        )
    db.commit()
    db.refresh(trade)
    # Positions must reflect this fill before any later fill checks holdings.
    portfolio_service.recompute_positions(db, trade.user_id)
    return FillOutcome("FILLED")


def held_quantity(db: Session, user_id: str, asset: str, exclude_trade_id: str | None = None) -> Decimal:
    """Quantity held, exactly as the positions rebuild counts it
    (portfolio_service.recompute_positions): filled buys and sells in fill
    order, a sell never taking the holding below zero."""
    query = select(_Trade.side, _Trade.quantity).where(
        _Trade.user_id == user_id, _Trade.asset == asset, _Trade.status == "FILLED"
    ).order_by(_Trade.filled_at)
    if exclude_trade_id is not None:
        query = query.where(_Trade.id != exclude_trade_id)
    held = Decimal(0)
    for side, quantity in db.execute(query):
        held = held + quantity if side == "buy" else max(Decimal(0), held - quantity)
    return held


# ---------------------------------------------------------------------------
# Read models
# ---------------------------------------------------------------------------

def open_order_exposure(db: Session, user_id: str) -> tuple[dict[str, int], dict[str, float]]:
    """Per-asset (reserved buy cash in micros, pending sell quantity)."""
    rows = db.execute(
        select(_Trade).where(_Trade.user_id == user_id, _Trade.status.in_(OPEN_STATUSES))
    ).scalars().all()
    buy_reserved: dict[str, int] = {}
    pending_sell: dict[str, float] = {}
    for t in rows:
        if t.side == "buy":
            buy_reserved[t.asset] = buy_reserved.get(t.asset, 0) + trade_reservation_micros(t)
        else:
            pending_sell[t.asset] = pending_sell.get(t.asset, 0.0) + float(t.quantity)
    return buy_reserved, pending_sell


def account_snapshot(db: Session, user_id: str, positions: list[dict]) -> dict:
    """Cash, buying power and exposure computed from current positions.

    Exposure is measured on what is held now (quantity x mark), so a closed
    round trip contributes nothing — unlike summing historical trade notional.
    """
    account = get_or_create_account(db, user_id)
    cash, reserved = int(account.cash_micros), int(account.reserved_micros)
    # Unrounded quantity x mark: the per-row market_value is rounded for display.
    values = [float(p["quantity"]) * float(p["current_price"]) for p in positions]
    long_exposure = sum(v for v in values if v > 0)
    short_exposure = -sum(v for v in values if v < 0)
    equity = from_micros(cash) + sum(values)
    return {
        "initial_cash": PAPER_INITIAL_CASH,
        "cash": from_micros(cash),
        "reserved_cash": from_micros(reserved),
        "available_cash": from_micros(cash - reserved),
        "positions_value": round(sum(values), 6),
        "equity": round(equity, 6),
        "total_pnl": round(equity - PAPER_INITIAL_CASH, 6),
        "gross_exposure": round(long_exposure + short_exposure, 6),
        "net_exposure": round(long_exposure - short_exposure, 6),
        "long_exposure": round(long_exposure, 6),
        "short_exposure": round(short_exposure, 6),
        "prices_stale": any(p.get("price_stale") for p in positions),
    }


# ---------------------------------------------------------------------------
# Resting-order sweeper
# ---------------------------------------------------------------------------

def process_open_orders(db: Session) -> list[tuple[models.Trade, FillOutcome]]:
    """Fill every resting order that the current market has made executable.

    Called by the background sweeper, never by a read endpoint. One price
    fetch per asset; an asset without a current price is skipped (its orders
    stay open) rather than filled at a guessed price. Safe to run from
    several workers at once: each fill is a compare-and-set.

    Only the orders the current price can act on are loaded (see
    _actionable): a book of resting orders far from the market costs one
    small indexed query per asset, not a load of every order every sweep.
    """
    assets = db.execute(
        select(_Trade.asset).where(_Trade.status == "ACCEPTED").distinct()
    ).scalars().all()

    outcomes: list[tuple[models.Trade, FillOutcome]] = []
    for asset in assets:
        try:
            last = trading_service.get_last_price(asset)
        except trading_service.MarketDataUnavailable:
            continue
        trades = db.execute(
            select(_Trade).where(_Trade.status == "ACCEPTED", _Trade.asset == asset, _actionable(last))
            .order_by(_Trade.submitted_at)
        ).scalars().all()
        for t in trades:
            limit = float(t.limit_price) if t.limit_price is not None else None
            stop = float(t.stop_price) if t.stop_price is not None else None
            if t.order_type == "limit" and limit is not None:
                fill_price = trading_service.check_pending_limit_fill(asset, t.side, limit, last_price=last)
            elif t.order_type in ("stop", "stop_limit") and stop is not None:
                fill = trading_service.simulate_fill(asset, t.side, float(t.quantity), t.order_type,
                                                     limit, stop, last_price=last)
                fill_price = fill["filled_price"] if fill["status"] == "FILLED" else None
                if fill_price is None and t.order_type == "stop_limit" and stop_triggered(t.side, stop, last):
                    # A triggered stop-limit is a live limit order from now on;
                    # later sweeps must not re-test the stop and "untrigger" it.
                    db.execute(update(_Trade).where(_Trade.id == t.id, _Trade.status == "ACCEPTED")
                               .values(order_type="limit").execution_options(synchronize_session=False))
                    db.commit()
            else:
                fill_price = last
            if fill_price is not None:
                outcomes.append((t, fill_order(db, t, fill_price)))
    return outcomes


def _actionable(last: float):
    """SQL for exactly the resting orders the sweep loop below acts on at
    price ``last``: marketable limits, triggered stops and stop-limits (a
    triggered stop-limit that is not marketable is converted to a limit),
    and any order without the price its type needs (filled at ``last``).

    ``last`` is bound as float8 so PostgreSQL compares in the same double
    precision as the Python checks that make the final decision.
    """
    p = literal(last, Float)
    has_limit = and_(_Trade.order_type == "limit", _Trade.limit_price.is_not(None))
    has_stop = and_(_Trade.order_type.in_(("stop", "stop_limit")), _Trade.stop_price.is_not(None))
    return or_(
        and_(has_limit, or_(and_(_Trade.side == "buy", _Trade.limit_price >= p),
                            and_(_Trade.side == "sell", _Trade.limit_price <= p))),
        and_(has_stop, or_(and_(_Trade.side == "buy", _Trade.stop_price <= p),
                           and_(_Trade.side == "sell", _Trade.stop_price >= p))),
        and_(not_(has_limit), not_(has_stop)),
    )


def stop_triggered(side: str, stop: float, last: float) -> bool:
    return last >= stop if side == "buy" else last <= stop
