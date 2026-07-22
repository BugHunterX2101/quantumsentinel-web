"""QuantumSentinel — Portfolio analytics: positions, P&L, VaR, Sharpe, drawdown."""
import math
from sqlalchemy.orm import Session
from sqlalchemy import select

from .. import models
from .trading_service import get_last_price


def recompute_positions(db: Session, user_id: str):
    """Rebuild the `positions` materialized view from filled trades (event
    sourcing pattern from the architecture doc: trades are the source of
    truth; positions are derived)."""
    trades = db.execute(
        select(models.Trade).where(
            models.Trade.user_id == user_id, models.Trade.status == "FILLED"
        ).order_by(models.Trade.filled_at)
    ).scalars().all()

    book: dict[str, dict] = {}
    for t in trades:
        b = book.setdefault(t.asset, {"qty": 0.0, "cost": 0.0, "realized": 0.0})
        qty, price = float(t.quantity), float(t.filled_price or 0)
        if t.side == "buy":
            b["cost"] += qty * price
            b["qty"] += qty
        else:
            avg_cost = (b["cost"] / b["qty"]) if b["qty"] else 0.0
            b["realized"] += (price - avg_cost) * min(qty, b["qty"])
            b["qty"] -= qty
            b["cost"] = avg_cost * b["qty"]

    # replace existing rows
    db.query(models.Position).filter(models.Position.user_id == user_id).delete()
    for asset, b in book.items():
        avg_entry = (b["cost"] / b["qty"]) if b["qty"] else 0.0
        db.add(models.Position(
            user_id=user_id, asset=asset, quantity=round(b["qty"], 6),
            avg_entry_price=round(avg_entry, 4), realized_pnl=round(b["realized"], 2),
        ))
    db.commit()


def get_positions_with_pnl(db: Session, user_id: str) -> list[dict]:
    positions = db.execute(
        select(models.Position).where(
            models.Position.user_id == user_id, models.Position.quantity != 0
        )
    ).scalars().all()
    out = []
    for p in positions:
        current_price = get_last_price(p.asset)
        unrealized = (current_price - float(p.avg_entry_price)) * float(p.quantity)
        out.append({
            "asset": p.asset, "quantity": float(p.quantity),
            "avg_entry_price": float(p.avg_entry_price), "current_price": current_price,
            "unrealized_pnl": round(unrealized, 2), "realized_pnl": float(p.realized_pnl),
            "market_value": round(current_price * float(p.quantity), 2),
        })
    return out


def equity_curve_from_trades(db: Session, user_id: str, starting_capital: float = 100_000.0) -> list[float]:
    trades = db.execute(
        select(models.Trade).where(
            models.Trade.user_id == user_id, models.Trade.status == "FILLED"
        ).order_by(models.Trade.filled_at)
    ).scalars().all()
    equity = starting_capital
    curve = [equity]
    book: dict[str, float] = {}
    for t in trades:
        qty, price = float(t.quantity), float(t.filled_price or 0)
        notional = qty * price
        if t.side == "buy":
            equity -= notional
            book[t.asset] = book.get(t.asset, 0) + qty
        else:
            equity += notional
            book[t.asset] = book.get(t.asset, 0) - qty
        curve.append(equity)
    return curve


def risk_metrics(db: Session, user_id: str) -> dict:
    trades = db.execute(
        select(models.Trade).where(
            models.Trade.user_id == user_id, models.Trade.status == "FILLED"
        ).order_by(models.Trade.filled_at)
    ).scalars().all()

    if len(trades) < 2:
        return {
            "sharpe_ratio": 0.0, "max_drawdown": 0.0, "win_rate": 0.0,
            "total_trades": len(trades), "var_95": 0.0, "var_99": 0.0,
            "equity_curve": equity_curve_from_trades(db, user_id),
        }

    curve = equity_curve_from_trades(db, user_id)
    returns = [
        (curve[i] - curve[i - 1]) / curve[i - 1] if curve[i - 1] else 0.0
        for i in range(1, len(curve))
    ]
    mean_r = sum(returns) / len(returns) if returns else 0.0
    var_r = sum((r - mean_r) ** 2 for r in returns) / len(returns) if returns else 0.0
    std_r = math.sqrt(var_r)
    sharpe = (mean_r / (std_r + 1e-9)) * math.sqrt(252) if returns else 0.0

    peak = curve[0]
    max_dd = 0.0
    for v in curve:
        peak = max(peak, v)
        dd = (peak - v) / peak if peak else 0.0
        max_dd = max(max_dd, dd)

    positions = get_positions_with_pnl(db, user_id)
    winners = sum(1 for p in positions if p["unrealized_pnl"] > 0)
    total_trades = len(trades)
    win_rate = winners / len(positions) if positions else 0.0

    sorted_returns = sorted(returns)
    var_idx = max(0, int(0.05 * len(sorted_returns)) - 1)
    var_95 = sorted_returns[var_idx] if sorted_returns else 0.0
    var99_idx = max(0, int(0.01 * len(sorted_returns)) - 1)
    var_99 = sorted_returns[var99_idx] if sorted_returns else 0.0

    return {
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown": round(max_dd, 4),
        "win_rate": round(win_rate, 3),
        "total_trades": total_trades,
        "var_95": round(var_95, 4),
        "var_99": round(var_99, 4),
        "equity_curve": [round(v, 2) for v in curve],
    }
