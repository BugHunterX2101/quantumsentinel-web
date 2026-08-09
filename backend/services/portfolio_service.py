"""QuantumSentinel — Portfolio analytics: positions, P&L, VaR, Sharpe, drawdown."""
import math
from sqlalchemy.orm import Session
from sqlalchemy import select

from .. import models
from .trading_service import get_last_price


def recompute_positions(db: Session, user_id: str) -> None:
    """Rebuild the `positions` materialised view from filled trades.

    Event-sourcing pattern: trades are the source of truth; positions are
    a derived projection rebuilt from scratch on every fill.
    """
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
            # Clamp sell qty to available holding — prevents negative book on any
            # stale or concurrent fill that bypassed the HTTP-layer oversell guard.
            sell_qty = min(qty, b["qty"])
            b["realized"] += (price - avg_cost) * sell_qty
            b["qty"] -= sell_qty
            b["cost"] = avg_cost * max(0.0, b["qty"])

    # Replace existing position rows — SQLAlchemy 2.0 execute style
    db.execute(
        models.Position.__table__.delete().where(models.Position.user_id == user_id)
    )
    for asset, b in book.items():
        if b["qty"] <= 0:
            continue  # flat position — no row needed
        avg_entry = b["cost"] / b["qty"]
        db.add(models.Position(
            user_id=user_id, asset=asset, quantity=round(b["qty"], 6),
            avg_entry_price=round(avg_entry, 4), realized_pnl=round(b["realized"], 2),
        ))
    db.commit()


def get_positions_with_pnl(db: Session, user_id: str) -> list[dict]:
    """Return open positions enriched with live mark-to-market data."""
    positions = db.execute(
        select(models.Position).where(
            models.Position.user_id == user_id,
            models.Position.quantity > 0,
        )
    ).scalars().all()
    out = []
    for p in positions:
        current_price = get_last_price(p.asset)
        unrealized = (current_price - float(p.avg_entry_price)) * float(p.quantity)
        out.append({
            "asset": p.asset,
            "quantity": float(p.quantity),
            "avg_entry_price": float(p.avg_entry_price),
            "current_price": current_price,
            "unrealized_pnl": round(unrealized, 2),
            "realized_pnl": float(p.realized_pnl),
            "market_value": round(current_price * float(p.quantity), 2),
        })
    return out


def equity_curve_from_trades(
    db: Session, user_id: str, starting_capital: float = 100_000.0
) -> list[float]:
    """Build a mark-to-market equity curve from filled trades.

    FIX: Batch-fetches live prices for each unique asset ONCE before the
    trade loop instead of calling get_last_price() inside the loop.
    The old approach caused O(N_trades × N_assets) yfinance calls per
    risk_metrics request.
    """
    trades = db.execute(
        select(models.Trade).where(
            models.Trade.user_id == user_id, models.Trade.status == "FILLED"
        ).order_by(models.Trade.filled_at)
    ).scalars().all()

    # One live-price fetch per unique asset — not per trade event
    unique_assets = {t.asset for t in trades}
    current_prices: dict[str, float] = {
        asset: get_last_price(asset) for asset in unique_assets
    }

    equity = starting_capital
    curve = [equity]
    book: dict[str, float] = {}

    for t in trades:
        qty, price = float(t.quantity), float(t.filled_price or 0)
        notional = qty * price
        if t.side == "buy":
            equity -= notional
            book[t.asset] = book.get(t.asset, 0.0) + qty
        else:
            equity += notional
            book[t.asset] = book.get(t.asset, 0.0) - qty

        # Mark open positions to market using the pre-fetched current prices
        open_value = sum(
            held_qty * current_prices.get(asset, 0.0)
            for asset, held_qty in book.items()
            if held_qty > 0
        )
        curve.append(equity + open_value)
    return curve


def risk_metrics(db: Session, user_id: str) -> dict:
    """Compute portfolio risk metrics from the user's trade history.

    Returns Sharpe ratio (annualised, sample-std), max drawdown, win rate,
    VaR at 95% and 99%, total trades, and the full equity curve.
    """
    trades = db.execute(
        select(models.Trade).where(
            models.Trade.user_id == user_id, models.Trade.status == "FILLED"
        ).order_by(models.Trade.filled_at)
    ).scalars().all()

    def closed_trade_results() -> tuple[int, int]:
        """Count round-trip wins and total closed trades."""
        book: dict[str, tuple[float, float]] = {}  # asset -> (qty, total_cost)
        wins = closed = 0
        for trade in trades:
            qty, price = float(trade.quantity), float(trade.filled_price or 0)
            current_qty, cost = book.get(trade.asset, (0.0, 0.0))
            if trade.side == "buy":
                book[trade.asset] = (current_qty + qty, cost + qty * price)
            elif current_qty > 0:
                avg = cost / current_qty if current_qty else 0.0
                closed += 1
                wins += int(price > avg)
                remaining = max(0.0, current_qty - qty)
                book[trade.asset] = (remaining, avg * remaining)
        return wins, closed

    wins, closed = closed_trade_results()

    if len(trades) < 2:
        return {
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": round(wins / closed, 3) if closed else 0.0,
            "total_trades": len(trades),
            "var_95": 0.0,
            "var_99": 0.0,
            "equity_curve": equity_curve_from_trades(db, user_id),
        }

    curve = equity_curve_from_trades(db, user_id)
    returns = [
        (curve[i] - curve[i - 1]) / curve[i - 1] if curve[i - 1] else 0.0
        for i in range(1, len(curve))
    ]
    mean_r = sum(returns) / len(returns) if returns else 0.0

    # FIX: sample standard deviation (N-1 denominator) — industry standard for
    # Sharpe ratio.  Population std (N) systematically understates volatility
    # and inflates Sharpe, especially with short trade histories.
    n = len(returns)
    var_r = (
        sum((r - mean_r) ** 2 for r in returns) / max(1, n - 1)
        if n > 1 else 0.0
    )
    std_r = math.sqrt(var_r)
    sharpe = (mean_r / (std_r + 1e-9)) * math.sqrt(252) if returns else 0.0

    peak = curve[0]
    max_dd = 0.0
    for v in curve:
        peak = max(peak, v)
        dd = (peak - v) / peak if peak else 0.0
        max_dd = max(max_dd, dd)

    total_trades = len(trades)
    win_rate = wins / closed if closed else 0.0

    sorted_returns = sorted(returns)
    # FIX: Empirical VaR percentile — int(0.05 * N) is the correct index.
    # The previous code used int(0.05 * N) - 1 which was off by one and
    # produced a less conservative (understated) risk estimate.
    var_idx_95 = min(max(0, int(0.05 * len(sorted_returns))), len(sorted_returns) - 1)
    var_idx_99 = min(max(0, int(0.01 * len(sorted_returns))), len(sorted_returns) - 1)
    var_95 = max(0.0, -sorted_returns[var_idx_95]) if sorted_returns else 0.0
    var_99 = max(0.0, -sorted_returns[var_idx_99]) if sorted_returns else 0.0

    return {
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown": round(max_dd, 4),
        "win_rate": round(win_rate, 3),
        "total_trades": total_trades,
        "var_95": round(var_95, 4),
        "var_99": round(var_99, 4),
        "equity_curve": [round(v, 2) for v in curve],
    }
