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

    Returns comprehensive risk metrics including:
      - Sharpe ratio (annualised, sample-std)
      - Sortino ratio
      - Calmar ratio
      - Omega ratio
      - VaR and CVaR (Expected Shortfall) at 95% and 99%
      - Downside deviation
      - Beta, Alpha, Tracking Error vs SPY benchmark
      - Information Ratio
      - Max drawdown, win rate, total trades, equity curve
    """
    import numpy as np

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
    empty_metrics = {
        "sharpe_ratio": 0.0, "sortino_ratio": 0.0, "calmar_ratio": 0.0,
        "omega_ratio": 0.0, "max_drawdown": 0.0,
        "win_rate": round(wins / closed, 3) if closed else 0.0,
        "total_trades": len(trades),
        "var_95": 0.0, "var_99": 0.0, "cvar_95": 0.0, "cvar_99": 0.0,
        "downside_deviation": 0.0, "volatility": 0.0,
        "beta": 0.0, "alpha": 0.0, "tracking_error": 0.0,
        "information_ratio": 0.0,
        "equity_curve": equity_curve_from_trades(db, user_id),
    }

    if len(trades) < 2:
        return empty_metrics

    curve = equity_curve_from_trades(db, user_id)
    returns = np.array([
        (curve[i] - curve[i - 1]) / curve[i - 1] if curve[i - 1] else 0.0
        for i in range(1, len(curve))
    ])

    if len(returns) < 2:
        return empty_metrics

    mean_r = float(np.mean(returns))
    n = len(returns)

    # ── Sharpe ratio (annualised, sample-std) ──
    std_r = float(np.std(returns, ddof=1))
    sharpe = (mean_r / (std_r + 1e-9)) * math.sqrt(252)

    # ── Sortino ratio ──
    downside = returns[returns < 0]
    downside_dev = float(np.sqrt(np.mean(downside ** 2))) if len(downside) > 0 else 1e-9
    sortino = (mean_r / (downside_dev + 1e-9)) * math.sqrt(252)

    # ── Max drawdown ──
    peak = curve[0]
    max_dd = 0.0
    for v in curve:
        peak = max(peak, v)
        dd = (peak - v) / peak if peak else 0.0
        max_dd = max(max_dd, dd)

    # ── Calmar ratio ──
    ann_return = mean_r * 252
    calmar = ann_return / max_dd if max_dd > 1e-9 else 0.0

    # ── Omega ratio ──
    gains = np.sum(np.maximum(returns, 0))
    losses = np.sum(np.maximum(-returns, 0))
    omega = float(gains / losses) if losses > 1e-9 else 0.0

    # ── VaR and CVaR (Expected Shortfall) ──
    sorted_returns = np.sort(returns)
    var_idx_95 = min(max(0, int(0.05 * n)), n - 1)
    var_idx_99 = min(max(0, int(0.01 * n)), n - 1)
    var_95 = max(0.0, -float(sorted_returns[var_idx_95]))
    var_99 = max(0.0, -float(sorted_returns[var_idx_99]))
    # CVaR = average of losses beyond VaR threshold
    tail_95 = sorted_returns[:var_idx_95 + 1]
    cvar_95 = max(0.0, -float(np.mean(tail_95))) if len(tail_95) > 0 else var_95
    tail_99 = sorted_returns[:var_idx_99 + 1]
    cvar_99 = max(0.0, -float(np.mean(tail_99))) if len(tail_99) > 0 else var_99

    # ── Annualised volatility ──
    volatility = std_r * math.sqrt(252)

    # ── Win rate ──
    total_trades = len(trades)
    win_rate = wins / closed if closed else 0.0

    # ── Beta, Alpha, Tracking Error vs SPY ──
    beta = 0.0
    alpha = 0.0
    tracking_error = 0.0
    information_ratio = 0.0

    try:
        import yfinance as yf_bench
        spy_data = yf_bench.Ticker("SPY").history(period="1y", auto_adjust=True)
        if not spy_data.empty and len(spy_data) > 10:
            spy_close = spy_data["Close"].to_numpy(dtype=float)
            spy_returns = np.diff(spy_close) / spy_close[:-1]
            # Align lengths
            min_len = min(len(returns), len(spy_returns))
            if min_len > 5:
                port_r = returns[-min_len:]
                bench_r = spy_returns[-min_len:]
                # Beta = Cov(Rp, Rm) / Var(Rm)
                cov_matrix = np.cov(port_r, bench_r)
                var_bench = cov_matrix[1, 1]
                if var_bench > 1e-12:
                    beta = float(cov_matrix[0, 1] / var_bench)
                # Alpha = Rp - β * Rm (annualised)
                alpha = float((np.mean(port_r) - beta * np.mean(bench_r)) * 252)
                # Tracking error
                diff = port_r - bench_r
                tracking_error = float(np.std(diff, ddof=1) * math.sqrt(252))
                # Information ratio
                if tracking_error > 1e-9:
                    information_ratio = float(np.mean(diff) * 252 / tracking_error)
    except Exception:
        pass  # Benchmark data unavailable — metrics default to 0

    return {
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "calmar_ratio": round(calmar, 3),
        "omega_ratio": round(omega, 3),
        "max_drawdown": round(max_dd, 4),
        "win_rate": round(win_rate, 3),
        "total_trades": total_trades,
        "var_95": round(var_95, 4),
        "var_99": round(var_99, 4),
        "cvar_95": round(cvar_95, 4),
        "cvar_99": round(cvar_99, 4),
        "downside_deviation": round(downside_dev, 6),
        "volatility": round(volatility, 4),
        "beta": round(beta, 4),
        "alpha": round(alpha, 4),
        "tracking_error": round(tracking_error, 4),
        "information_ratio": round(information_ratio, 3),
        "equity_curve": [round(v, 2) for v in curve],
    }
