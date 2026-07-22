"""Deterministic, long-only moving-average backtester for paper strategies."""
import math
import numpy as np
import yfinance as yf


def run_moving_average_backtest(asset: str, fast_window: int, slow_window: int,
                                period: str, initial_capital: float = 100_000.0) -> dict:
    data = yf.Ticker(asset).history(period=period, auto_adjust=True)
    if data.empty or len(data) <= slow_window + 2:
        raise ValueError("not enough historical data to run this backtest")
    close = data["Close"].astype(float)
    fast = close.rolling(fast_window).mean()
    slow = close.rolling(slow_window).mean()
    position, capital, shares, entries, wins, trades = False, initial_capital, 0.0, [], 0, 0
    curve = []
    for index in range(slow_window, len(close)):
        price = float(close.iloc[index])
        was_above = fast.iloc[index - 1] > slow.iloc[index - 1]
        is_above = fast.iloc[index] > slow.iloc[index]
        if not position and not was_above and is_above:
            shares, capital, position = capital / price, 0.0, True
            entries.append(price)
            trades += 1
        elif position and was_above and not is_above:
            proceeds = shares * price
            wins += int(proceeds > shares * entries[-1])
            capital, shares, position = proceeds, 0.0, False
            trades += 1
        curve.append(capital + shares * price)
    if position:
        final_capital = capital + shares * float(close.iloc[-1])
        wins += int(float(close.iloc[-1]) > entries[-1])
    else:
        final_capital = capital
    returns = np.diff(curve) / np.maximum(np.asarray(curve[:-1]), 1e-9) if len(curve) > 1 else np.array([])
    sharpe = float(returns.mean() / (returns.std() + 1e-9) * math.sqrt(252)) if len(returns) else 0.0
    peak, max_drawdown = initial_capital, 0.0
    for value in curve:
        peak = max(peak, value)
        max_drawdown = max(max_drawdown, (peak - value) / peak)
    return {
        "asset": asset, "period": period, "fast_window": fast_window, "slow_window": slow_window,
        "initial_capital": initial_capital, "final_capital": round(final_capital, 2),
        "total_return": round(final_capital / initial_capital - 1, 4),
        "sharpe_ratio": round(sharpe, 3), "max_drawdown": round(max_drawdown, 4),
        "total_trades": trades, "win_rate": round(wins / max(1, math.ceil(trades / 2)), 3),
        "equity_curve": [round(float(v), 2) for v in curve[::max(1, len(curve) // 100)]],
    }
