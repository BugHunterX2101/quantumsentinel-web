"""QuantumSentinel — date-aligned historical price panels.

Every multi-asset calculation must compare prices from the same date.
yfinance returns tickers downloaded together on a shared (union) date index
with NaN on days a ticker did not trade — weekends for equities next to
crypto, and each exchange's own holidays. Dropping NaNs per ticker and then
truncating the arrays to a common length pairs prices from different days
(months apart by the end of a one-year equity/crypto download).

``aligned_panel`` aligns on the calendar instead: it keeps only the dates on
which every usable ticker has a valid close, so row ``i`` of every column is
the same trading day.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_VOLUME = 1_000_000.0


@dataclass(frozen=True)
class PricePanel:
    """Close/High/Low/Volume on one shared calendar (row i = same date)."""
    dates: pd.DatetimeIndex
    close: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    volume: pd.DataFrame

    @property
    def tickers(self) -> list[str]:
        return list(self.close.columns)

    def __len__(self) -> int:
        return len(self.dates)

    def date_strings(self) -> list[str]:
        return [d.strftime("%Y-%m-%d") for d in self.dates]

    def asset_data(self) -> dict[str, dict[str, np.ndarray]]:
        """Per-ticker float arrays in the shape the backtest engines consume."""
        return {t: {"close": self.close[t].to_numpy(dtype=float),
                    "high": self.high[t].to_numpy(dtype=float),
                    "low": self.low[t].to_numpy(dtype=float),
                    "volume": self.volume[t].to_numpy(dtype=float)}
                for t in self.tickers}


def field_frame(data: pd.DataFrame | None, field: str, tickers: list[str]) -> pd.DataFrame:
    """One numeric column per requested ticker for ``field`` (NaN where absent).

    Handles both yfinance layouts: (field, ticker) MultiIndex columns for a
    multi-symbol download, and flat columns, which only ever describe a
    single symbol.
    """
    if data is None or data.empty:
        return pd.DataFrame()
    wanted = list(dict.fromkeys(tickers))
    columns: dict[str, pd.Series] = {}
    if isinstance(data.columns, pd.MultiIndex):
        if field in data.columns.get_level_values(0):
            block = data[field]
            for t in wanted:
                if t in block.columns:
                    col = block[t]
                    columns[t] = col.iloc[:, 0] if isinstance(col, pd.DataFrame) else col
    elif len(wanted) == 1 and field in data.columns:
        col = data[field]
        columns[wanted[0]] = col.iloc[:, 0] if isinstance(col, pd.DataFrame) else col
    frame = pd.DataFrame(columns, index=data.index)
    return frame.apply(pd.to_numeric, errors="coerce")


def aligned_panel(data: pd.DataFrame | None, tickers: list[str], min_rows: int = 1) -> PricePanel:
    """Align ``tickers`` on the dates where every usable one has a valid close.

    A ticker is usable when it has at least ``min_rows`` valid closes; the
    rest are dropped (as before) rather than shrinking everyone's history.
    Valid means finite and strictly positive. High/Low fall back to the
    close and Volume to the previous observation (then DEFAULT_VOLUME) on
    the rare aligned day where only the close was published.
    """
    close = field_frame(data, "Close", tickers)
    close = close.where(np.isfinite(close) & (close > 0))
    usable = [t for t in close.columns if int(close[t].notna().sum()) >= min_rows]
    close = close[usable].dropna(how="any") if usable else pd.DataFrame()
    dates = pd.DatetimeIndex(close.index)

    def on_calendar(field: str) -> pd.DataFrame:
        frame = field_frame(data, field, usable)
        return frame.reindex(index=dates, columns=usable) if usable else pd.DataFrame(index=dates)

    high = on_calendar("High").where(lambda f: f > 0).fillna(close)
    low = on_calendar("Low").where(lambda f: f > 0).fillna(close)
    volume = on_calendar("Volume").where(lambda f: f >= 0).ffill().fillna(DEFAULT_VOLUME)
    return PricePanel(dates=dates, close=close, high=high, low=low, volume=volume)


def series_on_calendar(data: pd.DataFrame | None, ticker: str,
                       dates: pd.DatetimeIndex) -> np.ndarray | None:
    """``ticker``'s close on each of ``dates``, carrying its last close forward.

    Used for a benchmark whose market may be closed on some trading days of
    the asset calendar: a closed market's value is its last close. Returns
    None if the ticker has no close on or before the first date.
    """
    frame = field_frame(data, "Close", [ticker])
    if ticker not in frame.columns or not len(dates):
        return None
    series = frame[ticker].where(lambda s: np.isfinite(s) & (s > 0)).dropna()
    if series.empty:
        return None
    aligned = series.reindex(series.index.union(dates)).ffill().reindex(dates)
    if aligned.isna().any():
        return None
    return aligned.to_numpy(dtype=float)
