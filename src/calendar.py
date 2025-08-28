from __future__ import annotations
from typing import Literal
import pandas as pd

GridMode = Literal["union", "vnindex"]


def _ensure_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.MultiIndex) and list(df.index.names) == ["date", "ticker"]:
        return df.sort_index()
    if "date" not in df.columns or "ticker" not in df.columns:
        raise ValueError("expected columns 'date' and 'ticker' or MultiIndex [date,ticker]")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    return df.set_index(["date", "ticker"]).sort_index()


def build_trading_grid(
    df: pd.DataFrame,
    calendar: GridMode = "union",
    indices_df: pd.DataFrame | None = None,
) -> pd.DatetimeIndex:
    """Construct a trading-date grid from OHLCV or VNINDEX.

    - union: unique dates from df index
    - vnindex: unique dates for VNINDEX from indices_df
    """
    df = _ensure_multiindex(df)
    if calendar == "vnindex":
        if indices_df is None:
            raise ValueError("indices_df must be provided when calendar='vnindex'")
        req = {"date", "index", "close"}
        if not req.issubset(indices_df.columns):
            raise ValueError("indices_df must have columns ['date','index','close']")
        sub = indices_df[indices_df["index"] == "VNINDEX"].copy()
        dates = pd.to_datetime(sub["date"], errors="raise").unique()
        return pd.DatetimeIndex(sorted(dates))
    else:
        dates = df.index.get_level_values("date").unique()
        return pd.DatetimeIndex(sorted(dates))


def month_ends(grid: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return last available trading day per calendar month on the grid."""
    if not isinstance(grid, pd.DatetimeIndex):
        grid = pd.DatetimeIndex(pd.to_datetime(grid))
    if len(grid) == 0:
        return pd.DatetimeIndex([])
    months = grid.to_period("M")
    end_by_month = pd.Series(grid).groupby(months).max()
    return pd.DatetimeIndex(end_by_month.values)


def nth_trading_day_in_month(grid: pd.DatetimeIndex, n: int) -> pd.Series:
    """Map each month (Period[M]) to its nth trading day on the grid (or NaT)."""
    if n <= 0:
        raise ValueError("n must be >= 1")
    if not isinstance(grid, pd.DatetimeIndex):
        grid = pd.DatetimeIndex(pd.to_datetime(grid))
    if len(grid) == 0:
        return pd.Series(dtype="datetime64[ns]")
    g = pd.Series(grid)
    months = grid.to_period("M")
    # For each month, pick nth if exists
    def _pick(s: pd.Series) -> pd.Timestamp | pd.NaT:
        if len(s) >= n:
            return pd.to_datetime(s.iloc[n - 1])
        return pd.NaT

    out = g.groupby(months).apply(_pick)
    out.index = out.index.astype("period[M]")
    return out


def shift_trading_days(
    grid: pd.DatetimeIndex, dates: pd.DatetimeIndex | pd.Series, k: int
) -> pd.DatetimeIndex | pd.Series:
    """Shift each date by k trading days on the grid.

    - If date not in grid: result is NaT
    - If shift goes beyond ends: clip to first/last date
    """
    if not isinstance(grid, pd.DatetimeIndex):
        grid = pd.DatetimeIndex(pd.to_datetime(grid))
    pos = pd.Series(range(len(grid)), index=grid)

    def _shift_one(dt: pd.Timestamp) -> pd.Timestamp | pd.NaT:
        if dt not in pos.index:
            return pd.NaT
        i = pos.loc[dt] + k
        i = max(0, min(int(i), len(grid) - 1))
        return grid[i]

    if isinstance(dates, pd.Series):
        return dates.apply(lambda x: _shift_one(pd.to_datetime(x)))
    else:
        return pd.DatetimeIndex([_shift_one(pd.to_datetime(x)) for x in dates])


def align_to_grid(
    df: pd.DataFrame, grid: pd.DatetimeIndex, how: Literal["inner", "left"] = "left"
) -> pd.DataFrame:
    """Align MultiIndex [date,ticker] data to the provided grid.

    - left: full product of grid × unique tickers, reindexed
    - inner: keep only rows whose date exists in grid
    """
    df = _ensure_multiindex(df)
    if not isinstance(grid, pd.DatetimeIndex):
        grid = pd.DatetimeIndex(pd.to_datetime(grid))
    tickers = df.index.get_level_values("ticker").unique()
    if how == "left":
        full_idx = pd.MultiIndex.from_product([grid, tickers], names=["date", "ticker"])
        return df.reindex(full_idx).sort_index()
    elif how == "inner":
        mask = df.index.get_level_values("date").isin(grid)
        return df.loc[mask].sort_index()
    else:
        raise ValueError("how must be 'inner' or 'left'")

