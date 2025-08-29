from __future__ import annotations
from typing import Literal
import numpy as np
import pandas as pd

from .calendar import build_trading_grid, month_ends, align_to_grid


def _ensure_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.MultiIndex) and list(df.index.names) == ["date", "ticker"]:
        return df.sort_index()
    if "date" not in df.columns or "ticker" not in df.columns:
        raise ValueError("expected columns 'date' and 'ticker' or MultiIndex [date,ticker]")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    return df.set_index(["date", "ticker"]).sort_index()


def daily_simple_returns(df: pd.DataFrame, price_col: str = "close") -> pd.Series:
    """Per-ticker simple daily returns: close/shift(close) - 1.

    Returns a MultiIndex Series aligned to df index with name 'ret_1d'.
    """
    df = _ensure_multiindex(df)
    if price_col not in df.columns:
        raise ValueError(f"missing required price column '{price_col}'")
    px = pd.to_numeric(df[price_col], errors="coerce")
    # Avoid pandas FutureWarning by disabling implicit ffill within pct_change
    ret = px.groupby(level="ticker").pct_change(fill_method=None)
    ret.name = "ret_1d"
    return ret


def daily_log_returns(df: pd.DataFrame, price_col: str = "close") -> pd.Series:
    """Per-ticker log returns: log(close/shift(close))."""
    df = _ensure_multiindex(df)
    if price_col not in df.columns:
        raise ValueError(f"missing required price column '{price_col}'")
    px = pd.to_numeric(df[price_col], errors="coerce")
    grp = px.groupby(level="ticker")
    log_ret = np.log(grp.shift(-0) / grp.shift(1))
    log_ret.name = "ret_log_1d"
    return log_ret


def forward_returns(
    df: pd.DataFrame, horizons: list[int] | tuple[int, ...] = (1, 5, 21), price_col: str = "close"
) -> pd.DataFrame:
    """Evaluation-only forward simple returns for each horizon h in days.

    Columns: fwd_ret_{h}d. Index: MultiIndex [date,ticker].
    """
    df = _ensure_multiindex(df)
    if price_col not in df.columns:
        raise ValueError(f"missing required price column '{price_col}'")
    px = pd.to_numeric(df[price_col], errors="coerce")
    by_tk = px.groupby(level="ticker")
    out = pd.DataFrame(index=df.index)
    for h in horizons:
        fwd = by_tk.shift(-h) / px - 1.0
        out[f"fwd_ret_{int(h)}d"] = fwd
    return out.sort_index()


def monthly_returns(
    df: pd.DataFrame,
    calendar: Literal["union", "vnindex"] = "union",
    indices_df: pd.DataFrame | None = None,
    price_col: str = "close",
) -> pd.DataFrame:
    """Compute calendar-aware month-end close-to-close returns per ticker.

    Returns DataFrame with columns [month_end, ticker, ret_1m]. First month per
    ticker is NaN. Uses last observed close on or before each month-end even if
    that prior date is not on the chosen grid (as-of sampling).
    """
    df = _ensure_multiindex(df)
    if price_col not in df.columns:
        raise ValueError(f"missing required price column '{price_col}'")
    # Build month-ends from requested calendar
    grid = build_trading_grid(df, calendar=calendar, indices_df=indices_df)
    me = month_ends(grid)

    tickers = df.index.get_level_values("ticker").unique()
    records = []
    for tk in tickers:
        s = pd.to_numeric(df.xs(tk, level="ticker")[price_col], errors="coerce").sort_index()
        # Union original dates with month-ends, ffill, then pick month-ends
        s2 = s.reindex(s.index.union(me)).sort_index().ffill().reindex(me)
        r = s2.pct_change()
        rec = pd.DataFrame({"month_end": me, "ticker": tk, "ret_1m": r.values})
        records.append(rec)
    out = pd.concat(records, axis=0, ignore_index=True)
    out = out.sort_values(["month_end", "ticker"]).reset_index(drop=True)
    return out


def eligible_monthly_returns(
    df: pd.DataFrame,
    monthly_flags: pd.DataFrame,
    calendar: Literal["union", "vnindex"] = "union",
    indices_df: pd.DataFrame | None = None,
    price_col: str = "close",
) -> pd.DataFrame:
    """Filter monthly returns by M2 monthly eligibility flags.

    Expects monthly_flags with columns ['month_end','ticker','eligible'].
    """
    req = {"month_end", "ticker", "eligible"}
    if req - set(monthly_flags.columns):
        raise ValueError("monthly_flags must have columns ['month_end','ticker','eligible']")
    mret = monthly_returns(df, calendar=calendar, indices_df=indices_df, price_col=price_col)
    merged = mret.merge(
        monthly_flags[["month_end", "ticker", "eligible"]], on=["month_end", "ticker"], how="left"
    )
    merged = merged[merged["eligible"].fillna(False).astype(bool)].drop(columns=["eligible"]).reset_index(drop=True)
    return merged
