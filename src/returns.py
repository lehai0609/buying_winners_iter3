from __future__ import annotations
from typing import Literal
import numpy as np
import pandas as pd

from .calendar import build_trading_grid, month_ends, align_to_grid, shift_trading_days


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


def cum_return_skip(
    ret_d: pd.Series | pd.DataFrame,
    J_months: int,
    skip_days: int = 5,
    calendar: Literal["union", "vnindex"] = "union",
    indices_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute cumulative return over the last J months excluding the most recent `skip_days`.

    Inputs
    - ret_d: MultiIndex [date,ticker] of simple daily returns. Accepts a Series (name 'ret_1d') or a
      single-column DataFrame with column 'ret_1d'. Use ``daily_simple_returns`` to construct.
    - J_months: Lookback window in calendar months.
    - skip_days: Number of trading days to skip prior to the formation date (default 5).
    - calendar / indices_df: Used only to derive month-end formation dates when "vnindex" is requested.

    Output
    - DataFrame with columns ["month_end","ticker","window_ret","n_days_used"].

    Logic
    - For each formation month-end t on the chosen grid, define e_excl = shift(t, -skip_days).
      Include daily returns with dates d such that s < d < e_excl, where s is the month-end exactly
      J months before t. The product of (1+ret_d) over that interval minus 1 is returned as ``window_ret``.
    """
    # Normalize input to Series named 'ret_1d'
    if isinstance(ret_d, pd.DataFrame):
        if "ret_1d" not in ret_d.columns:
            raise ValueError("ret_d DataFrame must have a 'ret_1d' column")
        series = ret_d["ret_1d"].copy()
    else:
        series = ret_d.copy()
    if series.name is None:
        series.name = "ret_1d"

    # Ensure MultiIndex [date,ticker]
    if not isinstance(series.index, pd.MultiIndex) or list(series.index.names) != ["date", "ticker"]:
        raise ValueError("ret_d must be indexed by MultiIndex ['date','ticker']")

    # Build grid and month-ends for formation schedule
    # We can reuse build_trading_grid by passing a dummy frame with this index
    dummy = pd.DataFrame(index=series.index)
    grid = build_trading_grid(dummy, calendar=calendar, indices_df=indices_df)
    me = month_ends(grid)
    if J_months < 1 or skip_days < 0:
        raise ValueError("J_months must be >=1 and skip_days >= 0")

    # Map month period to its month-end trading day on the grid
    me_series = pd.Series(me, index=me.to_period("M"))

    out_rows: list[dict] = []
    # Pre-split by ticker for speed
    by_ticker = {tk: s.sort_index() for tk, s in series.groupby(level="ticker")}

    for t in me:
        # Locate the month-end J months earlier
        t_per = t.to_period("M")
        s_per = t_per - J_months
        if s_per not in me_series.index:
            continue  # insufficient history on calendar
        s_anchor = pd.to_datetime(me_series.loc[s_per])
        # Exclude last `skip_days` via exclusive end marker
        e_excl = shift_trading_days(grid, pd.DatetimeIndex([t]), -int(skip_days))[0]
        if pd.isna(e_excl):
            continue
        # Construct open interval (s_anchor, e_excl)
        for tk, s in by_ticker.items():
            # Slice the ticker's daily returns strictly within the bounds
            # Note: index level 0 is date
            s_tk = s.xs(tk, level="ticker")
            mask = (s_tk.index > s_anchor) & (s_tk.index < e_excl)
            window = s_tk.loc[mask].dropna()
            if len(window) == 0:
                win_ret = np.nan
                n_days = 0
            else:
                # Product of (1+ret_d) minus 1
                win_ret = float(np.prod(1.0 + window.values) - 1.0)
                n_days = int(len(window))
            out_rows.append({
                "month_end": t,
                "ticker": tk,
                "window_ret": win_ret,
                "n_days_used": n_days,
            })

    out = pd.DataFrame(out_rows)
    if len(out) == 0:
        return pd.DataFrame(columns=["month_end", "ticker", "window_ret", "n_days_used"])
    out["month_end"] = pd.to_datetime(out["month_end"])  # ensure dtype
    return out.sort_values(["month_end", "ticker"]).reset_index(drop=True)
