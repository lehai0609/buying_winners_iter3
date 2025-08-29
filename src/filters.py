from __future__ import annotations
from typing import Literal
import pandas as pd
import numpy as np

GridMode = Literal["union", "vnindex"]


def _ensure_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.MultiIndex) and list(df.index.names) == ["date", "ticker"]:
        return df.sort_index()
    if "date" not in df.columns or "ticker" not in df.columns:
        raise ValueError("expected columns 'date' and 'ticker' or MultiIndex [date,ticker]")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    return df.set_index(["date", "ticker"]).sort_index()


def compute_turnover(df: pd.DataFrame) -> pd.Series:
    df = _ensure_multiindex(df)
    for c in ["close", "volume"]:
        if c not in df.columns:
            raise ValueError(f"missing required column '{c}'")
    close = pd.to_numeric(df["close"], errors="coerce")
    vol = pd.to_numeric(df["volume"], errors="coerce")
    return (close * vol).astype(float)


def rolling_adv(df: pd.DataFrame, window: int = 126, min_periods: int = 60) -> pd.Series:
    df = _ensure_multiindex(df)
    # Use compute_turnover with missing rows treated as zeros by reindexing in caller if desired.
    trn = compute_turnover(df)
    # Group rolling by ticker
    return (
        trn.groupby(level="ticker").rolling(window=window, min_periods=min_periods).mean().droplevel(0)
    )


def _grid_dates_union(df: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted(df.index.get_level_values("date").unique()))


def _grid_dates_vnindex(indices_df: pd.DataFrame) -> pd.DatetimeIndex:
    # Expect columns [date, index, close]; pick VNINDEX rows
    if indices_df is None:
        raise ValueError("indices_df must be provided when calendar='vnindex'")
    if not {"date", "index", "close"}.issubset(indices_df.columns):
        raise ValueError("indices_df must have columns [date, index, close]")
    sub = indices_df[indices_df["index"] == "VNINDEX"]
    return pd.DatetimeIndex(sorted(pd.to_datetime(sub["date"]).unique()))


def eligibility_flags(
    df: pd.DataFrame,
    lookback_days: int = 126,
    min_history_days: int = 126,
    min_price_vnd: float = 1000.0,
    min_adv_vnd: float = 100_000_000.0,
    max_nontrading_days: int = 15,
    calendar: GridMode = "union",
    indices_df: pd.DataFrame | None = None,
    price_scale: float = 1.0,
    anomalies_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute daily eligibility flags using Vietnam-specific rules.

    Returns a DataFrame indexed by [date, ticker] with component flags and diagnostics.
    """
    df = _ensure_multiindex(df)
    required = ["open", "high", "low", "close", "volume"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"missing required column '{c}'")

    # Build grid dates
    if calendar == "vnindex":
        grid = _grid_dates_vnindex(indices_df)
    else:
        grid = _grid_dates_union(df)

    tickers = df.index.get_level_values("ticker").unique()
    full_idx = pd.MultiIndex.from_product([grid, tickers], names=["date", "ticker"])

    # Prepare base frame on the grid
    base = df.reindex(full_idx).sort_index()
    # Forward-fill close by ticker to get last known price at date t, then scale to desired currency units
    base["close_ffill"] = base.groupby(level="ticker")["close"].ffill()
    base["close_scaled"] = base["close_ffill"] * float(price_scale)
    # Volume: treat missing as zero (counts as non-trading)
    base["volume_filled"] = pd.to_numeric(base["volume"], errors="coerce").fillna(0.0)
    # Presence flag (had an original row for this date)
    present = df.assign(_present=1)['_present'].reindex(full_idx).fillna(0.0)

    # Turnover and ADV (missing rows -> zero turnover)
    base["turnover_vnd"] = (base["close_scaled"].fillna(0.0) * base["volume_filled"]).astype(float)
    adv = (
        base["turnover_vnd"]
        .groupby(level="ticker")
        .rolling(window=lookback_days, min_periods=min(lookback_days, max(1, min_history_days // 2)))
        .mean()
        .droplevel(0)
    )

    # Days with any history in window (presence), and traded days (volume>0)
    days_hist = (
        present.groupby(level="ticker")
        .rolling(window=lookback_days, min_periods=1)
        .sum()
        .droplevel(0)
    )
    days_traded = (
        (base["volume_filled"] > 0)
        .groupby(level="ticker")
        .rolling(window=lookback_days, min_periods=1)
        .sum()
        .droplevel(0)
    )
    nontrading_days = (lookback_days - days_traded).clip(lower=0)

    # Component flags
    price_ok = (base["close_scaled"] >= float(min_price_vnd)).fillna(False)
    adv_ok = (adv >= float(min_adv_vnd)).fillna(False)
    min_history_ok = (days_hist >= float(min_history_days))
    nontrading_ok = (nontrading_days <= float(max_nontrading_days))

    # Quality: default True; if anomalies provided, fail if any qualifying rule within window
    if anomalies_df is not None and len(anomalies_df) > 0:
        a = anomalies_df.copy()
        if not {"date", "ticker", "rule"}.issubset(a.columns):
            # tolerate minimal schema
            raise ValueError("anomalies_df must contain columns ['date','ticker','rule']")
        a["date"] = pd.to_datetime(a["date"], errors="coerce")
        a = a.dropna(subset=["date", "ticker"])
        bad_rules = {"ohlc_ordering_violation", "price_non_positive"}
        a["bad"] = a["rule"].astype(str).isin(bad_rules)
        a = a[a["bad"]]
        if len(a) > 0:
            # Reindex then handle the 'bad' column explicitly to avoid DataFrame-wide fillna downcasting warnings
            af = a.set_index(["date", "ticker"])[["bad"]].reindex(full_idx)
            af["bad"] = af["bad"].astype("boolean").fillna(False).astype(bool)
            # Rolling any within window => quality failure at t
            bad_any = (
                af["bad"].groupby(level="ticker").rolling(window=lookback_days, min_periods=1).max().droplevel(0)
            )
            # Ensure boolean dtype before invert
            bad_any = bad_any.fillna(False).astype(bool)
            quality_ok = (~bad_any).reindex(full_idx)
        else:
            quality_ok = pd.Series(True, index=full_idx)
    else:
        quality_ok = pd.Series(True, index=full_idx)

    # Consolidate output
    out = pd.DataFrame(index=full_idx)
    out["min_history_ok"] = min_history_ok.reindex(full_idx, fill_value=False).astype(bool)
    out["price_ok"] = price_ok.reindex(full_idx, fill_value=False).astype(bool)
    out["adv_ok"] = adv_ok.reindex(full_idx, fill_value=False).astype(bool)
    out["nontrading_ok"] = nontrading_ok.reindex(full_idx, fill_value=False).astype(bool)
    out["quality_ok"] = quality_ok.reindex(full_idx).fillna(True).astype(bool)
    out["days_history"] = days_hist.reindex(full_idx).fillna(0.0).astype(float)
    out["days_traded_in_window"] = days_traded.reindex(full_idx).fillna(0.0).astype(float)
    out["nontrading_days"] = nontrading_days.reindex(full_idx).fillna(float(lookback_days)).astype(float)
    out["adv_vnd"] = adv.reindex(full_idx).fillna(0.0).astype(float)
    out["eligible"] = (
        out["min_history_ok"]
        & out["price_ok"]
        & out["adv_ok"]
        & out["nontrading_ok"]
        & out["quality_ok"]
    )
    return out.sort_index()


def monthly_universe(flags: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(flags.index, pd.MultiIndex) or list(flags.index.names) != ["date", "ticker"]:
        raise ValueError("flags must be indexed by [date, ticker]")
    df = flags.copy()
    idx_dates = df.index.get_level_values("date")
    # Month-end per calendar present in flags
    months = idx_dates.to_period("M")
    month_end_by_period = pd.Series(idx_dates).groupby(months).max()
    month_end_set = set(month_end_by_period.values)
    sel = idx_dates.isin(month_end_set)
    sub = df.loc[sel, ["eligible"]].reset_index()
    sub = sub.rename(columns={"date": "month_end"})
    sub = sub[["month_end", "ticker", "eligible"]].sort_values(["month_end", "ticker"]).reset_index(drop=True)
    return sub


def apply_eligibility(df: pd.DataFrame, monthly_flags: pd.DataFrame) -> pd.DataFrame:
    """Filter df to rows for which the ticker is eligible at that month's formation date.

    This implementation keeps only rows where ticker is marked eligible at the month-end corresponding to the row date.
    """
    df = _ensure_multiindex(df)
    if set(["month_end", "ticker", "eligible"]) - set(monthly_flags.columns):
        raise ValueError("monthly_flags must have columns ['month_end','ticker','eligible']")
    # Map each row date to its calendar month-end present in monthly_flags
    # Build a lookup of month -> month_end date
    me = monthly_flags.copy()
    me["month"] = pd.to_datetime(me["month_end"]).dt.to_period("M")
    me_idx = me.set_index(["month", "ticker"]).sort_index()

    df2 = df.copy()
    months = df2.index.get_level_values("date").to_period("M")
    join = pd.DataFrame({
        "month": months,
        "ticker": df2.index.get_level_values("ticker"),
    }, index=df2.index)
    # Align eligibility
    elig = me_idx[["eligible"]].reindex(join.set_index(["month", "ticker"]).index).reset_index(drop=True)
    elig.index = df2.index
    mask = elig["eligible"].fillna(False).astype(bool)
    return df2.loc[mask]
