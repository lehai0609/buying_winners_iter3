from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

from .calendar import build_trading_grid, month_ends, shift_trading_days
from .data_io import load_indices


SelectionMode = Literal["top_decile", "top_quantile"]


@dataclass
class PortfolioConfig:
    k_months: int = 6
    selection: SelectionMode = "top_decile"
    long_decile: int = 10
    top_quantile: float = 0.10
    renormalize_within_cohort: bool = True
    redistribute_across_cohorts: bool = False
    exclude_on_missing_price: bool = True
    calendar: Literal["union", "vnindex"] | str = "union"
    # Optional constraints (applied at trade generation stage)
    max_weight_per_name: Optional[float] = None
    turnover_cap: Optional[float] = None


def _cfg_from_dict(raw: dict | None) -> PortfolioConfig:
    """Parse a loose cfg dict (typically loaded from YAML) into PortfolioConfig.

    Falls back to safe defaults if keys are missing.
    """
    raw = raw or {}
    por = (raw.get("portfolio", {}) or {})
    k = int(por.get("k_months", 6))
    sel = por.get("selection", "top_decile")
    if sel not in ("top_decile", "top_quantile"):
        sel = "top_decile"
    dec = int(por.get("long_decile", 10))
    tq = float(por.get("top_quantile", 0.10))
    ren = bool(por.get("renormalize_within_cohort", True))
    redist = bool(por.get("redistribute_across_cohorts", False))
    excl = bool(por.get("exclude_on_missing_price", True))
    cal = por.get("calendar", raw.get("signals", {}).get("momentum", {}).get("calendar", "union"))
    wmax = por.get("max_weight_per_name", None)
    wmax = (float(wmax) if wmax is not None else None)
    to_cap = por.get("turnover_cap", None)
    to_cap = (float(to_cap) if to_cap is not None else None)
    cfg = PortfolioConfig(
        k_months=k,
        selection=sel,  # type: ignore
        long_decile=dec,
        top_quantile=tq,
        renormalize_within_cohort=ren,
        redistribute_across_cohorts=redist,
        exclude_on_missing_price=excl,
        calendar=cal,
        max_weight_per_name=wmax,
        turnover_cap=to_cap,
    )
    # Basic validation
    if cfg.k_months < 1:
        raise ValueError("k_months must be >= 1")
    if cfg.selection == "top_decile" and cfg.long_decile < 1:
        raise ValueError("long_decile must be >= 1")
    if cfg.selection == "top_quantile" and not (0.0 < cfg.top_quantile <= 1.0):
        raise ValueError("top_quantile must be in (0,1]")
    return cfg


def select_winners(
    signals: pd.DataFrame,
    selection: SelectionMode = "top_decile",
    long_decile: int = 10,
    top_quantile: float = 0.10,
) -> pd.DataFrame:
    """Select winners per month from momentum signals.

    Expects columns: ['month_end','ticker','valid', ('decile' and/or 'pct_rank')].
    Returns a DataFrame with ['month_end','ticker'] of selected winners, per month.
    """
    req = {"month_end", "ticker", "valid"}
    if req - set(signals.columns):
        raise ValueError("signals must include ['month_end','ticker','valid']")
    d = signals.copy()
    d["month_end"] = pd.to_datetime(d["month_end"], errors="raise")
    d = d[d["valid"].fillna(False).astype(bool)].copy()
    if selection == "top_decile":
        if "decile" not in d.columns:
            raise ValueError("'decile' column missing for top_decile selection")
        winners = d[d["decile"].astype("Int64") == int(long_decile)][["month_end", "ticker"]]
    elif selection == "top_quantile":
        if "pct_rank" not in d.columns:
            raise ValueError("'pct_rank' column missing for top_quantile selection")
        thr = 1.0 - float(top_quantile)
        winners = d[d["pct_rank"].astype(float) >= thr][["month_end", "ticker"]]
    else:
        raise ValueError("unknown selection mode")
    return winners.sort_values(["month_end", "ticker"]).reset_index(drop=True)


def _month_add(dt: pd.Timestamp, n: int) -> pd.Timestamp:
    """Add n calendar months and return the month-end timestamp.

    Assumes dt is a month-end timestamp; tests rely on calendar month-ends.
    """
    return (pd.Timestamp(dt) + pd.offsets.MonthEnd(n)).normalize() + pd.offsets.Day(0)


def _build_active_mask_from_daily_prices(
    ohlcv_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build an activity mask at month frequency from daily OHLCV.

    Marks a name as active in month m if it has at least one daily observation
    in that calendar month. Returns columns ['month_end','ticker','active'].
    """
    if isinstance(ohlcv_df.index, pd.MultiIndex) and list(ohlcv_df.index.names) == ["date", "ticker"]:
        d = ohlcv_df.reset_index()[["date", "ticker"]].copy()
    else:
        if {"date", "ticker"} - set(ohlcv_df.columns):
            raise ValueError("ohlcv_df must have MultiIndex [date,ticker] or columns ['date','ticker']")
        d = ohlcv_df[["date", "ticker"]].copy()
    d["date"] = pd.to_datetime(d["date"], errors="raise")
    # Map each trading day to its calendar month-end (month-end at 00:00)
    d["month_end"] = (d["date"] + pd.offsets.MonthEnd(0)).dt.normalize()
    pres = (
        d.groupby(["month_end", "ticker"], as_index=False)
        .size()
        .rename(columns={"size": "cnt"})
    )
    pres["active"] = pres["cnt"] > 0
    return pres[["month_end", "ticker", "active"]].sort_values(["month_end", "ticker"]).reset_index(drop=True)


def build_overlapping_portfolio(
    winners: pd.DataFrame,
    k_months: int,
    renorm_within: bool = True,
    active_mask: Optional[pd.DataFrame] = None,
    redistribute_across_cohorts: bool = False,
) -> pd.DataFrame:
    """Construct overlapping K-month cohorts and compute holdings.

    Parameters
    - winners: DataFrame with ['month_end','ticker'] of winners per formation month.
    - k_months: K in (J,K) construction (>=1).
    - renorm_within: if True, keep each cohort sum at 1/K by renormalizing among active names.
      If False, freeze per-name weight at formation (1/K / n_t) and allow cohort sum to drop if names disappear.
    - active_mask: optional DataFrame ['month_end','ticker','active'] to indicate whether a name is active in a carry month.
      If None, all names remain active for K months by assumption.
    - redistribute_across_cohorts: if True, renormalize across cohorts to make portfolio sum to 1. Default False.

    Returns holdings with columns: ['month_end','ticker','weight','cohort_start','age_months','cohort_id','n_names_in_cohort','n_active_cohorts'].
    """
    if {"month_end", "ticker"} - set(winners.columns):
        raise ValueError("winners must include ['month_end','ticker']")
    if k_months < 1:
        raise ValueError("k_months must be >= 1")

    w = winners[["month_end", "ticker"]].copy()
    w["month_end"] = pd.to_datetime(w["month_end"], errors="raise")
    w = w.drop_duplicates().sort_values(["month_end", "ticker"]).reset_index(drop=True)

    # Cohort sizes at formation
    cohort_sizes = w.groupby("month_end")["ticker"].nunique().rename("n_form").reset_index()
    w = w.merge(cohort_sizes, on="month_end", how="left")
    w = w.rename(columns={"month_end": "cohort_start"})

    # Expand each (cohort_start, ticker) across ages 0..K-1
    ages = pd.DataFrame({"age_months": np.arange(k_months, dtype=int)})
    w_exp = (
        w.assign(key=1)
        .merge(ages.assign(key=1), on="key", how="outer")
        .drop(columns=["key"])
    )
    w_exp["month_end"] = w_exp.apply(lambda r: _month_add(r["cohort_start"], int(r["age_months"])), axis=1)

    # Active mask application
    if active_mask is not None and not active_mask.empty:
        am = active_mask.copy()
        if {"month_end", "ticker", "active"} - set(am.columns):
            raise ValueError("active_mask must have columns ['month_end','ticker','active']")
        am["month_end"] = pd.to_datetime(am["month_end"], errors="raise")
        am["active"] = am["active"].astype(bool)
        w_exp = w_exp.merge(am, on=["month_end", "ticker"], how="left")
        # Default to inactive when mask missing that entry (conservative: require explicit presence)
        # Ensure boolean dtype before filling to avoid downcasting warnings
        w_exp["active"] = w_exp["active"].astype("boolean").fillna(False).astype(bool)
        w_exp = w_exp[w_exp["active"]].drop(columns=["active"]).copy()

    # Weight per cohort
    wC = 1.0 / float(k_months)

    # Compute per-month cohort membership counts after activity filter
    cur_counts = (
        w_exp.groupby(["cohort_start", "month_end"])  # number of names alive in that cohort at that month
        .agg(n_alive=("ticker", "nunique"))
        .reset_index()
    )
    w_exp = w_exp.merge(cur_counts, on=["cohort_start", "month_end"], how="left")

    if renorm_within:
        # Each active name gets equal share of 1/K per cohort
        w_exp["cohort_weight_each"] = np.where(w_exp["n_alive"] > 0, wC / w_exp["n_alive"], 0.0)
    else:
        # Freeze weights at formation size
        w_exp["cohort_weight_each"] = np.where(w_exp["n_form"] > 0, wC / w_exp["n_form"], 0.0)

    # Compose final holdings per cohort
    w_exp["weight"] = w_exp["cohort_weight_each"].astype(float)

    # Aggregate across cohorts for total per-name weight per month
    total = (
        w_exp.groupby(["month_end", "ticker"], as_index=False)
        .agg(weight=("weight", "sum"))
    )

    # n_active_cohorts per month
    n_active_cohorts = (
        w_exp.groupby(["month_end", "cohort_start"]).size().reset_index().groupby("month_end").size().rename("n_active_cohorts").reset_index()
    )

    # n_names_in_cohort at that month (attach to per-cohort rows, then to aggregated names using a representative stat)
    per_cohort = w_exp[["month_end", "cohort_start", "ticker", "age_months", "weight", "n_alive"]].copy()
    per_cohort = per_cohort.rename(columns={"n_alive": "n_names_in_cohort"})

    # cohort_id based on sorted unique cohort_start modulo K for diagnostics
    unique_starts = (
        per_cohort[["cohort_start"]].drop_duplicates().sort_values("cohort_start").reset_index(drop=True)
    )
    unique_starts["rank"] = np.arange(len(unique_starts), dtype=int)
    unique_starts["cohort_id"] = (unique_starts["rank"] % int(k_months)).astype(int)
    per_cohort = per_cohort.merge(unique_starts.drop(columns=["rank"]), on="cohort_start", how="left")

    # Attach n_active_cohorts to per_cohort
    per_cohort = per_cohort.merge(n_active_cohorts, on="month_end", how="left")

    if redistribute_across_cohorts:
        # Rescale weights per month so the total equals 1 when there is at least one active cohort
        s = per_cohort.groupby("month_end")["weight"].sum().rename("sum_w").reset_index()
        per_cohort = per_cohort.merge(s, on="month_end", how="left")
        per_cohort["adj"] = np.where(per_cohort["sum_w"] > 0, 1.0 / per_cohort["sum_w"], 0.0)
        per_cohort["weight"] = per_cohort["weight"] * per_cohort["adj"]
        per_cohort = per_cohort.drop(columns=["sum_w", "adj"])

    # Return per-cohort holdings (richer diagnostics) and ensure schema
    per_cohort = per_cohort[[
        "month_end",
        "ticker",
        "weight",
        "cohort_start",
        "age_months",
        "cohort_id",
        "n_names_in_cohort",
        "n_active_cohorts",
    ]].sort_values(["month_end", "cohort_start", "ticker"]).reset_index(drop=True)

    return per_cohort


def holdings_to_trades(holdings: pd.DataFrame) -> pd.DataFrame:
    """Compute trades from holdings. Aggregates per month,ticker and diffs.

    Returns DataFrame with ['month_end','ticker','prev_weight','target_weight','trade_dW','side'].
    """
    req = {"month_end", "ticker", "weight"}
    if req - set(holdings.columns):
        raise ValueError("holdings must include ['month_end','ticker','weight']")
    d = holdings[["month_end", "ticker", "weight"]].copy()
    d["month_end"] = pd.to_datetime(d["month_end"], errors="raise")
    d = d.groupby(["month_end", "ticker"], as_index=False).agg(target_weight=("weight", "sum"))
    d = d.sort_values(["ticker", "month_end"]).reset_index(drop=True)
    d["prev_weight"] = d.groupby("ticker")["target_weight"].shift(1).fillna(0.0)
    d["trade_dW"] = (d["target_weight"] - d["prev_weight"]).astype(float)
    eps = 1e-12
    d["side"] = np.where(d["trade_dW"] > eps, "buy", np.where(d["trade_dW"] < -eps, "sell", "none"))
    return d


def generate_trades(
    holdings: pd.DataFrame,
    *,
    t_plus: int = 1,
    settlement: str = "T+2",
    ohlcv_df: Optional[pd.DataFrame] = None,
    calendar: Literal["union", "vnindex"] = "union",
    indices_dir: str | Path | None = None,
    max_weight_per_name: Optional[float] = None,
    turnover_cap: Optional[float] = None,
) -> pd.DataFrame:
    """Generate monthly trades with optional caps and trading dates.

    - Caps per-name target weight at `max_weight_per_name` (if provided).
    - Caps per-month gross turnover at `turnover_cap` by proportional scaling of trades (if provided).
    - Annotates `trade_date` shifted by `t_plus` trading days after the true month-end on the grid,
      and `settlement_date` if settlement == 'T+2' (trade_date + 2 trading days).

    Returns columns: ['month_end','ticker','prev_weight','target_weight','trade_dW','side',
                      'trade_date','settlement','settlement_date'].
    """
    req = {"month_end", "ticker", "weight"}
    if req - set(holdings.columns):
        raise ValueError("holdings must include ['month_end','ticker','weight']")
    # Aggregate to monthly targets
    d = holdings[["month_end", "ticker", "weight"]].copy()
    d["month_end"] = pd.to_datetime(d["month_end"], errors="raise")
    d = d.groupby(["month_end", "ticker"], as_index=False).agg(target_weight=("weight", "sum"))

    # Apply per-name weight cap if requested
    if max_weight_per_name is not None:
        wmax = float(max_weight_per_name)
        if wmax <= 0:
            raise ValueError("max_weight_per_name must be > 0 when provided")
        d["target_weight"] = d["target_weight"].clip(upper=wmax)

    # Prev weights by ticker
    d = d.sort_values(["ticker", "month_end"]).reset_index(drop=True)
    d["prev_weight"] = d.groupby("ticker")["target_weight"].shift(1).fillna(0.0)
    d["trade_dW"] = (d["target_weight"] - d["prev_weight"]).astype(float)

    # Turnover cap per month: scale all trades proportionally
    if turnover_cap is not None:
        cap = float(turnover_cap)
        if cap <= 0:
            raise ValueError("turnover_cap must be > 0 when provided")
        grp = d.groupby("month_end", as_index=False)
        turn = grp["trade_dW"].apply(lambda x: float(0.5 * np.abs(x).sum())).rename(columns={"trade_dW": "gross_turnover"})
        d = d.merge(turn, on="month_end", how="left")
        # scale = min(1, cap/turnover) per month
        eps = 1e-15
        scale = np.where(d["gross_turnover"] > eps, np.minimum(1.0, cap / d["gross_turnover"].astype(float)), 1.0)
        d["trade_dW"] = d["trade_dW"].astype(float) * scale
        # Adjust target_weight to reflect scaled trade
        d["target_weight"] = d["prev_weight"].astype(float) + d["trade_dW"].astype(float)
        d = d.drop(columns=["gross_turnover"])  # no longer needed

    # Side
    eps = 1e-12
    d["side"] = np.where(d["trade_dW"] > eps, "buy", np.where(d["trade_dW"] < -eps, "sell", "none"))

    # Build trading grid and map trade/settlement dates
    # Load OHLCV if needed
    if ohlcv_df is None:
        p_daily = Path("data/clean/ohlcv.parquet")
        if p_daily.exists():
            ohlcv_df = pd.read_parquet(p_daily)
        else:
            ohlcv_df = pd.DataFrame(columns=["date", "ticker", "close"]).set_index(pd.MultiIndex.from_arrays([[], []], names=["date", "ticker"]))
    # Load indices if calendar='vnindex' and indices_dir provided
    idx_df: Optional[pd.DataFrame] = None
    if calendar == "vnindex":
        # Try to load from provided folder; fall back to config default folder name
        try:
            dir_guess = Path(indices_dir) if indices_dir is not None else Path("vn_indices")
            if dir_guess.exists():
                idx_df = load_indices(dir_guess)
        except Exception:
            idx_df = None
    grid = build_trading_grid(ohlcv_df, calendar=calendar, indices_df=idx_df)
    # Map provided month_end timestamps to actual month-end on grid
    me = pd.to_datetime(pd.Series(d["month_end"].unique())).sort_values()
    me_on_grid = month_ends(grid)
    # period->month_end mapping
    period_to_me = pd.Series(me_on_grid.values, index=me_on_grid.to_period("M"))
    mapped_me = me.dt.to_period("M").map(period_to_me)
    # Build mapping from month_end to trade_date shifted by t_plus
    exec_dates = shift_trading_days(grid, pd.DatetimeIndex(mapped_me.values), int(t_plus))
    exec_map = {pd.Timestamp(k): pd.Timestamp(v) for k, v in zip(me.values, exec_dates.values)}
    d["trade_date"] = d["month_end"].map(lambda x: exec_map.get(pd.Timestamp(x), pd.NaT))

    # Settlement label and dates
    d["settlement"] = str(settlement)
    if settlement.upper() == "T+2":
        # settlement_date = trade_date + 2 trading days
        d["settlement_date"] = shift_trading_days(grid, pd.to_datetime(d["trade_date"]).fillna(pd.NaT), 2).values
    else:
        d["settlement_date"] = pd.NaT

    return d.sort_values(["month_end", "ticker"]).reset_index(drop=True)


def compute_portfolio(
    cfg_dict: dict | None = None,
    signals_df: pd.DataFrame | None = None,
    write: bool = True,
    out_holdings: str | Path = "data/clean/portfolio_holdings.parquet",
    out_trades: str | Path | None = "data/clean/portfolio_trades.parquet",
    out_summary: str | Path | None = None,
    ohlcv_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """End-to-end portfolio construction from momentum signals.

    Loads signals from data/clean/momentum.parquet if signals_df is None.
    Returns per-cohort holdings; callers may aggregate or derive trades.
    """
    cfg = _cfg_from_dict(cfg_dict)

    if signals_df is None:
        # Path resolution from cfg_dict's out section (if later added), default to standard path
        p = Path("data/clean/momentum.parquet")
        if not p.exists():
            raise FileNotFoundError(f"signals parquet not found: {p}")
        signals_df = pd.read_parquet(p)

    winners = select_winners(
        signals_df,
        selection=cfg.selection,
        long_decile=cfg.long_decile,
        top_quantile=cfg.top_quantile,
    )

    # Build active mask to drop names in carry months that lack prices
    active_mask: Optional[pd.DataFrame] = None
    if cfg.exclude_on_missing_price:
        try_df: Optional[pd.DataFrame]
        if ohlcv_df is not None:
            try_df = ohlcv_df
        else:
            # Default to clean OHLCV parquet from M1
            p_daily = Path("data/clean/ohlcv.parquet")
            try_df = pd.read_parquet(p_daily) if p_daily.exists() else None
        if try_df is not None and not try_df.empty:
            active_mask = _build_active_mask_from_daily_prices(try_df)

    # If we have an active mask, drop winners that are inactive at formation months
    if active_mask is not None and not active_mask.empty:
        am0 = active_mask[["month_end", "ticker", "active"]].copy()
        am0["month_end"] = pd.to_datetime(am0["month_end"], errors="raise")
        winners = winners.merge(am0, on=["month_end", "ticker"], how="left")
        # Avoid fillna downcasting warning by using pandas BooleanDtype first
        winners["active"] = winners["active"].astype("boolean").fillna(False).astype(bool)
        winners = winners[winners["active"]].drop(columns=["active"]).reset_index(drop=True)

    holdings = build_overlapping_portfolio(
        winners,
        k_months=cfg.k_months,
        renorm_within=cfg.renormalize_within_cohort,
        active_mask=active_mask,
        redistribute_across_cohorts=cfg.redistribute_across_cohorts,
    )

    # Generate trades with optional caps and trade/settlement dates
    # Attempt to infer indices folder for VNINDEX calendar from config default layout
    indices_dir = Path("vn_indices")
    trades = generate_trades(
        holdings,
        t_plus=1,
        settlement="T+2",
        ohlcv_df=ohlcv_df,
        calendar=cfg.calendar if cfg.calendar in ("union", "vnindex") else "union",
        indices_dir=indices_dir,
        max_weight_per_name=cfg.max_weight_per_name,
        turnover_cap=cfg.turnover_cap,
    )

    if write:
        out_holdings = Path(out_holdings)
        out_holdings.parent.mkdir(parents=True, exist_ok=True)
        holdings.to_parquet(out_holdings, index=False)
        if out_trades is not None:
            p_tr = Path(out_trades)
            p_tr.parent.mkdir(parents=True, exist_ok=True)
            trades.to_parquet(p_tr, index=False)
        if out_summary is not None:
            # Summary with active cohorts, holdings, turnover, and cash weight
            # n_active_cohorts: take max per month from holdings
            nac = (
                holdings.groupby("month_end")["n_active_cohorts"].max().rename("n_active_cohorts").reset_index()
                if "n_active_cohorts" in holdings.columns else
                holdings.groupby("month_end").size().rename("n_active_cohorts").reset_index()
            )
            # n_holdings: unique names held (any cohort)
            nh = holdings.groupby("month_end")["ticker"].nunique().rename("n_holdings").reset_index()
            # gross_turnover from trades
            gt = trades.groupby("month_end")["trade_dW"].apply(lambda x: float(0.5 * np.sum(np.abs(x)))).rename("gross_turnover").reset_index()
            # cash_weight = 1 - sum of invested weights across cohorts
            inv = holdings.groupby("month_end")["weight"].sum().rename("invested_weight").reset_index()
            inv["cash_weight"] = (1.0 - inv["invested_weight"]).clip(lower=0.0)
            s = nac.merge(nh, on="month_end", how="outer").merge(gt, on="month_end", how="outer").merge(inv[["month_end", "cash_weight"]], on="month_end", how="outer").sort_values("month_end")
            Path(out_summary).parent.mkdir(parents=True, exist_ok=True)
            s.to_csv(out_summary, index=False)

    return holdings
