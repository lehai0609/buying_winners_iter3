from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import pandas as pd

from .calendar import build_trading_grid, month_ends, shift_trading_days, align_to_grid, GridMode
from .returns import daily_simple_returns


@dataclass
class BacktestConfig:
    lag_days: int = 1
    exec_price: Literal["open", "close"] = "open"
    calendar: GridMode = "union"
    skip_on_halt: bool = True
    skip_on_limit: bool = True
    price_limit_pct: float = 0.07  # 7% default; set 0 to disable
    write_holdings_daily: bool = False
    write_executions: bool = False


def _cfg_from_dict(raw: dict | None) -> BacktestConfig:
    raw = raw or {}
    b = (raw.get("backtest", {}) or {})
    cfg = BacktestConfig(
        lag_days=int(b.get("lag_days", 1)),
        exec_price=b.get("exec_price", "open"),  # type: ignore
        calendar=b.get("calendar", "union"),  # type: ignore
        skip_on_halt=bool(b.get("skip_on_halt", True)),
        skip_on_limit=bool(b.get("skip_on_limit", True)),
        price_limit_pct=float(b.get("price_limit_pct", 0.07)),
        write_holdings_daily=bool(b.get("write_holdings_daily", False)),
        write_executions=bool(b.get("write_executions", False)),
    )
    if cfg.lag_days < 0:
        raise ValueError("lag_days must be >= 0")
    if cfg.exec_price not in ("open", "close"):
        raise ValueError("exec_price must be 'open' or 'close'")
    if cfg.calendar not in ("union", "vnindex"):
        raise ValueError("calendar must be 'union' or 'vnindex'")
    if cfg.price_limit_pct < 0:
        raise ValueError("price_limit_pct must be >= 0")
    return cfg


def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.MultiIndex) and list(df.index.names) == ["date", "ticker"]:
        return df.sort_index()
    if {"date", "ticker"} - set(df.columns):
        raise ValueError("ohlcv must have MultiIndex [date,ticker] or columns ['date','ticker']")
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="raise")
    return d.set_index(["date", "ticker"]).sort_index()


def _cap_returns(ret: pd.Series, limit: float) -> pd.Series:
    if limit is None or limit <= 0:
        return ret
    return ret.clip(lower=-float(limit), upper=float(limit))


def _derive_exec_dates(
    ohlcv: pd.DataFrame,
    months: pd.Series | pd.Index | list,
    calendar: GridMode = "union",
    indices_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    grid = build_trading_grid(ohlcv, calendar=calendar, indices_df=indices_df)
    me = pd.to_datetime(pd.Index(months)).to_series(index=None)
    me = pd.Series(month_ends(grid)).to_frame(name="month_end") if len(me) == 0 else pd.DataFrame({"month_end": me})
    me = me.drop_duplicates().sort_values("month_end")
    return me


def _prepare_targets(holdings: pd.DataFrame) -> pd.DataFrame:
    if {"month_end", "ticker", "weight"} - set(holdings.columns):
        raise ValueError("holdings must include ['month_end','ticker','weight']")
    d = holdings[["month_end", "ticker", "weight"]].copy()
    d["month_end"] = pd.to_datetime(d["month_end"], errors="raise")
    d = d.groupby(["month_end", "ticker"], as_index=False).agg(target_weight=("weight", "sum"))
    return d


def _prepare_trades_costed(trades_costed: pd.DataFrame) -> pd.DataFrame:
    req = {"month_end", "ticker", "prev_weight", "target_weight", "trade_dW", "total_cost_bps"}
    if req - set(trades_costed.columns):
        raise ValueError("portfolio_trades_costed missing required columns")
    d = trades_costed[list(req)].copy()
    d["month_end"] = pd.to_datetime(d["month_end"], errors="raise")
    d[["prev_weight", "target_weight", "trade_dW", "total_cost_bps"]] = d[[
        "prev_weight",
        "target_weight",
        "trade_dW",
        "total_cost_bps",
    ]].apply(pd.to_numeric, errors="coerce")
    d = d.sort_values(["month_end", "ticker"]).reset_index(drop=True)
    return d


def _build_tradable_flags(ohlcv: pd.DataFrame) -> pd.Series:
    d = _ensure_ohlcv(ohlcv)
    vol = pd.to_numeric(d["volume"], errors="coerce")
    px = pd.to_numeric(d["close"], errors="coerce")
    tradable = (vol.fillna(0) > 0) & px.notna()
    tradable.name = "tradable"
    return tradable


def compute_backtest(
    cfg_dict: dict | None = None,
    ohlcv_df: pd.DataFrame | None = None,
    holdings_df: pd.DataFrame | None = None,
    trades_costed_df: pd.DataFrame | None = None,
    indices_df: pd.DataFrame | None = None,
    write: bool = True,
    out_daily: str | Path = "data/clean/backtest_daily.parquet",
    out_monthly: str | Path = "data/clean/backtest_monthly.parquet",
    out_holdings_daily: str | Path = "data/clean/holdings_daily.parquet",
    out_executions: str | Path = "data/clean/executions.parquet",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """Simulate monthly-rebalanced, long-only backtest with lags and trading frictions.

    Inputs
    - cfg_dict: nested config dict with 'backtest' options
    - ohlcv_df: daily OHLCV MultiIndex [date,ticker]
    - holdings_df: monthly target weights from M5
    - trades_costed_df: monthly costed trades from M6
    - indices_df: optional indices for VNINDEX calendar

    Outputs
    - daily_df: index 'date' with columns ['nav','ret_gross_d','ret_net_d','turnover_d','cash_weight']
    - monthly_df: index 'month_end' with monthly summary
    - holdings_daily_df: optional MultiIndex [date,ticker] post-trade weights and tradable flags
    - executions_df: optional per-[month_end,ticker] execution outcomes
    """
    cfg = _cfg_from_dict(cfg_dict)

    # Load defaults if not provided
    if ohlcv_df is None:
        p_ohlcv = Path("data/clean/ohlcv.parquet")
        if not p_ohlcv.exists():
            raise FileNotFoundError(f"ohlcv parquet not found: {p_ohlcv}")
        ohlcv_df = pd.read_parquet(p_ohlcv)
    if holdings_df is None:
        p_hold = Path("data/clean/portfolio_holdings.parquet")
        if not p_hold.exists():
            raise FileNotFoundError(f"holdings parquet not found: {p_hold}")
        holdings_df = pd.read_parquet(p_hold)
    if trades_costed_df is None:
        p_tc = Path("data/clean/portfolio_trades_costed.parquet")
        if not p_tc.exists():
            raise FileNotFoundError(f"trades_costed parquet not found: {p_tc}")
        trades_costed_df = pd.read_parquet(p_tc)

    ohlcv = _ensure_ohlcv(ohlcv_df)
    targets = _prepare_targets(holdings_df)
    tc = _prepare_trades_costed(trades_costed_df)

    # Build calendar grid and month-ends present in trades
    grid = build_trading_grid(ohlcv, calendar=cfg.calendar, indices_df=indices_df)
    # Map provided month_end timestamps to the grid's actual month-end trading days
    all_mes = pd.to_datetime(pd.Series(tc["month_end"].unique())).sort_values()
    me_on_grid = month_ends(grid)
    # Build mapping from period[M] -> month-end date on grid
    period_to_me = pd.Series(me_on_grid.values, index=me_on_grid.to_period("M"))
    mapped_mes = all_mes.dt.to_period("M").map(period_to_me)
    exec_dates = shift_trading_days(grid, pd.DatetimeIndex(mapped_mes.values), cfg.lag_days)
    exec_map = pd.Series(exec_dates.values, index=all_mes.values)

    # Daily returns (raw + capped for PnL modeling)
    raw_ret = daily_simple_returns(ohlcv, price_col="close").rename("ret_d")
    capped_ret = _cap_returns(raw_ret, cfg.price_limit_pct).rename("ret_d_capped")

    # Tradable flags and limit flags
    tradable = _build_tradable_flags(ohlcv)
    # Map per-day limit-hit: |raw_ret| >= limit
    if cfg.price_limit_pct and cfg.price_limit_pct > 0:
        limit_hit = raw_ret.abs() >= float(cfg.price_limit_pct) - 1e-15
    else:
        limit_hit = pd.Series(False, index=raw_ret.index)
    limit_hit.name = "limit_hit"

    # Execution outcomes per (month_end, ticker)
    recs = []
    # We'll also build a per-date weights setpoint map for later ffill
    weights_setpoints = {}

    # Group by month_end to evaluate executions on exec_date
    for m, g in tc.groupby("month_end", sort=True):
        d_exec = exec_map.get(pd.Timestamp(m), pd.NaT)
        if pd.isna(d_exec) or d_exec not in grid:
            # If we can't find exec date on grid, skip this month safely
            continue

        # Build per-ticker data
        # Merge with targets to ensure we capture names present only in holdings
        tg = targets[targets["month_end"].astype("datetime64[ns]") == m][["ticker", "target_weight"]]
        gg = g.merge(tg, on=["ticker"], how="outer", suffixes=("", "_chk"))
        # If tc didn't include target (should), fill from tg
        gg["target_weight"] = gg["target_weight"].fillna(gg["target_weight_chk"]).fillna(0.0)
        gg["prev_weight"] = gg["prev_weight"].fillna(0.0)
        gg["trade_dW"] = gg["trade_dW"].fillna(gg["target_weight"] - gg["prev_weight"]).fillna(0.0)
        gg["total_cost_bps"] = gg["total_cost_bps"].fillna(0.0)
        gg = gg[["ticker", "prev_weight", "target_weight", "trade_dW", "total_cost_bps"]].copy()

        # Determine tradability and limit on exec date
        # Build index for that date across tickers
        idx = pd.MultiIndex.from_product([[pd.Timestamp(d_exec)], gg["ticker"].astype(str)], names=["date", "ticker"])
        # Use fill_value to avoid downcasting warnings on fillna for boolean-like series
        trad = tradable.reindex(idx, fill_value=False)
        lhit = limit_hit.reindex(idx, fill_value=False)

        # Decide fills
        filled_weight = []
        filled_dW = []
        reasons = []
        adjustable = []  # tradable names we can scale during renorm
        for i, row in gg.iterrows():
            tk = str(row["ticker"])
            is_trad = bool(trad.loc[(pd.Timestamp(d_exec), tk)])
            is_limited = bool(lhit.loc[(pd.Timestamp(d_exec), tk)])
            skip = False
            reason = "none"
            if cfg.skip_on_halt and (not is_trad):
                skip = True
                reason = "halt"
            elif cfg.skip_on_limit and is_limited:
                skip = True
                reason = "limit"
            if skip:
                fw = float(row["prev_weight"])
                fdw = 0.0
            else:
                fw = float(row["target_weight"])
                fdw = float(row["target_weight"] - row["prev_weight"])
            filled_weight.append(fw)
            filled_dW.append(fdw)
            reasons.append(reason)
            adjustable.append(bool(is_trad))

        gg["filled_weight"] = filled_weight
        gg["filled_dW"] = filled_dW
        gg["skipped_reason"] = reasons
        gg["tradable"] = adjustable

        # Renormalize across tradable names to ensure invested <= 1 while keeping non-tradable fixed
        sum_fixed = float(gg.loc[~gg["tradable"], "filled_weight"].sum())
        sum_tradable = float(gg.loc[gg["tradable"], "filled_weight"].sum())
        cap_for_tradable = max(0.0, 1.0 - sum_fixed)
        if sum_tradable > 0 and cap_for_tradable < sum_tradable - 1e-15:
            scale = cap_for_tradable / sum_tradable
            gg.loc[gg["tradable"], "filled_weight"] *= scale
            # Update filled_dW to reflect scaling vs prev_weight
            gg.loc[gg["tradable"], "filled_dW"] = (
                gg.loc[gg["tradable"], "filled_weight"] - gg.loc[gg["tradable"], "prev_weight"]
            )

        # Costs: apply only for executed portion (skip -> 0); total_cost_bps already scaled by |dW|
        # If we had partial scaling, approximate scaling of costs by execution fraction relative to intended |trade_dW|
        eps = 1e-15
        # Compute execution fraction safely to avoid divide warnings
        denom = np.abs(gg["trade_dW"]).values
        numer = np.abs(gg["filled_dW"]).values
        mask = denom > eps
        exec_frac = np.zeros_like(denom, dtype=float)
        exec_frac[mask] = numer[mask] / denom[mask]
        gg["cost_bps_applied"] = gg["total_cost_bps"].astype(float) * exec_frac

        # Record execution setpoint weights
        weights_setpoints[pd.Timestamp(d_exec)] = gg.set_index("ticker")["filled_weight"].astype(float)

        # Append to execution logs
        rec = gg.copy()
        rec.insert(0, "rebalance_date", pd.Timestamp(d_exec))
        rec.insert(0, "month_end", pd.Timestamp(m))
        recs.append(rec)

    executions_df = None
    if recs:
        executions_df = pd.concat(recs, axis=0, ignore_index=True)
        executions_df = executions_df.sort_values(["rebalance_date", "ticker"]).reset_index(drop=True)

    # Build daily weights by forward-filling setpoints
    if len(grid) == 0:
        raise ValueError("empty trading grid from OHLCV")
    tickers = ohlcv.index.get_level_values("ticker").unique().astype(str)
    full_idx = pd.MultiIndex.from_product([grid, tickers], names=["date", "ticker"])
    wdf = pd.DataFrame(index=full_idx)
    # Start all-zero; set setpoints at exec dates where present
    wdf["weight"] = 0.0
    for d_exec, series in sorted(weights_setpoints.items(), key=lambda x: x[0]):
        # Set that day's weights to series (for listed tickers); others keep previous values and will be ffilled
        # We implement setpoint by writing series at date, then ffill later.
        idx = pd.MultiIndex.from_product([[pd.Timestamp(d_exec)], series.index.astype(str)], names=["date", "ticker"])
        wdf.loc[idx, "weight"] = series.values
    # Forward-fill per ticker, ensuring earlier days before first setpoint remain zero
    wdf = wdf.sort_index()
    # Forward-fill weights per ticker from the last setpoint; keep zeros before first set
    wdf["weight"] = (
        wdf.groupby(level="ticker")["weight"].transform(lambda s: s.replace(0.0, np.nan).ffill().fillna(0.0)).astype(float)
    )

    # Cash weight as 1 - sum weights per date (clip to [0,1])
    invested = wdf.groupby(level="date")["weight"].sum().rename("invested_weight")
    cash_weight = (1.0 - invested).clip(lower=0.0).rename("cash_weight")

    # Align returns to grid and compute daily portfolio returns
    ret = align_to_grid(pd.DataFrame(capped_ret), grid=grid, how="left")["ret_d_capped"].astype(float)
    # Compute weighted return per date: sum over tickers of w * ret
    tmp = wdf.join(ret, how="left")
    tmp["ret_d_capped"] = tmp["ret_d_capped"].fillna(0.0)
    gross_by_day = tmp.assign(prod=lambda x: x["weight"] * x["ret_d_capped"]).groupby(level="date")["prod"].sum()
    gross_by_day.name = "ret_gross_d"

    # Daily turnover: on exec dates only, compute 0.5 * sum |dW|
    turnover_d = pd.Series(0.0, index=grid, name="turnover_d")
    prev_w = None
    for dt in grid:
        cur = wdf.xs(dt, level="date")["weight"].reindex(tickers).fillna(0.0).values
        if prev_w is None:
            turnover_d.loc[dt] = 0.0
        else:
            dW = np.abs(cur - prev_w).sum() * 0.5
            turnover_d.loc[dt] = float(dW)
        prev_w = cur
    # Only keep non-zero at exec dates by zeroing out non-exec days where no setpoint changed
    exec_dates_set = set(weights_setpoints.keys())
    for dt in grid:
        if dt not in exec_dates_set:
            turnover_d.loc[dt] = 0.0

    # Daily cost drag on exec dates: - sum(cost_bps_applied)/1e4
    cost_drag = pd.Series(0.0, index=grid, name="cost_drag_d")
    if executions_df is not None and not executions_df.empty:
        csum = (
            executions_df.groupby("rebalance_date")["cost_bps_applied"].sum().rename("sum_cost_bps")
        )
        for dt, val in csum.items():
            if dt in cost_drag.index:
                cost_drag.loc[dt] = -float(val) / 1e4

    net_by_day = (gross_by_day.reindex(grid).fillna(0.0) + cost_drag.reindex(grid).fillna(0.0)).rename("ret_net_d")

    # NAV path
    nav = (1.0 + net_by_day.fillna(0.0)).cumprod().rename("nav")
    # Ensure index name consistency for join
    idx_named = pd.DatetimeIndex(grid, name="date")
    daily = pd.DataFrame({
        "nav": nav.reindex(idx_named),
        "ret_gross_d": gross_by_day.reindex(idx_named).fillna(0.0),
        "ret_net_d": net_by_day.reindex(idx_named).fillna(0.0),
        "turnover_d": turnover_d.reindex(idx_named).fillna(0.0),
        # Forward-fill explicitly to avoid deprecated fillna(method=...) usage
        "cash_weight": cash_weight.reindex(idx_named).ffill().fillna(1.0),
    }).sort_index()

    # Monthly summary
    months = pd.to_datetime(pd.Series(grid)).dt.to_period("M")
    dfm = daily.copy()
    dfm["month"] = pd.to_datetime(dfm.index).to_period("M")
    agg = dfm.groupby("month")
    # Compound daily into monthly simple returns
    # Apply on selected columns to avoid future change in GroupBy.apply including grouping cols
    rg = agg["ret_gross_d"].apply(lambda x: float((1.0 + x).prod() - 1.0)).rename("ret_gross_m")
    rn = agg["ret_net_d"].apply(lambda x: float((1.0 + x).prod() - 1.0)).rename("ret_net_m")
    to = agg["turnover_d"].sum().rename("gross_turnover_m")
    # Counts on exec dates
    n_holds = []
    n_trad = []
    n_halts = []
    n_limits = []
    for m in rg.index:
        # Find exec date in this calendar month
        month_start = (m.to_timestamp("M") - pd.offsets.MonthEnd(1)) + pd.offsets.Day(1)
        month_end_ts = m.to_timestamp("M")
        mask_days = (grid >= month_start) & (grid <= month_end_ts)
        dts = grid[mask_days]
        # Exec is any setpoint in this month (there should be at most one); if none, use last day to count holdings
        exec_in_m = [dt for dt in dts if dt in exec_dates_set]
        snap_dt = exec_in_m[0] if exec_in_m else (dts[-1] if len(dts) else None)
        if snap_dt is None:
            n_holds.append(0)
            n_trad.append(0)
            n_halts.append(0)
            n_limits.append(0)
            continue
        w = wdf.xs(snap_dt, level="date")["weight"]
        n_holds.append(int((w > 0).sum()))
        # Tradability and skip reasons from executions (if any in month)
        if executions_df is not None and not executions_df.empty and exec_in_m:
            ex = executions_df[executions_df["rebalance_date"] == exec_in_m[0]]
            n_trad.append(int(ex["tradable"].sum()))
            n_halts.append(int((ex["skipped_reason"] == "halt").sum()))
            n_limits.append(int((ex["skipped_reason"] == "limit").sum()))
        else:
            n_trad.append(0)
            n_halts.append(0)
            n_limits.append(0)

    monthly = pd.DataFrame({
        "month_end": [p.to_timestamp("M") for p in rg.index],
        "ret_gross_m": rg.values,
        "ret_net_m": rn.values,
        "gross_turnover_m": to.values,
        "n_holdings": n_holds,
        "n_tradable": n_trad,
        "n_halts": n_halts,
        "n_limit_skips": n_limits,
    }).set_index("month_end").sort_index()

    # Optional outputs
    holdings_daily_out = None
    if cfg.write_holdings_daily:
        trad = align_to_grid(pd.DataFrame(tradable), grid=grid, how="left")["tradable"].astype(bool)
        holdings_daily_out = (
            wdf.join(trad, how="left")
            .reset_index()
            .rename(columns={"weight": "weight", "tradable": "tradable"})
            .sort_values(["date", "ticker"])
            .set_index(["date", "ticker"])
        )

    if write:
        Path(out_daily).parent.mkdir(parents=True, exist_ok=True)
        daily.to_parquet(out_daily)
        Path(out_monthly).parent.mkdir(parents=True, exist_ok=True)
        monthly.to_parquet(out_monthly)
        if cfg.write_holdings_daily and holdings_daily_out is not None:
            Path(out_holdings_daily).parent.mkdir(parents=True, exist_ok=True)
            holdings_daily_out.to_parquet(out_holdings_daily)
        if cfg.write_executions and executions_df is not None:
            Path(out_executions).parent.mkdir(parents=True, exist_ok=True)
            executions_df.to_parquet(out_executions, index=False)

    return daily, monthly, holdings_daily_out, executions_df
