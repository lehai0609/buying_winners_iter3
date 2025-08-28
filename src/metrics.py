from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def _to_series(x: pd.Series | Iterable[float]) -> pd.Series:
    if isinstance(x, pd.Series):
        return x.copy()
    return pd.Series(list(x))


def _align(a: pd.Series, b: pd.Series | None) -> tuple[pd.Series, pd.Series | None]:
    a = a.copy()
    a.index = pd.to_datetime(a.index)
    a = a.sort_index()
    if b is None:
        return a, None
    b = b.copy()
    b.index = pd.to_datetime(b.index)
    b = b.sort_index()
    idx = a.index.intersection(b.index)
    return a.loc[idx], b.loc[idx]


def _drawdown_path(returns_m: pd.Series) -> pd.DataFrame:
    r = returns_m.fillna(0.0)
    nav = (1.0 + r).cumprod()
    peak = nav.cummax()
    dd = nav / peak - 1.0
    out = pd.DataFrame({"nav": nav, "peak": peak, "drawdown": dd})
    return out


def drawdown_stats(returns_m: pd.Series) -> dict:
    """Compute max drawdown and duration statistics from monthly returns.

    Returns keys:
    - maxDD: float (negative)
    - dd_start, dd_trough, dd_recover: timestamps (may be NaT if never recovers)
    - dd_duration_months: int (max underwater stretch)
    """
    if returns_m is None or len(returns_m) == 0:
        return {
            "maxDD": np.nan,
            "dd_start": pd.NaT,
            "dd_trough": pd.NaT,
            "dd_recover": pd.NaT,
            "dd_duration_months": 0,
        }
    path = _drawdown_path(returns_m)
    dd = path["drawdown"]
    max_dd = float(dd.min()) if not dd.empty else np.nan
    # Locate start (prior peak), trough, and recovery
    trough_idx = dd.idxmin() if not dd.empty else None
    if trough_idx is None or pd.isna(trough_idx):
        return {
            "maxDD": np.nan,
            "dd_start": pd.NaT,
            "dd_trough": pd.NaT,
            "dd_recover": pd.NaT,
            "dd_duration_months": 0,
        }
    # Start: last time peak == nav strictly before trough
    before = path.loc[:trough_idx]
    # find last index where drawdown == 0 up to trough
    zeros = before.index[before["drawdown"] == 0]
    dd_start = zeros.max() if len(zeros) else before.index.min()
    # Recovery: first index after trough where nav reaches previous peak
    after = path.loc[trough_idx:]
    prev_peak = path.loc[trough_idx, "peak"]
    rec_idx = after.index[(after["nav"] >= prev_peak)]
    dd_recover = rec_idx.min() if len(rec_idx) else pd.NaT
    # Duration (max underwater streak length)
    underwater = (dd < 0).astype(int)
    # longest consecutive run length
    max_dur = 0
    cur = 0
    for v in underwater:
        if v == 1:
            cur += 1
            max_dur = max(max_dur, cur)
        else:
            cur = 0
    return {
        "maxDD": float(max_dd),
        "dd_start": dd_start,
        "dd_trough": trough_idx,
        "dd_recover": dd_recover,
        "dd_duration_months": int(max_dur),
    }


def perf_summary(
    returns_m: pd.Series,
    benchmark_m: pd.Series | None = None,
    rf: float | pd.Series = 0.0,
) -> dict:
    """Compute headline performance metrics on monthly returns.

    Parameters
    - returns_m: monthly simple returns (date-indexed Series)
    - benchmark_m: optional monthly simple returns for VN-Index
    - rf: monthly risk-free rate (scalar or date-aligned Series). If a scalar
      is provided, it is interpreted as per-month (not annualized).
    """
    if not isinstance(returns_m, pd.Series):
        returns_m = _to_series(returns_m)
    r = returns_m.dropna().sort_index()
    b = None
    if benchmark_m is not None:
        b = pd.Series(benchmark_m).dropna()
    r, b = _align(r, b)

    n = len(r)
    if n == 0:
        return {
            "N_months": 0,
            "CAGR": np.nan,
            "vol_ann": np.nan,
            "Sharpe": np.nan,
            "IR": np.nan,
            "maxDD": np.nan,
            "dd_duration": 0,
            "hit_rate_m": np.nan,
            "VaR95": np.nan,
            "VaR99": np.nan,
            "beta": np.nan,
        }

    # Risk-free alignment (monthly)
    if isinstance(rf, pd.Series):
        rf = rf.reindex(r.index).fillna(0.0)
        r_ex_rf = r - rf
    else:
        r_ex_rf = r - float(rf)

    # Annualization helpers
    ann_scale = np.sqrt(12.0)
    mean_m = r_ex_rf.mean()
    std_m = r_ex_rf.std(ddof=1)
    vol_ann = float(std_m * ann_scale) if np.isfinite(std_m) else np.nan
    sharpe = float(mean_m / std_m * ann_scale) if std_m and std_m > 0 else np.nan

    # CAGR uses total return path from raw returns (not excess)
    total_ret = (1.0 + r).prod()
    cagr = float(total_ret ** (12.0 / n) - 1.0) if n > 0 else np.nan

    # IR vs benchmark (if provided)
    if b is not None and len(b) == n:
        ex_b = r - b
        mu = ex_b.mean()
        sd = ex_b.std(ddof=1)
        ir = float(mu / sd * ann_scale) if sd and sd > 0 else np.nan
        # beta (with intercept) via OLS slope
        x = b.values
        y = r.values
        X = np.column_stack([np.ones_like(x), x])
        beta = float(np.linalg.lstsq(X, y, rcond=None)[0][1])
    else:
        ir = np.nan
        beta = np.nan

    dd = drawdown_stats(r)
    hit_rate = float((r > 0).mean()) if n > 0 else np.nan
    var95 = float(r.quantile(0.05))
    var99 = float(r.quantile(0.01))

    return {
        "N_months": int(n),
        "CAGR": cagr,
        "vol_ann": vol_ann,
        "Sharpe": sharpe,
        "IR": ir,
        "maxDD": dd["maxDD"],
        "dd_duration": dd["dd_duration_months"],
        "hit_rate_m": hit_rate,
        "VaR95": var95,
        "VaR99": var99,
        "beta": beta,
    }


def subperiod_metrics(
    returns_m: pd.Series,
    benchmark_m: pd.Series | None,
    periods: list[tuple[pd.Timestamp, pd.Timestamp]],
    rf: float | pd.Series = 0.0,
) -> pd.DataFrame:
    """Compute perf_summary over multiple subperiods.

    periods: list of (start_ts, end_ts) month-end boundaries (inclusive).
    """
    r = returns_m.copy()
    r.index = pd.to_datetime(r.index)
    b = None if benchmark_m is None else pd.Series(benchmark_m).copy()
    if b is not None:
        b.index = pd.to_datetime(b.index)
    records: list[dict] = []
    for start, end in periods:
        s = pd.to_datetime(start)
        e = pd.to_datetime(end)
        rr = r.loc[(r.index >= s) & (r.index <= e)]
        bb = None
        if b is not None:
            bb = b.loc[(b.index >= s) & (b.index <= e)]
        summ = perf_summary(rr, bb, rf=rf)
        summ["period_start"] = s
        summ["period_end"] = e
        records.append(summ)
    out = pd.DataFrame.from_records(records).set_index(["period_start", "period_end"]).sort_index()
    return out

