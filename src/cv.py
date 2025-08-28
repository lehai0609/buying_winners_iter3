from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from .momentum import compute_momentum_signals
from .portfolio import compute_portfolio, holdings_to_trades
from .costs import compute_costs
from .backtest import compute_backtest
from .data_io import load_indices, get_index_series
from .metrics import perf_summary
from .stats import alpha_newey_west


Window = Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]


def rolling_splits(
    month_ends: pd.DatetimeIndex,
    train_months: int,
    valid_months: int,
) -> List[Window]:
    """Build rolling walk-forward (train, validate) windows on monthly grid.

    The first validation window starts immediately after the training window.
    Each subsequent fold advances by `valid_months`.

    Args:
        month_ends: Sorted, unique month-end timestamps.
        train_months: Number of months in training window (>=1).
        valid_months: Number of months in validation window (>=1).

    Returns:
        List of 4-tuples: (train_start, train_end, valid_start, valid_end).
    """
    if train_months < 1 or valid_months < 1:
        raise ValueError("train_months and valid_months must be >= 1")
    if len(month_ends) == 0:
        return []

    # Ensure sorted unique index
    idx = pd.DatetimeIndex(sorted(month_ends.unique()))
    n = len(idx)
    folds: List[Window] = []

    # Anchor is the start index of validation window
    anchor = train_months
    while anchor + valid_months - 1 < n:
        train_start = idx[anchor - train_months]
        train_end = idx[anchor - 1]
        valid_start = idx[anchor]
        valid_end = idx[anchor + valid_months - 1]
        folds.append((train_start, train_end, valid_start, valid_end))
        anchor += valid_months

    return folds


def param_grid(j_grid: Sequence[int], k_grid: Sequence[int]) -> List[Tuple[int, int]]:
    """Cartesian product of J and K grids.

    Args:
        j_grid: Iterable of formation window lengths (months).
        k_grid: Iterable of holding window lengths (months).

    Returns:
        List of (J, K) tuples in input order.
    """
    combos: List[Tuple[int, int]] = []
    for j in j_grid:
        for k in k_grid:
            if j < 1 or k < 1:
                raise ValueError("J and K must be >= 1")
            combos.append((int(j), int(k)))
    return combos


def evaluate_combo(j: int, k: int, windows: Window, config: Mapping[str, Any]) -> Dict[str, Any]:
    """Evaluate a (J, K) combo on the provided validation window.

    Integrate M4/M5 (signals/portfolio), M6 (costs), M7 (backtest), and M8 (metrics)
    to compute validation metrics for a single (J, K) on a given fold.

    Returns a dict containing at least: J, K, window bounds, 'metric', 'value',
    and auxiliary metrics like 'turnover', 'ret_ann', 'vol_ann', 'sharpe', 'alpha_nw'.
    """
    # Unpack fold windows
    tr_s, tr_e, va_s, va_e = windows

    # Extract knobs
    cv_cfg = (config.get("cv", {}) or {})
    metric_name = str(cv_cfg.get("metric", "sharpe")).lower()
    skip_days = int(cv_cfg.get("skip_days", 5))

    # Clone config and override J, K, and backtest lag
    cfg = dict(config)
    cfg.setdefault("signals", {}).setdefault("momentum", {})
    cfg["signals"]["momentum"]["lookback_months"] = int(j)
    cfg.setdefault("portfolio", {})
    cfg["portfolio"]["k_months"] = int(k)
    # Ensure long_decile aligns with number of deciles to avoid empty winners
    ndecs = int(cfg.get("signals", {}).get("momentum", {}).get("n_deciles", 10))
    cfg["portfolio"]["long_decile"] = int(ndecs)
    cfg.setdefault("backtest", {})
    cfg["backtest"]["lag_days"] = skip_days

    # Load OHLCV once for downstream modules that need it
    p_ohlcv = Path((cfg.get("out", {}) or {}).get("ohlcv_parquet", "data/clean/ohlcv.parquet"))
    ohlcv_df = pd.read_parquet(p_ohlcv) if p_ohlcv.exists() else None

    # Build signals (M4)
    signals = compute_momentum_signals(
        df=ohlcv_df,
        cfg_dict=cfg,
        clean_parquet_path=None,
        indices_dir=None,
        write=False,
    )

    # Portfolio holdings (M5)
    holdings = compute_portfolio(
        cfg_dict=cfg,
        signals_df=signals,
        write=False,
        ohlcv_df=ohlcv_df,
    )
    trades = holdings_to_trades(holdings)

    # Costs (M6)
    costed, _ = compute_costs(cfg_dict=cfg, trades_df=trades, ohlcv_df=None, write=False)

    # Backtest (M7)
    daily, monthly, _, _ = compute_backtest(
        cfg_dict=cfg,
        ohlcv_df=ohlcv_df,
        holdings_df=holdings,
        trades_costed_df=costed,
        indices_df=None,
        write=False,
    )

    # Restrict to validation window for metrics (OOS)
    m = monthly.copy()
    m.index = pd.to_datetime(m.index)
    mask = (m.index >= pd.to_datetime(va_s)) & (m.index <= pd.to_datetime(va_e))
    m_val = m.loc[mask]
    # If empty (e.g., invalid/truncated fold), return NaNs safely
    if m_val.empty:
        return {
            "J": int(j),
            "K": int(k),
            "train_start": tr_s,
            "train_end": tr_e,
            "valid_start": va_s,
            "valid_end": va_e,
            "skip_days": skip_days,
            "metric": metric_name,
            "value": np.nan,
            "turnover": np.nan,
            "ret_ann": np.nan,
            "vol_ann": np.nan,
            "sharpe": np.nan,
            "alpha_nw": np.nan,
            "cost_bps": float(((config.get("costs", {}) or {}).get("per_side_bps", 25.0))),
        }

    # Benchmark and RF for metrics
    alpha_ann = np.nan
    try:
        indices_dir = (config.get("raw_dirs", {}) or {}).get("indices", "vn_indices")
        idx_df = load_indices(indices_dir, names=["VNINDEX"])  # type: ignore[arg-type]
        bench_close = get_index_series(idx_df, "VNINDEX").sort_index()
        # benchmark monthly simple returns aligned to backtest months
        bench = bench_close.reindex(bench_close.index.union(m_val.index)).ffill().reindex(m_val.index)
        bench_ret_m = bench.pct_change()
        # Compute alpha (M8)
        alpha = alpha_newey_west(m_val["ret_net_m"], bench_ret_m, rf=0.0, lags=6, intercept=True)
        alpha_ann = float(alpha.get("alpha_ann", np.nan))
    except Exception:
        alpha_ann = np.nan

    summ = perf_summary(m_val["ret_net_m"], benchmark_m=None, rf=0.0)
    sharpe = float(summ.get("Sharpe", np.nan))
    vol_ann = float(summ.get("vol_ann", np.nan))
    ret_ann = float(summ.get("CAGR", np.nan))
    # Use average monthly gross turnover on OOS window as tie-break proxy
    turnover = float(m_val.get("gross_turnover_m", pd.Series(dtype=float)).mean()) if "gross_turnover_m" in m_val.columns else np.nan

    # Select primary metric value
    if metric_name == "sharpe":
        value = sharpe
    elif metric_name == "alpha_nw":
        value = alpha_ann
    elif metric_name == "ir_vs_benchmark":
        # If benchmark not computed above, IR is NaN
        # We didn’t compute IR directly for OOS; use perf_summary with benchmark=None → NaN
        value = float("nan")
    elif metric_name == "sortino":
        # Not implemented separately; approximate with Sharpe for now
        value = sharpe
    else:
        value = sharpe

    return {
        "J": int(j),
        "K": int(k),
        "train_start": tr_s,
        "train_end": tr_e,
        "valid_start": va_s,
        "valid_end": va_e,
        "skip_days": skip_days,
        "metric": metric_name,
        "value": float(value) if np.isfinite(value) else np.nan,
        "turnover": float(turnover) if np.isfinite(turnover) else np.nan,
        "ret_ann": float(ret_ann) if np.isfinite(ret_ann) else np.nan,
        "vol_ann": float(vol_ann) if np.isfinite(vol_ann) else np.nan,
        "sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "alpha_nw": float(alpha_ann) if np.isfinite(alpha_ann) else np.nan,
        "cost_bps": float(((config.get("costs", {}) or {}).get("per_side_bps", 25.0))),
    }


def select_best(
    results: Sequence[Mapping[str, Any]],
    metric: str = "sharpe",
    tie_breaker: str = "lower_turnover",
) -> Mapping[str, Any]:
    """Select the best (J, K) result by metric with tie-breaking.

    Tie-breaker priority when primary metric ties:
      - lower_turnover: prefer smaller 'turnover'
      - higher_return: prefer larger 'ret_ann'
      - lower_k: prefer smaller 'K'

    Args:
        results: Sequence of per-combo results (dict-like).
        metric: Primary metric name to compare by. If results contain a
            per-row 'metric' string and 'value', only rows where 'metric'
            equals this value are considered; otherwise 'value' is assumed to
            correspond to `metric` for all rows.
        tie_breaker: One of {'lower_turnover','higher_return','lower_k'}.

    Returns:
        The best result mapping.
    """
    if not results:
        raise ValueError("No results to select from")

    # Filter results to the requested metric if rows carry a 'metric' label
    filtered: List[Mapping[str, Any]]
    if any("metric" in r for r in results):
        filtered = [r for r in results if r.get("metric") == metric]
        if not filtered:
            raise ValueError(f"No results with metric == {metric}")
    else:
        filtered = list(results)

    def sort_key(r: Mapping[str, Any]):
        # Primary: value descending
        primary = float(r.get("value", float("nan")))
        # Tie-breakers
        turnover = float(r.get("turnover", float("inf")))
        ret_ann = float(r.get("ret_ann", float("nan")))
        k_val = int(r.get("K", 10**9))

        if tie_breaker == "lower_turnover":
            tb = (-turnover,)  # lower is better -> higher negative is better
        elif tie_breaker == "higher_return":
            tb = (ret_ann,)
        elif tie_breaker == "lower_k":
            tb = (-k_val,)  # lower is better
        else:
            tb = ()
        return (primary, *tb)

    # Sort descending by primary metric and tie-breaker proxy
    best = sorted(filtered, key=sort_key, reverse=True)[0]
    return best


def cross_validate(config: Mapping[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run cross-validation over (J, K) with rolling splits.

    Orchestrate CV across (J, K) combos using rolling 36/12-style folds.

    Returns three DataFrames: (cv_results, cv_selection, cv_oos_summary).
    """
    # Resolve CV config with defaults
    cv_cfg = (config.get("cv", {}) or {})
    train_months = int(cv_cfg.get("train_months", 36))
    valid_months = int(cv_cfg.get("valid_months", 12))
    j_grid = list(cv_cfg.get("j_grid", [3, 6, 9, 12]))
    k_grid = list(cv_cfg.get("k_grid", [1, 3, 6, 12]))

    # Build month-end grid from OHLCV via momentum signals to ensure alignment
    sig = compute_momentum_signals(df=None, cfg_dict=config, write=False)
    if sig.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    me = pd.to_datetime(sig["month_end"]).sort_values().unique()
    folds = rolling_splits(pd.DatetimeIndex(me), train_months=train_months, valid_months=valid_months)
    if not folds:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    combos = param_grid(j_grid, k_grid)

    # Evaluate all combos per fold
    rows: list[dict] = []
    for fold_id, w in enumerate(folds, start=1):
        for (j, k) in combos:
            res = evaluate_combo(int(j), int(k), w, config)
            res["fold_id"] = int(fold_id)
            rows.append(res)

    cv_results = pd.DataFrame(rows)
    # Ensure column order
    base_cols = [
        "fold_id",
        "train_start",
        "train_end",
        "valid_start",
        "valid_end",
        "J",
        "K",
        "skip_days",
        "metric",
        "value",
        "turnover",
        "cost_bps",
        "sharpe",
        "alpha_nw",
        "ret_ann",
        "vol_ann",
    ]
    cv_results = cv_results[[c for c in base_cols if c in cv_results.columns]]

    # Select best per fold
    selection_rows = []
    tie_breaker = str(cv_cfg.get("tie_breaker", "lower_turnover"))
    metric = str(cv_cfg.get("metric", "sharpe"))
    for fid, g in cv_results.groupby("fold_id"):
        best = select_best(g.to_dict("records"), metric=metric, tie_breaker=tie_breaker)
        rec = dict(best)
        rec["fold_id"] = int(fid)
        selection_rows.append(rec)
    cv_selection = pd.DataFrame(selection_rows)

    # Selection frequency of (J,K)
    if not cv_selection.empty:
        sel_counts = (
            cv_selection.groupby(["J", "K"]).size().rename("select_count").reset_index()
        )
        sel_counts["select_rate"] = sel_counts["select_count"] / float(cv_selection["fold_id"].nunique())
        cv_selection = cv_selection.merge(sel_counts, on=["J", "K"], how="left")

    # OOS summary across folds: fixed combos and dynamic selection
    oos_rows = []
    # Fixed (J,K): aggregate by combo
    for (j, k), grp in cv_results.groupby(["J", "K"]):
        o = {
            "policy": "fixed",
            "J": int(j),
            "K": int(k),
            "metric": metric,
            "value_mean": float(grp["value"].mean()),
            "value_std": float(grp["value"].std(ddof=1)) if len(grp) > 1 else np.nan,
            "sharpe_mean": float(grp["sharpe"].mean()),
            "ret_ann_mean": float(grp["ret_ann"].mean()),
            "vol_ann_mean": float(grp["vol_ann"].mean()),
        }
        oos_rows.append(o)

    # Dynamic selection: take best rows per fold then aggregate
    dyn = []
    for fid, g in cv_results.groupby("fold_id"):
        best = select_best(g.to_dict("records"), metric=metric, tie_breaker=tie_breaker)
        dyn.append(best)
    if dyn:
        dg = pd.DataFrame(dyn)
        o = {
            "policy": "dynamic_selection",
            "J": int(dg["J"].mode().iloc[0]) if not dg["J"].empty else np.nan,
            "K": int(dg["K"].mode().iloc[0]) if not dg["K"].empty else np.nan,
            "metric": metric,
            "value_mean": float(dg["value"].mean()),
            "value_std": float(dg["value"].std(ddof=1)) if len(dg) > 1 else np.nan,
            "sharpe_mean": float(dg["sharpe"].mean()),
            "ret_ann_mean": float(dg["ret_ann"].mean()),
            "vol_ann_mean": float(dg["vol_ann"].mean()),
        }
        oos_rows.append(o)

    cv_oos_summary = pd.DataFrame(oos_rows)

    return cv_results, cv_selection, cv_oos_summary
