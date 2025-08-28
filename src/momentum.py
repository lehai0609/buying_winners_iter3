from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

from .returns import monthly_returns
from .data_io import load_indices


GridMode = Literal["union", "vnindex"]


def _ensure_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.MultiIndex) and list(df.index.names) == ["date", "ticker"]:
        return df.sort_index()
    if {"date", "ticker"}.issubset(df.columns):
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"], errors="raise")
        return d.set_index(["date", "ticker"]).sort_index()
    raise ValueError("expected MultiIndex [date,ticker] or columns ['date','ticker']")


def compute_monthly_prices(df: pd.DataFrame, close_col: str = "close") -> pd.DataFrame:
    """Return last observed close per (ticker, calendar month).

    Output columns: ['ticker','month_end','close'] with month_end equal to the
    last available date for that ticker in each calendar month.
    """
    df = _ensure_multiindex(df)
    if close_col not in df.columns:
        raise ValueError(f"missing required price column '{close_col}'")
    d = df[[close_col]].copy()
    d = d.reset_index()
    d["month"] = d["date"].dt.to_period("M")
    # pick the last row per (ticker, month)
    d = d.sort_values(["ticker", "date"]).groupby(["ticker", "month"], as_index=False).tail(1)
    d = d.rename(columns={"date": "month_end", close_col: "close"})
    out = d[["ticker", "month_end", "close"]].sort_values(["month_end", "ticker"]).reset_index(drop=True)
    return out


def compute_monthly_returns(monthly_px: pd.DataFrame) -> pd.DataFrame:
    """Compute per-ticker monthly close-to-close returns from monthly closes.

    Expects columns ['ticker','month_end','close']; returns ['month_end','ticker','ret_1m'].
    """
    req = {"ticker", "month_end", "close"}
    if req - set(monthly_px.columns):
        raise ValueError("monthly_px must have columns ['ticker','month_end','close']")
    d = monthly_px.copy()
    d["month_end"] = pd.to_datetime(d["month_end"], errors="raise")
    d["close"] = pd.to_numeric(d["close"], errors="coerce")
    d = d.sort_values(["ticker", "month_end"]).reset_index(drop=True)
    d["ret_1m"] = d.groupby("ticker")["close"].pct_change()
    return d[["month_end", "ticker", "ret_1m"]]


def compute_momentum(
    returns_df: pd.DataFrame,
    lookback: int,
    gap: int,
    min_months_history: Optional[int] = None,
) -> pd.DataFrame:
    """Compute J-month momentum with G-month gap from monthly returns.

    Parameters
    - returns_df: DataFrame with ['month_end','ticker','ret_1m'].
    - lookback: J-month window size (>=1).
    - gap: G-month gap (>=0) between formation month and last included month.
    - min_months_history: if provided, require at least this many monthly observations
      historically prior to formation (default: lookback + gap).

    Returns columns: ['month_end','ticker','momentum','n_months_used','valid'].
    """
    req = {"month_end", "ticker", "ret_1m"}
    if req - set(returns_df.columns):
        raise ValueError("returns_df must have columns ['month_end','ticker','ret_1m']")
    if lookback < 1 or gap < 0:
        raise ValueError("lookback must be >=1 and gap >=0")
    if min_months_history is None:
        min_months_history = lookback + gap

    d = returns_df.copy()
    d["month_end"] = pd.to_datetime(d["month_end"], errors="raise")
    d["ret_1m"] = pd.to_numeric(d["ret_1m"], errors="coerce")
    d = d.sort_values(["ticker", "month_end"]).reset_index(drop=True)

    # Shift by gap, then rolling product over J months
    by = d.groupby("ticker", sort=False)
    shifted = by["ret_1m"].shift(gap)
    # Count finite observations in the window
    is_finite = shifted.replace([np.inf, -np.inf], np.nan).notna().astype(int)
    n_used = is_finite.groupby(d["ticker"]).rolling(window=lookback, min_periods=1).sum().reset_index(level=0, drop=True)

    prod = ((1.0 + shifted).groupby(d["ticker"]).rolling(window=lookback, min_periods=lookback).apply(lambda x: float(np.prod(x)), raw=False))
    prod = prod.reset_index(level=0, drop=True)
    momentum = prod - 1.0

    out = pd.DataFrame({
        "month_end": d["month_end"],
        "ticker": d["ticker"],
        "momentum": momentum,
        "n_months_used": n_used,
    })
    # Valid if enough months used in lookback and momentum is finite
    out["valid"] = (out["n_months_used"] >= lookback) & out["momentum"].replace([np.inf, -np.inf], np.nan).notna()

    return out


def assign_deciles(
    signal_df: pd.DataFrame,
    n_deciles: int = 10,
    min_names_per_month: int = 50,
) -> pd.DataFrame:
    """Assign cross-sectional deciles and percentile ranks per month.

    Adds 'decile' (int) and 'pct_rank' ([0,1]) where group size >= min_names_per_month.
    Leaves NaN for groups below threshold or invalid signals.
    """
    if {"month_end", "ticker", "momentum", "valid"} - set(signal_df.columns):
        raise ValueError("signal_df must have columns ['month_end','ticker','momentum','valid']")
    if n_deciles < 2:
        raise ValueError("n_deciles must be >= 2")
    if min_names_per_month < 1:
        raise ValueError("min_names_per_month must be >= 1")

    d = signal_df.copy()
    d["month_end"] = pd.to_datetime(d["month_end"], errors="raise")

    d["decile"] = np.nan
    d["pct_rank"] = np.nan

    def _assign(group: pd.DataFrame) -> pd.DataFrame:
        g = group.copy()
        mask = g["valid"].fillna(False).astype(bool)
        idx = g.index[mask]
        if mask.sum() < min_names_per_month:
            # leave NaNs
            return g
        vals = g.loc[idx, "momentum"].astype(float)
        # Percentile rank deterministic in [0,1]
        n = float(len(vals))
        if n > 1:
            ranks = vals.rank(method="average", ascending=True)
            g.loc[idx, "pct_rank"] = ((ranks - 1.0) / (n - 1.0)).astype(float)
        else:
            g.loc[idx, "pct_rank"] = np.nan
        # Deciles via qcut with duplicates='drop'
        try:
            cuts = pd.qcut(vals, q=n_deciles, duplicates="drop")
            g.loc[idx, "decile"] = (cuts.cat.codes + 1).astype(float)
        except Exception:
            # Fallback: rank-based binning
            ranks_dense = vals.rank(method="first")
            g.loc[idx, "decile"] = np.ceil(ranks_dense / (len(vals) / n_deciles)).clip(1, n_deciles)
        return g

    # Exclude grouping columns from applied frames to silence pandas future warning
    # and reattach the group key as a column for downstream consumers.
    d = d.groupby("month_end", group_keys=False).apply(
        lambda g: _assign(g).assign(month_end=g.name), include_groups=False
    )
    # cast decile to Int64 where available
    if "decile" in d.columns:
        try:
            d["decile"] = d["decile"].round().astype("Int64")
        except Exception:
            pass
    return d


@dataclass
class MomentumConfig:
    lookback_months: int = 12
    gap_months: int = 1
    n_deciles: int = 10
    min_months_history: Optional[int] = None
    min_names_per_month: int = 50
    exclude_hard_errors: bool = True
    calendar: GridMode = "union"
    price_col: str = "close"


def _cfg_from_dict(raw: dict | None) -> MomentumConfig:
    raw = raw or {}
    sig = (raw.get("signals", {}) or {}).get("momentum", {}) or {}
    lookback = int(sig.get("lookback_months", 12))
    gap = int(sig.get("gap_months", 1))
    ndecs = int(sig.get("n_deciles", 10))
    min_hist = sig.get("min_months_history", None)
    if min_hist is not None:
        min_hist = int(min_hist)
    min_names = int(sig.get("min_names_per_month", 50))
    excl = bool(sig.get("exclude_hard_errors", True))
    calendar = sig.get("calendar", "union")
    if calendar not in ("union", "vnindex"):
        calendar = "union"
    price_col = sig.get("price_col", "close")
    cfg = MomentumConfig(
        lookback_months=lookback,
        gap_months=gap,
        n_deciles=ndecs,
        min_months_history=min_hist,
        min_names_per_month=min_names,
        exclude_hard_errors=excl,
        calendar=calendar,  # type: ignore
        price_col=price_col,
    )
    # Basic validation
    if cfg.lookback_months < 1 or cfg.gap_months < 0 or cfg.n_deciles < 2:
        raise ValueError("invalid momentum config values")
    if cfg.min_months_history is None:
        cfg.min_months_history = cfg.lookback_months + cfg.gap_months
    if cfg.min_names_per_month < 1:
        cfg.min_names_per_month = 1
    return cfg


def compute_momentum_signals(
    df: pd.DataFrame | None = None,
    cfg_dict: dict | None = None,
    clean_parquet_path: str | Path | None = None,
    indices_dir: str | Path | None = None,
    write: bool = True,
    out_parquet: str | Path = "data/clean/momentum.parquet",
    summary_csv: str | Path | None = None,
) -> pd.DataFrame:
    """End-to-end momentum computation and optional persistence.

    If df is None, loads from 'out.ohlcv_parquet' in config/data.yml or
    from clean_parquet_path if provided.
    """
    cfg = _cfg_from_dict(cfg_dict)

    # Load OHLCV if not provided
    if df is None:
        # Resolve input paths from cfg_dict or defaults
        raw_out = (cfg_dict or {}).get("out", {}) if isinstance(cfg_dict, dict) else {}
        p = Path(clean_parquet_path or raw_out.get("ohlcv_parquet", "data/clean/ohlcv.parquet"))
        if not p.exists():
            raise FileNotFoundError(f"clean OHLCV parquet not found: {p}")
        df = pd.read_parquet(p)

    df = _ensure_multiindex(df)

    # Optionally exclude hard errors if report exists
    if cfg.exclude_hard_errors and isinstance(cfg_dict, dict):
        hard_path = Path((cfg_dict.get("out", {}) or {}).get("hard_errors_csv", "data/clean/hard_errors.csv"))
        if hard_path.exists():
            try:
                he = pd.read_csv(hard_path)
                if {"date", "ticker"}.issubset(he.columns):
                    he["date"] = pd.to_datetime(he["date"], errors="coerce")
                    he = he.dropna(subset=["date", "ticker"]).copy()
                    idx_to_drop = pd.MultiIndex.from_frame(he[["date", "ticker"]])
                    df = df[~df.index.isin(idx_to_drop)]
            except Exception:
                pass

    # Calendar indices if needed
    indices_df = None
    if cfg.calendar == "vnindex":
        dirp = Path(indices_dir or (cfg_dict or {}).get("raw_dirs", {}).get("indices", "vn_indices"))
        indices_df = load_indices(str(dirp))

    # Monthly returns from daily prices
    mret = monthly_returns(df, calendar=cfg.calendar, indices_df=indices_df, price_col=cfg.price_col)

    # Momentum
    mom = compute_momentum(
        mret,
        lookback=cfg.lookback_months,
        gap=cfg.gap_months,
        min_months_history=cfg.min_months_history,
    )

    # Deciles and percentile ranks
    out = assign_deciles(mom, n_deciles=cfg.n_deciles, min_names_per_month=cfg.min_names_per_month)

    # Optional summary
    summary = None
    try:
        grp = out[out["valid"].fillna(False)].groupby("month_end")
        if len(out) > 0 and not grp.obj.empty:
            summary = grp.agg(
                n_names=("ticker", "size"),
                n_valid=("momentum", "size"),
                min_signal=("momentum", "min"),
                max_signal=("momentum", "max"),
            ).reset_index()
            # infer number of unique deciles used per month
            dec_used = out.dropna(subset=["decile"]).groupby("month_end")["decile"].nunique().rename("n_deciles_used").reset_index()
            summary = summary.merge(dec_used, on="month_end", how="left")
    except Exception:
        summary = None

    if write:
        out_path = Path(out_parquet)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.sort_values(["month_end", "ticker"]).to_parquet(out_path, index=False)
        if summary is not None and summary_csv:
            Path(summary_csv).parent.mkdir(parents=True, exist_ok=True)
            summary.to_csv(Path(summary_csv), index=False)

    return out
