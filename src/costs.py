from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import pandas as pd


@dataclass
class CostsConfig:
    per_side_bps: float = 25.0
    use_adv: bool = False  # default off for backward-compat without capital
    adv_window_days: int = 21
    min_adv_trading_days: int = 15
    slippage_bps_per_1pct_adv: float = 2.0
    slippage_cap_bps: float = 100.0
    impact_model: Literal["none", "threshold"] = "none"
    impact_threshold_pct_adv: float = 10.0
    impact_bps: float = 10.0
    capital_vnd: float | None = None
    slippage_per_turnover_bps: float = 0.0  # used when use_adv=False


def _cfg_from_dict(raw: dict | None) -> CostsConfig:
    raw = raw or {}
    c = (raw.get("costs", {}) or {})
    cfg = CostsConfig(
        per_side_bps=float(c.get("per_side_bps", 25.0)),
        use_adv=bool(c.get("use_adv", False)),
        adv_window_days=int(c.get("adv_window_days", 21)),
        min_adv_trading_days=int(c.get("min_adv_trading_days", 15)),
        slippage_bps_per_1pct_adv=float(c.get("slippage_bps_per_1pct_adv", 2.0)),
        slippage_cap_bps=float(c.get("slippage_cap_bps", 100.0)),
        impact_model=c.get("impact_model", "none"),  # type: ignore
        impact_threshold_pct_adv=float(c.get("impact_threshold_pct_adv", 10.0)),
        impact_bps=float(c.get("impact_bps", 10.0)),
        capital_vnd=(float(c.get("capital_vnd")) if c.get("capital_vnd") is not None else None),
        slippage_per_turnover_bps=float(c.get("slippage_per_turnover_bps", 0.0)),
    )
    # Validation
    if cfg.per_side_bps < 0:
        raise ValueError("per_side_bps must be >= 0")
    if cfg.adv_window_days < 5:
        raise ValueError("adv_window_days must be >= 5")
    if not (1 <= cfg.min_adv_trading_days <= cfg.adv_window_days):
        raise ValueError("min_adv_trading_days must be in [1, adv_window_days]")
    if cfg.slippage_bps_per_1pct_adv < 0 or cfg.slippage_cap_bps < 0:
        raise ValueError("slippage parameters must be >= 0")
    if cfg.impact_bps < 0:
        raise ValueError("impact_bps must be >= 0")
    if cfg.impact_model not in ("none", "threshold"):
        raise ValueError("impact_model must be 'none' or 'threshold'")
    if cfg.impact_model == "threshold" and cfg.impact_threshold_pct_adv <= 0:
        raise ValueError("impact_threshold_pct_adv must be > 0 when impact_model='threshold'")
    if cfg.use_adv and (cfg.capital_vnd is None or cfg.capital_vnd <= 0):
        raise ValueError("capital_vnd must be provided and > 0 when use_adv=True")
    if not cfg.use_adv and cfg.slippage_per_turnover_bps < 0:
        raise ValueError("slippage_per_turnover_bps must be >= 0 when use_adv=False")
    return cfg


def _ensure_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.MultiIndex) and list(df.index.names) == ["date", "ticker"]:
        return df.sort_index()
    if {"date", "ticker"} - set(df.columns):
        raise ValueError("expected MultiIndex [date,ticker] or columns ['date','ticker']")
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="raise")
    return d.set_index(["date", "ticker"]).sort_index()


def compute_monthly_adv(
    ohlcv: pd.DataFrame,
    month_ends: pd.Series | list | np.ndarray,
    window_days: int = 21,
    min_days: int = 15,
) -> pd.DataFrame:
    """Compute value-ADV per ticker aligned to month-ends using trailing window.

    Parameters
    - ohlcv: daily DataFrame with MultiIndex [date,ticker], columns include ['close','volume']
    - month_ends: sequence of month-end timestamps to align to
    - window_days: trailing trading days for the rolling mean
    - min_days: minimum observations to emit a valid ADV

    Returns
    - DataFrame with columns ['month_end','ticker','adv_value']
    """
    if ohlcv is None or ohlcv.empty:
        return pd.DataFrame(columns=["month_end", "ticker", "adv_value"])
    d = _ensure_multiindex(ohlcv)
    if {"close", "volume"} - set(d.columns):
        raise ValueError("ohlcv must include 'close' and 'volume'")
    val = pd.to_numeric(d["close"], errors="coerce") * pd.to_numeric(d["volume"], errors="coerce")
    val.name = "val"
    frames = []
    me = pd.to_datetime(pd.Index(month_ends)).to_series(index=None)
    for tk, s in val.groupby(level="ticker"):
        s = s.droplevel("ticker").sort_index()
        roll = s.rolling(window_days, min_periods=min_days).mean()
        # align to month-ends by forward-filling the last available rolling value on/before month-end
        idx = s.index.union(pd.DatetimeIndex(me))
        aligned = (
            roll.reindex(idx).sort_index().ffill().reindex(pd.DatetimeIndex(me))
        )
        df_t = pd.DataFrame({
            "month_end": pd.DatetimeIndex(me),
            "ticker": tk,
            "adv_value": aligned.values,
        })
        frames.append(df_t)
    out = pd.concat(frames, axis=0, ignore_index=True)
    return out.sort_values(["month_end", "ticker"]).reset_index(drop=True)


def apply_trading_costs(
    trades: pd.DataFrame,
    adv: pd.DataFrame | None,
    config: CostsConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Enrich trades with cost attribution and produce a monthly summary.

    trades columns: ['month_end','ticker','prev_weight','target_weight','trade_dW','side']
    adv columns: ['month_end','ticker','adv_value'] (optional depending on config)
    Returns: (trades_costed, monthly_summary)
    """
    req = {"month_end", "ticker", "trade_dW"}
    if req - set(trades.columns):
        raise ValueError("trades must include ['month_end','ticker','trade_dW']")
    d = trades.copy()
    d["month_end"] = pd.to_datetime(d["month_end"], errors="raise")
    # Ensure optional columns exist
    for c in ("prev_weight", "target_weight", "side"):
        if c not in d.columns:
            if c == "side":
                # derive from sign if not provided
                eps = 1e-12
                d["side"] = np.where(d["trade_dW"] > eps, "buy", np.where(d["trade_dW"] < -eps, "sell", "none"))
            else:
                d[c] = np.nan

    # Merge ADV if provided
    if adv is not None and not adv.empty:
        a = adv.copy()
        if {"month_end", "ticker", "adv_value"} - set(a.columns):
            raise ValueError("adv must include ['month_end','ticker','adv_value']")
        a["month_end"] = pd.to_datetime(a["month_end"], errors="raise")
        d = d.merge(a, on=["month_end", "ticker"], how="left")
    else:
        d["adv_value"] = np.nan

    # Notional traded (if capital provided)
    if config.capital_vnd is not None:
        d["notional_traded_vnd"] = np.abs(d["trade_dW"]).astype(float) * float(config.capital_vnd)
    else:
        d["notional_traded_vnd"] = np.nan

    # Participation %ADV (0..inf); requires both notional and adv_value
    d["participation_pct_adv"] = np.where(
        d["notional_traded_vnd"].notna() & d["adv_value"].notna() & (d["adv_value"] > 0),
        100.0 * d["notional_traded_vnd"] / d["adv_value"],
        np.nan,
    )
    d["adv_missing"] = d["participation_pct_adv"].isna()

    # Fees in bps-return units: per-side bps times |dW|
    abs_dW = np.abs(d["trade_dW"]).astype(float)
    d["fees_bps"] = float(config.per_side_bps) * abs_dW

    # Slippage
    slippage = np.zeros(len(d), dtype=float)
    capped = np.zeros(len(d), dtype=bool)
    if config.use_adv:
        base = config.slippage_bps_per_1pct_adv
        cap = config.slippage_cap_bps
        part = d["participation_pct_adv"].astype(float)
        raw = base * part
        # Missing ADV → conservative cap
        raw = raw.where(~d["adv_missing"], other=cap)
        capped = raw >= cap - 1e-12
        slippage = np.minimum(raw, cap) * abs_dW
    else:
        # Weight-space slippage per unit turnover
        slippage = float(config.slippage_per_turnover_bps) * abs_dW
        capped = np.zeros(len(d), dtype=bool)
    d["slippage_bps"] = slippage
    d["slippage_capped"] = capped

    # Impact
    impact = np.zeros(len(d), dtype=float)
    if config.impact_model == "threshold" and config.use_adv:
        thr = float(config.impact_threshold_pct_adv)
        hit = (~d["adv_missing"]) & (d["participation_pct_adv"].astype(float) >= thr)
        impact = np.where(hit, float(config.impact_bps) * abs_dW, 0.0)
    d["impact_bps"] = impact

    # Total cost in bps-return units
    d["total_cost_bps"] = d[["fees_bps", "slippage_bps", "impact_bps"]].sum(axis=1)

    # Zero out all costs for exact zero trades (numerical safety)
    eps = 1e-15
    zero_mask = abs_dW <= eps
    for c in ("fees_bps", "slippage_bps", "impact_bps", "total_cost_bps"):
        d.loc[zero_mask, c] = 0.0
    d.loc[zero_mask, "slippage_capped"] = False

    # Monthly summary (portfolio-level bps already scaled by |dW|)
    grp = d.groupby("month_end", as_index=False)
    # Gross turnover per month
    turnover = grp.agg(gross_turnover=("trade_dW", lambda x: float(0.5 * np.abs(x).sum())))
    # Aggregate cost components as sums (already in bps-return units)
    costs = grp[["fees_bps", "slippage_bps", "impact_bps", "total_cost_bps"]].sum()
    summ = turnover.merge(costs, on="month_end", how="left")
    # Counts
    tmp = d.copy()
    tmp["is_trade"] = (tmp["trade_dW"].abs() > eps)
    tmp["is_buy"] = (tmp["trade_dW"] > eps)
    tmp["is_sell"] = (tmp["trade_dW"] < -eps)
    cnt = (
        tmp.groupby("month_end", as_index=False)
        .agg(
            n_trades=("is_trade", "sum"),
            n_buys=("is_buy", "sum"),
            n_sells=("is_sell", "sum"),
            n_capped_slippage=("slippage_capped", "sum"),
        )
    )
    summ = summ.merge(cnt, on="month_end", how="left")

    d = d.sort_values(["month_end", "ticker"]).reset_index(drop=True)
    summ = summ.sort_values("month_end").reset_index(drop=True)
    return d, summ


def compute_costs(
    cfg_dict: dict | None = None,
    trades_df: pd.DataFrame | None = None,
    ohlcv_df: pd.DataFrame | None = None,
    write: bool = True,
    out_trades_costed: str | Path = "data/clean/portfolio_trades_costed.parquet",
    out_summary: str | Path = "data/clean/costs_summary.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """High-level entrypoint to compute ADV (optional) and apply costs to trades.

    If trades_df/ohlcv_df are None, reads defaults:
      - trades: data/clean/portfolio_trades.parquet
      - ohlcv:  data/clean/ohlcv.parquet (only needed if use_adv=True)
    """
    cfg = _cfg_from_dict(cfg_dict)

    if trades_df is None:
        p_tr = Path("data/clean/portfolio_trades.parquet")
        if not p_tr.exists():
            raise FileNotFoundError(f"trades parquet not found: {p_tr}")
        trades_df = pd.read_parquet(p_tr)

    adv_df: pd.DataFrame | None = None
    if cfg.use_adv:
        if ohlcv_df is None:
            p_daily = Path("data/clean/ohlcv.parquet")
            if not p_daily.exists():
                raise FileNotFoundError("ohlcv parquet required for ADV when use_adv=True")
            ohlcv_df = pd.read_parquet(p_daily)
        # Unique month_ends in trades
        mes = pd.to_datetime(pd.Series(trades_df["month_end"].unique())).sort_values()
        adv_df = compute_monthly_adv(
            ohlcv_df,
            month_ends=mes,
            window_days=cfg.adv_window_days,
            min_days=cfg.min_adv_trading_days,
        )

    costed, summary = apply_trading_costs(trades_df, adv=adv_df, config=cfg)

    if write:
        p_costed = Path(out_trades_costed)
        p_costed.parent.mkdir(parents=True, exist_ok=True)
        costed.to_parquet(p_costed, index=False)
        p_sum = Path(out_summary)
        p_sum.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(p_sum, index=False)

    return costed, summary
