from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from pathlib import Path

from .momentum import compute_momentum_signals
from .portfolio import compute_portfolio, holdings_to_trades
from .costs import compute_costs, _cfg_from_dict as _costs_cfg_from_dict
from .backtest import compute_backtest
from .data_io import load_indices, get_index_series
from .metrics import perf_summary
from .stats import alpha_newey_west


def cost_sensitivity(
    best_params: Mapping[str, Any],
    costs_bps: Sequence[float],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Evaluate performance sensitivity to transaction cost assumptions.

    Run backtests over the given cost grid for the provided parameter selection
    (J, K), returning a DataFrame with columns like
    [J, K, cost_bps, sharpe, ret_ann, alpha_nw, vol_ann, turnover].
    """
    if not best_params:
        return pd.DataFrame(columns=["J", "K", "cost_bps", "sharpe", "ret_ann", "alpha_nw", "vol_ann", "turnover"])
    J = int(best_params.get("J"))
    K = int(best_params.get("K"))

    # Prepare common artifacts once (signals/holdings/trades)
    cfg = dict(config)
    cfg.setdefault("signals", {}).setdefault("momentum", {})
    cfg["signals"]["momentum"]["lookback_months"] = J
    cfg.setdefault("portfolio", {})
    cfg["portfolio"]["k_months"] = K
    ndecs = int(cfg.get("signals", {}).get("momentum", {}).get("n_deciles", 10))
    cfg["portfolio"]["long_decile"] = int(ndecs)

    p_ohlcv = Path((cfg.get("out", {}) or {}).get("ohlcv_parquet", "data/clean/ohlcv.parquet"))
    ohlcv_df = pd.read_parquet(p_ohlcv) if p_ohlcv.exists() else None

    signals = compute_momentum_signals(df=ohlcv_df, cfg_dict=cfg, write=False)
    holdings = compute_portfolio(cfg_dict=cfg, signals_df=signals, write=False, ohlcv_df=ohlcv_df)
    trades = holdings_to_trades(holdings)

    recs: list[dict] = []
    for cbps in costs_bps:
        # Override costs per-side bps
        cfg_cost = dict(cfg)
        cfg_cost.setdefault("costs", {})
        cfg_cost["costs"]["per_side_bps"] = float(cbps)
        # Keep use_adv, slippage params as-is
        costed, _ = compute_costs(cfg_dict=cfg_cost, trades_df=trades, ohlcv_df=None, write=False)
        _, monthly, _, _ = compute_backtest(
            cfg_dict=cfg_cost,
            ohlcv_df=ohlcv_df,
            holdings_df=holdings,
            trades_costed_df=costed,
            indices_df=None,
            write=False,
        )
        # Metrics on full sample
        m = monthly.copy()
        m.index = pd.to_datetime(m.index)
        sharpe = np.nan
        vol_ann = np.nan
        ret_ann = np.nan
        alpha_ann = np.nan
        if not m.empty:
            summ = perf_summary(m["ret_net_m"], benchmark_m=None, rf=0.0)
            sharpe = float(summ.get("Sharpe", np.nan))
            vol_ann = float(summ.get("vol_ann", np.nan))
            ret_ann = float(summ.get("CAGR", np.nan))
            try:
                indices_dir = (config.get("raw_dirs", {}) or {}).get("indices", "vn_indices")
                idx_df = load_indices(indices_dir, names=["VNINDEX"])  # type: ignore[arg-type]
                bench_close = get_index_series(idx_df, "VNINDEX").sort_index()
                bench = bench_close.reindex(bench_close.index.union(m.index)).ffill().reindex(m.index)
                bench_ret_m = bench.pct_change()
                alpha = alpha_newey_west(m["ret_net_m"], bench_ret_m, rf=0.0, lags=6, intercept=True)
                alpha_ann = float(alpha.get("alpha_ann", np.nan))
            except Exception:
                alpha_ann = np.nan
        turnover = float(m.get("gross_turnover_m", pd.Series(dtype=float)).mean()) if "gross_turnover_m" in m.columns else np.nan
        recs.append({
            "J": J,
            "K": K,
            "cost_bps": float(cbps),
            "sharpe": sharpe,
            "ret_ann": ret_ann,
            "alpha_nw": alpha_ann,
            "vol_ann": vol_ann,
            "turnover": turnover,
        })
    return pd.DataFrame(recs).sort_values(["cost_bps"]).reset_index(drop=True)


def subperiod_metrics(
    best_params: Mapping[str, Any],
    subperiods: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Compute metrics on named subperiods and/or regimes for chosen (J, K).

    Slice the backtest results by provided subperiods (name, start, end) and
    compute key metrics per slice for the chosen (J, K).
    """
    if not best_params:
        return pd.DataFrame()
    J = int(best_params.get("J"))
    K = int(best_params.get("K"))

    cfg = dict(config)
    cfg.setdefault("signals", {}).setdefault("momentum", {})
    cfg["signals"]["momentum"]["lookback_months"] = J
    cfg.setdefault("portfolio", {})
    cfg["portfolio"]["k_months"] = K
    ndecs = int(cfg.get("signals", {}).get("momentum", {}).get("n_deciles", 10))
    cfg["portfolio"]["long_decile"] = int(ndecs)

    p_ohlcv = Path((cfg.get("out", {}) or {}).get("ohlcv_parquet", "data/clean/ohlcv.parquet"))
    ohlcv_df = pd.read_parquet(p_ohlcv) if p_ohlcv.exists() else None

    signals = compute_momentum_signals(df=ohlcv_df, cfg_dict=cfg, write=False)
    holdings = compute_portfolio(cfg_dict=cfg, signals_df=signals, write=False, ohlcv_df=ohlcv_df)
    trades = holdings_to_trades(holdings)
    costed, _ = compute_costs(cfg_dict=cfg, trades_df=trades, ohlcv_df=None, write=False)
    _, monthly, _, _ = compute_backtest(
        cfg_dict=cfg,
        ohlcv_df=ohlcv_df,
        holdings_df=holdings,
        trades_costed_df=costed,
        indices_df=None,
        write=False,
    )

    m = monthly.copy()
    m.index = pd.to_datetime(m.index)

    out_rows = []
    for sp in subperiods:
        name = sp.get("name") or "subperiod"
        start = pd.to_datetime(sp.get("start"))
        end = pd.to_datetime(sp.get("end"))
        if pd.isna(start) or pd.isna(end):
            continue
        sub = m.loc[(m.index >= start) & (m.index <= end)]
        if sub.empty:
            out_rows.append({
                "name": name,
                "start": start,
                "end": end,
                "N_months": 0,
                "sharpe": np.nan,
                "ret_ann": np.nan,
                "vol_ann": np.nan,
                "alpha_nw": np.nan,
            })
            continue
        summ = perf_summary(sub["ret_net_m"], benchmark_m=None, rf=0.0)
        sharpe = float(summ.get("Sharpe", np.nan))
        vol_ann = float(summ.get("vol_ann", np.nan))
        ret_ann = float(summ.get("CAGR", np.nan))
        # Optional alpha vs VNINDEX
        alpha_ann = np.nan
        try:
            indices_dir = (config.get("raw_dirs", {}) or {}).get("indices", "vn_indices")
            idx_df = load_indices(indices_dir, names=["VNINDEX"])  # type: ignore[arg-type]
            bench_close = get_index_series(idx_df, "VNINDEX").sort_index()
            bench = bench_close.reindex(bench_close.index.union(sub.index)).ffill().reindex(sub.index)
            bench_ret_m = bench.pct_change()
            alpha = alpha_newey_west(sub["ret_net_m"], bench_ret_m, rf=0.0, lags=6, intercept=True)
            alpha_ann = float(alpha.get("alpha_ann", np.nan))
        except Exception:
            alpha_ann = np.nan
        out_rows.append({
            "name": name,
            "start": start,
            "end": end,
            "N_months": int(len(sub)),
            "sharpe": sharpe,
            "ret_ann": ret_ann,
            "vol_ann": vol_ann,
            "alpha_nw": alpha_ann,
        })
    return pd.DataFrame(out_rows).sort_values(["start", "end"]).reset_index(drop=True)
