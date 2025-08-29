from __future__ import annotations

import argparse
from pathlib import Path
import sys
import yaml
import pandas as pd

# Ensure project root (containing `src/`) is importable when running as a script
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.momentum import compute_momentum_signals
from src.portfolio import compute_portfolio, holdings_to_trades
from src.costs import compute_costs
from src.backtest import compute_backtest
from src.metrics import perf_summary, subperiod_metrics
from src.stats import alpha_newey_west
from src.data_io import load_indices, get_index_series
from src.reports import assemble_report


def _load_cfg(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"config not found: {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _override_jk(cfg: dict, J: int, K: int) -> dict:
    # Defensive copy (shallow is enough for our simple overrides)
    out = dict(cfg)
    out.setdefault("signals", {}).setdefault("momentum", {})
    out["signals"]["momentum"]["lookback_months"] = int(J)
    # Keep existing gap/deciles settings
    ndecs = int(out["signals"]["momentum"].get("n_deciles", 10))
    out.setdefault("portfolio", {})
    out["portfolio"]["k_months"] = int(K)
    # Ensure top-decile selection aligns with decile count
    out["portfolio"]["long_decile"] = int(ndecs)
    return out


def _write_metrics(backtest_monthly: Path, indices_dir: Path, out_dir: Path, rf_annual: float = 0.0, benchmark: str = "VNINDEX", newey_lags: int = 6) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    dfm = pd.read_parquet(backtest_monthly)
    if "month_end" in dfm.columns:
        dfm = dfm.set_index("month_end")
    dfm.index = pd.to_datetime(dfm.index)
    col = "ret_net_m" if "ret_net_m" in dfm.columns else ("ret_gross_m" if "ret_gross_m" in dfm.columns else None)
    if col is None:
        raise SystemExit("backtest_monthly.parquet missing ret_* columns")
    strat = pd.to_numeric(dfm[col], errors="coerce").rename("strategy_ret_m")
    me = pd.to_datetime(strat.index)
    idx_df = load_indices(indices_dir, names=[benchmark])
    bench_close = get_index_series(idx_df, benchmark).sort_index()
    bench = bench_close.reindex(bench_close.index.union(me)).ffill().reindex(me).pct_change().rename("benchmark_ret_m")

    rf_m = (1.0 + float(rf_annual)) ** (1.0 / 12.0) - 1.0 if rf_annual else 0.0

    summary = perf_summary(strat, benchmark_m=bench, rf=rf_m)
    pd.DataFrame([summary]).to_csv(out_dir / "metrics_summary.csv", index=False)

    alpha = alpha_newey_west(strat, bench, rf=rf_m, lags=int(newey_lags), intercept=True)
    pd.DataFrame([alpha]).to_csv(out_dir / "alpha_newey_west.csv", index=False)

    # Default subperiods (optional)
    defaults = [
        ("2010-01-31", "2014-12-31"),
        ("2015-01-31", "2019-12-31"),
        ("2020-01-31", "2025-12-31"),
    ]
    periods = [(pd.to_datetime(s), pd.to_datetime(e)) for s, e in defaults]
    sp = subperiod_metrics(strat, bench, periods, rf=rf_m)
    sp.reset_index().to_csv(out_dir / "metrics_subperiods.csv", index=False)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run a single J/K momentum scenario end-to-end (M4->M8 + report)")
    p.add_argument("--config", "-c", default="config/data.yml", help="Path to config YAML")
    p.add_argument("--params", nargs=2, type=int, metavar=("J", "K"), required=True, help="Formation J and holding K months")
    p.add_argument("--run-id", default=None, help="Optional run identifier to tag artifacts")
    p.add_argument("--indices-dir", type=Path, default=Path("vn_indices"), help="Directory with benchmark indices")
    args = p.parse_args(argv)

    cfg = _load_cfg(args.config)
    J, K = map(int, args.params)
    cfg_jk = _override_jk(cfg, J, K)

    # M4: Signals
    compute_momentum_signals(df=None, cfg_dict=cfg_jk, clean_parquet_path=None, indices_dir=None, write=True)

    # M5: Portfolio
    holdings = compute_portfolio(cfg_dict=cfg_jk, signals_df=None, write=True)
    _ = holdings_to_trades(holdings)  # ensure trades exist for costs

    # M6: Costs
    compute_costs(cfg_dict=cfg_jk, trades_df=None, ohlcv_df=None, write=True)

    # M7: Backtest
    compute_backtest(cfg_dict=cfg_jk, ohlcv_df=None, holdings_df=None, trades_costed_df=None, indices_df=None, write=True)

    # M8: Metrics (write CSVs used by report)
    out_dir = Path((cfg_jk.get("out", {}) or {}).get("out_dir", "data/clean"))
    _write_metrics(
        backtest_monthly=Path((cfg_jk.get("out", {}) or {}).get("backtest_monthly", "data/clean/backtest_monthly.parquet")),
        indices_dir=args.indices_dir,
        out_dir=out_dir,
        rf_annual=float((cfg_jk.get("metrics", {}) or {}).get("rf_annual", 0.0)),
        benchmark=str((cfg_jk.get("metrics", {}) or {}).get("benchmark", "VNINDEX")),
        newey_lags=int((cfg_jk.get("metrics", {}) or {}).get("newey_lags", 6)),
    )

    # Report
    run_id = args.run_id or f"J{J}_K{K}"
    report = assemble_report(cfg_jk, run_id=run_id)
    print("Single-scenario run complete.")
    print(f" - Report: {report.get('report_md')}")
    print(f" - Figures: {len(report.get('figs', []))} created")
    print(f" - Tables: {len(report.get('tables', []))} created")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
