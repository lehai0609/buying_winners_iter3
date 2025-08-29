from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Ensure project root (containing `src/`) is importable when running as a script
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.metrics import perf_summary, drawdown_stats, subperiod_metrics
from src.stats import alpha_newey_west, bootstrap_cis
from src.data_io import load_indices, get_index_series


def _load_backtest_monthly(path: Path, use_net: bool = True) -> pd.Series:
    df = pd.read_parquet(path)
    if "month_end" in df.columns:
        df = df.set_index("month_end")
    df.index = pd.to_datetime(df.index)
    col = "ret_net_m" if use_net and "ret_net_m" in df.columns else "ret_gross_m"
    if col not in df.columns:
        raise ValueError("expected column 'ret_net_m' or 'ret_gross_m' in backtest monthly parquet")
    s = pd.to_numeric(df[col], errors="coerce").rename("strategy_ret_m")
    return s


def _benchmark_returns(indices_dir: Path, name: str, month_ends: pd.DatetimeIndex) -> pd.Series:
    idx_df = load_indices(indices_dir, names=[name])
    close = get_index_series(idx_df, name)
    close = close.sort_index()
    # As-of sampling at strategy month-ends
    s = close.reindex(close.index.union(month_ends)).sort_index().ffill().reindex(month_ends)
    r = s.pct_change().rename("benchmark_ret_m")
    return r


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compute metrics and alpha tests for momentum strategy")
    ap.add_argument("--backtest-monthly", type=Path, default=Path("data/clean/backtest_monthly.parquet"))
    ap.add_argument("--indices-dir", type=Path, default=Path("vn_indices"))
    ap.add_argument("--benchmark", type=str, default="VNINDEX")
    ap.add_argument("--use-net", action="store_true", help="use net returns if available (default)")
    ap.add_argument("--use-gross", action="store_true", help="override to use gross returns")
    ap.add_argument("--rf-annual", type=float, default=0.0, help="annualized risk-free rate (fraction)")
    ap.add_argument("--newey-lags", type=int, default=6, help="NW lags for monthly data")
    ap.add_argument("--bootstrap", action="store_true", help="compute bootstrap CIs for Sharpe and alpha")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--block-size", type=int, default=6)
    ap.add_argument(
        "--subperiod",
        action="append",
        default=[],
        help=(
            "Optional subperiod in the form START:END:LABEL (dates YYYY-MM-DD). "
            "Can be provided multiple times. If none provided, defaults to "
            "2010-01-31..2014-12-31, 2015-01-31..2019-12-31, 2020-01-31..2025-12-31."
        ),
    )
    ap.add_argument("--out-dir", type=Path, default=Path("data/clean"))
    args = ap.parse_args(argv)

    use_net = True
    if args.use_gross:
        use_net = False
    elif args.use_net:
        use_net = True

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    strat = _load_backtest_monthly(args.backtest_monthly, use_net=use_net)
    me = pd.to_datetime(strat.index)
    bench = _benchmark_returns(args.indices_dir, args.benchmark, me)

    rf_m = (1.0 + float(args.rf_annual)) ** (1.0 / 12.0) - 1.0 if args.rf_annual else 0.0

    summary = perf_summary(strat, benchmark_m=bench, rf=rf_m)
    pd.DataFrame([summary]).to_csv(out_dir / "metrics_summary.csv", index=False)

    alpha = alpha_newey_west(strat, bench, rf=rf_m, lags=int(args.newey_lags), intercept=True)
    pd.DataFrame([alpha]).to_csv(out_dir / "alpha_newey_west.csv", index=False)

    # Subperiod metrics
    periods = []
    labels = []
    if args.subperiod:
        for spec in args.subperiod:
            try:
                start_s, end_s, label = spec.split(":", 2)
            except ValueError:
                raise SystemExit(f"Invalid --subperiod format: {spec!r}; expected START:END:LABEL")
            start = pd.to_datetime(start_s)
            end = pd.to_datetime(end_s)
            periods.append((start, end))
            labels.append(label)
    else:
        defaults = [
            ("2010-01-31", "2014-12-31", "2010-2014"),
            ("2015-01-31", "2019-12-31", "2015-2019"),
            ("2020-01-31", "2025-12-31", "2020-2025"),
        ]
        for s, e, lab in defaults:
            periods.append((pd.to_datetime(s), pd.to_datetime(e)))
            labels.append(lab)

    if periods:
        sp_df = subperiod_metrics(strat, bench, periods, rf=rf_m)
        # Attach labels if lengths match; otherwise derive from index
        if len(labels) == len(sp_df):
            sp_df = sp_df.reset_index()
            sp_df["label"] = labels
            sp_df = sp_df.set_index(["period_start", "period_end"])  # keep label as a column
        sp_df.reset_index().to_csv(out_dir / "metrics_subperiods.csv", index=False)

    if args.bootstrap:
        ci_sharpe = bootstrap_cis(strat, bench, metric="Sharpe", n_boot=args.n_boot, block_size=args.block_size)
        ci_alpha = bootstrap_cis(strat, bench, metric="alpha_ann", n_boot=args.n_boot, block_size=args.block_size, lags=args.newey_lags)
        cis = pd.DataFrame([
            {"metric": "Sharpe", **ci_sharpe},
            {"metric": "alpha_ann", **ci_alpha},
        ])
        cis.to_csv(out_dir / "bootstrap_cis.csv", index=False)

    print("Wrote:")
    print(f" - {out_dir / 'metrics_summary.csv'}")
    print(f" - {out_dir / 'alpha_newey_west.csv'}")
    if periods:
        print(f" - {out_dir / 'metrics_subperiods.csv'}")
    if args.bootstrap:
        print(f" - {out_dir / 'bootstrap_cis.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
