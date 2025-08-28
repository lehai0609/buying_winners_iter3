from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd

# Ensure `src/` is on the import path for local execution
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.filters import eligibility_flags, monthly_universe
from src.data_io import load_indices


def _read_ohlcv_parquet(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"input parquet not found: {p}")
    df = pd.read_parquet(p)
    # Expect MultiIndex [date, ticker]; if not, try to coerce
    if not isinstance(df.index, pd.MultiIndex) or list(df.index.names) != ["date", "ticker"]:
        if {"date", "ticker"}.issubset(df.columns):
            df["date"] = pd.to_datetime(df["date"], errors="raise")
            df = df.set_index(["date", "ticker"]).sort_index()
        else:
            raise ValueError("ohlcv parquet must be indexed by [date,ticker] or contain 'date' and 'ticker' columns")
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute eligibility flags and monthly universe (M2)")
    parser.add_argument("--in-parquet", default="data/clean/ohlcv.parquet", help="Path to M1 clean OHLCV parquet")
    parser.add_argument("--indices-dir", default="vn_indices", help="Path to indices directory (when calendar=vnindex)")
    parser.add_argument("--anomalies-csv", default=None, help="Optional anomalies CSV from M1 to use for quality screen")
    parser.add_argument("--flags-parquet", default="data/clean/eligibility.parquet", help="Output daily eligibility flags parquet")
    parser.add_argument("--universe-parquet", default="data/clean/monthly_universe.parquet", help="Output monthly universe parquet")
    parser.add_argument("--calendar", choices=["union", "vnindex"], default="union", help="Trading grid construction")
    parser.add_argument("--lookback-days", type=int, default=126)
    parser.add_argument("--min-history-days", type=int, default=126)
    parser.add_argument("--min-price-vnd", type=float, default=1000.0)
    parser.add_argument("--min-adv-vnd", type=float, default=100_000_000.0)
    parser.add_argument("--max-nontrading", type=int, default=15)
    parser.add_argument("--price-scale", type=float, default=1.0, help="Multiply prices by this factor before ADV/threshold checks (e.g., 1000 for kVND→VND)")
    parser.add_argument("--dry-run", action="store_true", help="Do not write outputs; print summary only")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args(argv)

    df = _read_ohlcv_parquet(args.in_parquet)

    indices_df = None
    if args.calendar == "vnindex":
        indices_df = load_indices(args.indices_dir, names=["VNINDEX"])  # only need VNINDEX for calendar

    anomalies_df = None
    if args.anomalies_csv:
        ap = Path(args.anomalies_csv)
        if ap.exists():
            anomalies_df = pd.read_csv(ap)
        else:
            print(f"warning: anomalies CSV not found at {ap}; proceeding without quality screen from anomalies")

    flags = eligibility_flags(
        df,
        lookback_days=args.lookback_days,
        min_history_days=args.min_history_days,
        min_price_vnd=args.min_price_vnd,
        min_adv_vnd=args.min_adv_vnd,
        max_nontrading_days=args.max_nontrading,
        calendar=args.calendar,
        indices_df=indices_df,
        price_scale=args.price_scale,
        anomalies_df=anomalies_df,
    )

    uni = monthly_universe(flags)

    if args.verbose:
        dmin = flags.index.get_level_values("date").min()
        dmax = flags.index.get_level_values("date").max()
        n_tickers = flags.index.get_level_values("ticker").nunique()
        print(f"Flags computed: dates {dmin.date()}..{dmax.date()}, tickers={n_tickers}, rows={len(flags):,}")
        last_month = None
        if len(uni) > 0:
            last_month = pd.to_datetime(uni["month_end"]).max().date()
        print(f"Monthly universe rows: {len(uni):,}; last formation month: {last_month}")

    if args.dry_run:
        # Print a brief summary
        if len(uni) > 0:
            counts = uni.groupby("month_end")["eligible"].sum().tail(3)
            print("Eligible counts for last 3 formation months:")
            print(counts.to_string())
        return 0

    # Write outputs
    flags_path = Path(args.flags_parquet)
    uni_path = Path(args.universe_parquet)
    flags_path.parent.mkdir(parents=True, exist_ok=True)
    uni_path.parent.mkdir(parents=True, exist_ok=True)
    flags.to_parquet(flags_path)
    uni.to_parquet(uni_path, index=False)
    print(f"Wrote eligibility flags to {flags_path}")
    print(f"Wrote monthly universe to {uni_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
