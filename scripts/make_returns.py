from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd

# Ensure project root (containing `src/`) is importable when running as a script
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_io import load_indices
from src.filters import monthly_universe
from src.returns import (
    daily_simple_returns,
    daily_log_returns,
    forward_returns,
    monthly_returns,
    eligible_monthly_returns,
)


def _read_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"input not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"unsupported input extension: {path.suffix}")


def _write_frame(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"unsupported output extension: {path.suffix}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="M3: Returns & Calendar Utilities Helper")
    ap.add_argument("--in-parquet", type=Path, default=Path("data/clean/ohlcv.parquet"), help="Input OHLCV parquet/csv with MultiIndex [date,ticker] or columns including date,ticker")
    ap.add_argument("--indices-dir", type=Path, default=Path("vn_indices"), help="Directory of index files for VNINDEX calendar")
    ap.add_argument("--calendar", choices=["union", "vnindex"], default="union", help="Trading calendar mode")
    ap.add_argument("--price-col", default="close", help="Price column to use for returns")
    ap.add_argument("--out-daily", type=Path, default=None, help="Output path for daily simple/log returns (CSV/Parquet). If set, writes with columns [date,ticker,ret_1d,ret_log_1d]")
    ap.add_argument("--out-forward", type=Path, default=None, help="Output path for forward returns (CSV/Parquet). Columns fwd_ret_{h}d")
    ap.add_argument("--forward-horizons", type=int, nargs="*", default=[1, 5, 21], help="Forward return horizons in days")
    ap.add_argument("--out-monthly", type=Path, default=Path("data/clean/monthly_returns.csv"), help="Output path for monthly returns (CSV/Parquet)")
    ap.add_argument("--monthly-flags", type=Path, default=None, help="Optional path to monthly eligibility flags [month_end,ticker,eligible] (CSV/Parquet)")
    ap.add_argument("--out-eligible-monthly", type=Path, default=None, help="Output path for eligibility-filtered monthly returns (CSV/Parquet)")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    ns = parse_args(argv or [])
    # Load OHLCV (expect MultiIndex or columns with date,ticker)
    df = _read_frame(ns.in_parquet)
    if not isinstance(df.index, pd.MultiIndex) or list(df.index.names) != ["date", "ticker"]:
        # try to coerce
        if {"date", "ticker"}.issubset(df.columns):
            df["date"] = pd.to_datetime(df["date"], errors="raise")
            df = df.set_index(["date", "ticker"]).sort_index()
        else:
            raise ValueError("Input must be MultiIndex [date,ticker] or contain columns ['date','ticker']")

    indices = None
    if ns.calendar == "vnindex":
        indices = load_indices(str(ns.indices_dir))

    # Daily returns (optional write)
    if ns.out_daily is not None:
        r = daily_simple_returns(df, price_col=ns.price_col)
        lr = daily_log_returns(df, price_col=ns.price_col)
        daily = pd.concat([r, lr], axis=1).reset_index()
        _write_frame(daily, ns.out_daily)

    # Forward returns (optional write)
    if ns.out_forward is not None:
        fwd = forward_returns(df, horizons=ns.forward_horizons, price_col=ns.price_col)
        fwd = fwd.reset_index()
        _write_frame(fwd, ns.out_forward)

    # Monthly returns (always compute; write if path provided)
    mret = monthly_returns(df, calendar=ns.calendar, indices_df=indices, price_col=ns.price_col)
    if ns.out_monthly is not None:
        _write_frame(mret, ns.out_monthly)

    # Eligibility-filtered monthly returns
    if ns.monthly_flags is not None and ns.out_eligible_monthly is not None:
        flags = _read_frame(ns.monthly_flags)
        # Accept either monthly universe (columns) or daily eligibility flags (MultiIndex)
        if {"month_end", "ticker", "eligible"}.issubset(flags.columns):
            # Already monthly universe
            flags["month_end"] = pd.to_datetime(flags["month_end"], errors="raise")
        elif isinstance(flags.index, pd.MultiIndex) and list(flags.index.names) == ["date", "ticker"] and "eligible" in flags.columns:
            # Convert daily eligibility flags to monthly universe
            flags = monthly_universe(flags)
            flags["month_end"] = pd.to_datetime(flags["month_end"], errors="raise")
        else:
            raise ValueError(
                "monthly_flags must have columns ['month_end','ticker','eligible'] or be a daily eligibility parquet indexed by [date,ticker] with column 'eligible'"
            )
        mret_e = eligible_monthly_returns(df, monthly_flags=flags, calendar=ns.calendar, indices_df=indices, price_col=ns.price_col)
        _write_frame(mret_e, ns.out_eligible_monthly)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
