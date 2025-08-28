from __future__ import annotations
import argparse
from pathlib import Path
from typing import List
import sys
import yaml
import pandas as pd

import sys
from pathlib import Path

# Ensure `src/` is on the import path for local execution
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.data_io import load_ohlcv, validate_ohlcv, load_indices
from src.reports import write_coverage, write_anomalies, write_hard_errors


def load_config(path: str | Path = "config/data.yml") -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config file not found: {p}")
    return yaml.safe_load(p.read_text()) or {}


def discover_files(dirs: List[str | Path]) -> list[Path]:
    files: list[Path] = []
    for d in dirs:
        dp = Path(d)
        if not dp.exists():
            raise FileNotFoundError(f"data directory not found: {dp}")
        # accept both parquet and csv in case of mixed sources
        files.extend(sorted(dp.glob("*.parquet")))
        files.extend(sorted(dp.glob("*.csv")))
    if not files:
        raise FileNotFoundError("no input files discovered in provided directories")
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build clean OHLCV and validation reports (M1)")
    parser.add_argument("--config", default="config/data.yml", help="Path to config YAML")
    parser.add_argument("--hsx-dir", default=None, help="Override HSX directory")
    parser.add_argument("--hnx-dir", default=None, help="Override HNX directory")
    parser.add_argument("--indices-dir", default=None, help="Override indices directory")
    parser.add_argument("--start", default=None, help="Override inclusive start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="Override inclusive end date (YYYY-MM-DD)")
    parser.add_argument("--out-parquet", default=None, help="Override output parquet path")
    parser.add_argument("--coverage-csv", default=None, help="Override output coverage CSV path")
    parser.add_argument("--anomalies-csv", default=None, help="Override output anomalies CSV path")
    parser.add_argument("--dry-run", action="store_true", help="Do not write outputs; print summary only")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    raw_dirs = cfg.get("raw_dirs", {})
    hsx_dir = args.hsx_dir or raw_dirs.get("hsx", "HSX")
    hnx_dir = args.hnx_dir or raw_dirs.get("hnx", "HNX")
    indices_dir = args.indices_dir or raw_dirs.get("indices", "vn_indices")
    required_indices = cfg.get("indices_required", ["VNINDEX", "HNX-INDEX", "VN30"])

    out_cfg = cfg.get("out", {})
    out_parquet = args.out_parquet or out_cfg.get("ohlcv_parquet", "data/clean/ohlcv.parquet")
    coverage_csv = args.coverage_csv or out_cfg.get("coverage_csv", "data/clean/coverage_summary.csv")
    anomalies_csv = args.anomalies_csv or out_cfg.get("anomalies_csv", "data/clean/anomalies.csv")
    hard_errors_csv = out_cfg.get("hard_errors_csv", "data/clean/hard_errors.csv")

    hard_errors: list[dict] = []

    # Discover and load OHLCV
    equity_files = discover_files([hsx_dir, hnx_dir])
    if args.verbose:
        print(f"Discovered {len(equity_files)} equity files from {hsx_dir} and {hnx_dir}")

    try:
        df = load_ohlcv(equity_files, start=args.start, end=args.end)
    except Exception as e:
        msg = str(e)
        hard_errors.append({"stage": "load_ohlcv", "error": msg})
        # Attempt diagnostics for duplicates or schema issues
        try:
            req = ["time", "open", "high", "low", "close", "volume"]
            frames = []
            for p in equity_files:
                pth = Path(p)
                try:
                    if pth.suffix.lower() == ".csv":
                        d0 = pd.read_csv(pth)
                    else:
                        d0 = pd.read_parquet(pth)
                    missing = [c for c in req if c not in d0.columns]
                    if missing:
                        hard_errors.append({
                            "stage": "schema_check",
                            "error": f"missing columns {missing}",
                            "file": str(pth)
                        })
                        continue
                    d0 = d0.copy()
                    d0["date"] = pd.to_datetime(d0["time"], errors="coerce")
                    d0["ticker"] = pth.stem.upper()
                    frames.append(d0[["date", "ticker", "open", "high", "low", "close", "volume"]])
                except Exception as ie:
                    hard_errors.append({"stage": "load_file", "error": str(ie), "file": str(pth)})
            if frames:
                dcat = pd.concat(frames, ignore_index=True).set_index(["date", "ticker"])
                dup_mask = dcat.index.duplicated(keep=False)
                if dup_mask.any():
                    dup_rows = dcat[dup_mask].reset_index()
                    dup_rows.insert(2, "stage", "duplicate_date_ticker")
                    dup_rows.insert(3, "error", "duplicate (date,ticker)")
                    write_hard_errors(dup_rows, hard_errors_csv)
                    print(f"Wrote hard error duplicates to {hard_errors_csv}")
        except Exception as de:
            hard_errors.append({"stage": "diagnostics", "error": str(de)})

        # Always persist any hard errors collected and exit non-zero
        if hard_errors:
            write_hard_errors(hard_errors, hard_errors_csv)
            print(f"Wrote hard errors report to {hard_errors_csv}")
        return 1
    if args.verbose:
        dmin = df.index.get_level_values("date").min()
        dmax = df.index.get_level_values("date").max()
        print(f"Loaded OHLCV: {len(df):,} rows across {df.index.get_level_values('ticker').nunique()} tickers; dates {dmin.date()}..{dmax.date()}")

    clean_df, anoms = validate_ohlcv(df)

    # Collect hard removals from validation (e.g., price_non_positive)
    try:
        if isinstance(anoms, pd.DataFrame) and not anoms.empty:
            hard_mask = anoms.get("rule", pd.Series(dtype=object)).astype(str) == "price_non_positive"
            if hard_mask.any():
                hard_rows = anoms.loc[hard_mask].copy()
                for _, row in hard_rows.iterrows():
                    hard_errors.append({
                        "stage": "validate_ohlcv",
                        "error": "price_non_positive",
                        "date": str(row.get("date", "")),
                        "ticker": row.get("ticker", ""),
                        "open": row.get("open", None),
                        "high": row.get("high", None),
                        "low": row.get("low", None),
                        "close": row.get("close", None),
                        "volume": row.get("volume", None),
                    })
    except Exception as e:
        # Non-fatal; continue with pipeline
        print(f"warning: failed to collect hard removals: {e}")

    # Load indices and report alignment (no persistence)
    try:
        indices_df = load_indices(indices_dir, names=required_indices)
        eq_dates = set(clean_df.index.get_level_values("date").unique())
        if args.verbose:
            print(f"Loaded indices: {', '.join(required_indices)}; total rows: {len(indices_df):,}")
        for nm in required_indices:
            sub = indices_df[indices_df["index"] == nm]
            share = len(set(sub["date"]) & eq_dates) / max(1, len(eq_dates))
            print(f"Index alignment: {nm}: {share:.1%} of equity dates present")
    except Exception as e:
        msg = str(e)
        hard_errors.append({"stage": "load_indices", "error": msg})
        print(f"Index ingest warning: {e}")

    if args.dry_run:
        # Summary only
        print("Dry run: no files written")
        print("Anomalies by rule:")
        if len(anoms) == 0:
            print("  (none)")
        else:
            print(anoms.groupby("rule").size().sort_values(ascending=False).to_string())
        return 0

    # Write outputs
    out_parquet_path = Path(out_parquet)
    out_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_parquet(out_parquet_path)

    write_coverage(clean_df, coverage_csv)
    write_anomalies(anoms, anomalies_csv)
    if hard_errors:
        write_hard_errors(hard_errors, hard_errors_csv)
        print(f"Wrote hard errors report to {hard_errors_csv}")

    print(f"Wrote clean OHLCV to {out_parquet_path}")
    print(f"Wrote coverage report to {coverage_csv}")
    print(f"Wrote anomalies report to {anomalies_csv}")
    return 0 if not hard_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
