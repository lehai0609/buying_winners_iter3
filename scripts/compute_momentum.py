from __future__ import annotations
import argparse
from pathlib import Path
import sys
import yaml
import pandas as pd

# Ensure `src/` is importable for local runs
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.momentum import compute_momentum_signals


def load_config(path: str | Path = "config/data.yml") -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config file not found: {p}")
    return yaml.safe_load(p.read_text()) or {}


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="M4: Momentum Signal Computation (J-month, deciles)")
    ap.add_argument("--config", "-c", default="config/data.yml", help="Path to config YAML")
    ap.add_argument("--calendar", choices=["union", "vnindex"], default=None, help="Override calendar mode")
    ap.add_argument("--indices-dir", type=Path, default=None, help="Directory containing VN indices (CSV/Parquet)")
    ap.add_argument("--out-parquet", type=Path, default=Path("data/clean/momentum.parquet"), help="Output parquet for signals")
    ap.add_argument("--summary-csv", type=Path, default=Path("data/clean/momentum_summary.csv"), help="Optional summary CSV output")
    ap.add_argument("--dry-run", action="store_true", help="Compute but do not write outputs")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    ns = parse_args(argv or [])
    cfg = load_config(ns.config)
    # Optional override calendar via CLI
    if ns.calendar:
        cfg.setdefault("signals", {}).setdefault("momentum", {})["calendar"] = ns.calendar

    out = compute_momentum_signals(
        df=None,
        cfg_dict=cfg,
        clean_parquet_path=None,
        indices_dir=str(ns.indices_dir) if ns.indices_dir else None,
        write=(not ns.dry_run),
        out_parquet=str(ns.out_parquet),
        summary_csv=str(ns.summary_csv),
    )

    # Brief console summary
    months = pd.to_datetime(out["month_end"]).unique()
    nmons = len(months)
    ntk = out["ticker"].nunique()
    nvalid = out[out["valid"].fillna(False)].shape[0]
    print(f"Momentum computed: {nmons} months, {ntk} tickers, {nvalid} valid rows")
    if ns.dry_run:
        print("Dry run: outputs not written")
    else:
        print(f"Signals parquet: {ns.out_parquet}")
        print(f"Summary CSV: {ns.summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

