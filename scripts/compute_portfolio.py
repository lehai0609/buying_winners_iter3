from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

# Ensure project root (containing `src/`) is importable when running as a script
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from src.portfolio import compute_portfolio, holdings_to_trades


def main() -> int:
    p = argparse.ArgumentParser(description="Compute overlapping K-month momentum portfolio (long-only)")
    p.add_argument("-c", "--config", default="config/data.yml", help="Path to config YAML (default: config/data.yml)")
    p.add_argument("--dry-run", action="store_true", help="Run without writing outputs")
    args = p.parse_args()

    cfg = {}
    cfg_path = Path(args.config)
    if cfg_path.exists() and yaml is not None:
        try:
            cfg = yaml.safe_load(cfg_path.read_text()) or {}
        except Exception:
            cfg = {}

    holdings = compute_portfolio(cfg_dict=cfg, signals_df=None, write=not args.dry_run)
    trades = holdings_to_trades(holdings)

    # Print a brief summary
    months = pd.to_datetime(holdings["month_end"]) if not holdings.empty else pd.Series(dtype="datetime64[ns]")
    uniq_months = int(months.nunique()) if len(months) else 0
    uniq_names = int(holdings["ticker"].nunique()) if not holdings.empty else 0
    ttl_w = float(holdings.groupby("month_end")["weight"].sum().mean()) if not holdings.empty else 0.0
    print(f"Portfolio computed for {uniq_months} months, {uniq_names} names; avg total weight ~ {ttl_w:.3f}.")
    if args.dry_run:
        print("Dry run: no files written.")

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
