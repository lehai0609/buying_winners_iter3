from __future__ import annotations

import argparse
from pathlib import Path
import sys
import yaml
import pandas as pd

# Ensure local src is importable
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.cv import cross_validate


def load_config(path: str | Path = "config/data.yml") -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config file not found: {p}")
    return yaml.safe_load(p.read_text()) or {}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run momentum CV (M9)")
    ap.add_argument("--config", "-c", default="config/data.yml", help="Path to config YAML")
    ap.add_argument("--out-dir", default="data/clean", help="Output directory for CSVs")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    cv_results, cv_selection, cv_oos_summary = cross_validate(cfg)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cv_results.csv").write_text("") if cv_results is None else None
    cv_results.to_csv(out_dir / "cv_results.csv", index=False)
    cv_selection.to_csv(out_dir / "cv_selection.csv", index=False)
    cv_oos_summary.to_csv(out_dir / "cv_oos_summary.csv", index=False)

    # Print brief selection frequency summary
    if not cv_selection.empty:
        freq = cv_selection.groupby(["J", "K"]).size().sort_values(ascending=False)
        print("Selection frequency:\n" + freq.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

