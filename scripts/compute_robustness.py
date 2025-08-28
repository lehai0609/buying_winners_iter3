from __future__ import annotations

import argparse
from pathlib import Path
import sys
import yaml
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.robustness import cost_sensitivity as cost_sense, subperiod_metrics as subp_metrics


def load_config(path: str | Path = "config/data.yml") -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config file not found: {p}")
    return yaml.safe_load(p.read_text()) or {}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run robustness diagnostics (M9)")
    ap.add_argument("--config", "-c", default="config/data.yml", help="Path to config YAML")
    ap.add_argument("--params", nargs=2, type=int, metavar=("J", "K"), help="Formation J and holding K months")
    ap.add_argument("--all-combos", action="store_true", help="(Future) evaluate all combos; current script uses provided J,K")
    ap.add_argument("--out-dir", default="data/clean", help="Output directory for CSVs")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)

    J, K = args.params
    params = {"J": int(J), "K": int(K)}

    rob_cfg = (cfg.get("robustness", {}) or {})
    cost_grid = list(rob_cfg.get("cost_bps_grid", [0, 10, 25, 50]))
    subperiods = list(rob_cfg.get("subperiods", []))

    costs_df = cost_sense(params, cost_grid, cfg)
    subs_df = subp_metrics(params, subperiods, cfg) if subperiods else pd.DataFrame()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    costs_df.to_csv(out_dir / "robustness_costs.csv", index=False)
    if not subs_df.empty:
        subs_df.to_csv(out_dir / "robustness_subperiods.csv", index=False)
    print(f"Wrote robustness CSVs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

