from __future__ import annotations
import argparse
from pathlib import Path
import sys
import yaml

from src.experiment_tracking import append_run, get_git_commit, sha256_text


def _load_cfg(path: str | Path) -> dict:
    p = Path(path)
    return yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Log/update an experiment run row in runs.csv")
    parser.add_argument("run_id")
    parser.add_argument("-c", "--config", default="config/data.yml")
    parser.add_argument("--runs-csv", default=None)
    parser.add_argument("--metrics", nargs="*", default=[], help="key=value pairs, e.g., sharpe=1.2 ret_ann=0.18")
    parser.add_argument("--extras", nargs="*", default=[], help="key=value pairs, e.g., note=baseline tag=oos")
    args = parser.parse_args(argv)

    cfg = _load_cfg(args.config)
    tracking_cfg = (cfg.get("tracking") or {}) if isinstance(cfg, dict) else {}
    runs_dir = tracking_cfg.get("runs_dir", "data/clean/experiments")
    runs_csv = args.runs_csv or tracking_cfg.get("runs_csv", str(Path(runs_dir) / "runs.csv"))

    # Parse key=value lists
    def parse_kv(items: list[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for it in items:
            if "=" in it:
                k, v = it.split("=", 1)
                out[k] = v
        return out

    metrics = parse_kv(args.metrics)
    extras = parse_kv(args.extras)

    # Commit and config hash are unknown here (user can pass via extras if needed)
    commit = get_git_commit(short=False)
    cfg_text = Path(args.config).read_text(encoding="utf-8") if Path(args.config).exists() else ""
    cfg_sha = sha256_text(cfg_text) if cfg_text else ""

    append_run(runs_csv, args.run_id, commit, cfg_sha, metrics=metrics, extras=extras)
    print("updated:", runs_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

