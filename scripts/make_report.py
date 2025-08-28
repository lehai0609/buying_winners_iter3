from __future__ import annotations
import argparse
from pathlib import Path
import sys
import yaml

from src.reports import assemble_report
from src.experiment_tracking import (
    build_manifest,
    write_manifest,
    append_run,
    get_git_commit,
    sha256_text,
)


def _load_cfg(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"config not found: {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Assemble momentum report and log experiment run")
    parser.add_argument("-c", "--config", default="config/data.yml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    cfg = _load_cfg(args.config)
    run_id = args.run_id or Path.cwd().name + "_" + __import__("datetime").datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    # Assemble report (best-effort)
    report = assemble_report(cfg, run_id=run_id)

    # Tracking config
    report_cfg = (cfg.get("report") or {}) if isinstance(cfg, dict) else {}
    tracking_cfg = (cfg.get("tracking") or {}) if isinstance(cfg, dict) else {}
    runs_dir = tracking_cfg.get("runs_dir", "data/clean/experiments")
    runs_csv = tracking_cfg.get("runs_csv", str(Path(runs_dir) / "runs.csv"))
    hash_inputs = bool(tracking_cfg.get("hash_inputs", True))
    hash_outputs = bool(tracking_cfg.get("hash_outputs", True))
    capture_commit = bool(tracking_cfg.get("capture_commit", True))

    # Build manifest
    input_paths = []
    for k, v in (report.get("inputs") or {}).items():
        input_paths.append(v)
    output_paths = [report.get("report_md", "")] + (report.get("figs") or []) + (report.get("tables") or [])
    manifest = build_manifest(input_paths, output_paths, hash_inputs=hash_inputs, hash_outputs=hash_outputs, extra={"run_id": run_id})

    # Write manifest + config snapshot
    cfg_text = Path(args.config).read_text(encoding="utf-8")
    wrote = write_manifest(runs_dir, run_id, manifest, config_snapshot=cfg_text)

    # Append run row
    commit = get_git_commit(short=False) if capture_commit else "unknown"
    cfg_sha = sha256_text(cfg_text)
    headline = {}  # users can update later via log_experiment.py
    extras = {"report_md": report.get("report_md", "")}

    if not args.dry_run:
        append_run(runs_csv, run_id, commit, cfg_sha, metrics=headline, extras=extras)

    # Print summary
    print("report:", report.get("report_md"))
    print("manifest:", wrote.get("manifest"))
    print("runs_csv:", runs_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

