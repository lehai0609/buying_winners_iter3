from __future__ import annotations
from pathlib import Path
from typing import Iterable, Any
import hashlib
import json
import os
import sys
import subprocess
import csv
import datetime as _dt


def _ensure_dir(p: str | Path) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def get_git_commit(short: bool = False) -> str:
    try:
        cmd = ["git", "rev-parse", "--short" if short else "HEAD"]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        return out
    except Exception:
        return "unknown"


def now_utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def build_manifest(
    inputs: Iterable[str | Path] = (),
    outputs: Iterable[str | Path] = (),
    hash_inputs: bool = True,
    hash_outputs: bool = True,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    def _collect(paths: Iterable[str | Path], do_hash: bool) -> list[dict[str, str]]:
        records: list[dict[str, str]] = []
        for p in paths:
            pp = Path(p)
            if not pp.exists():
                records.append({"path": str(pp), "exists": "false"})
                continue
            rec = {"path": str(pp), "exists": "true"}
            if do_hash and pp.is_file():
                try:
                    rec["sha256"] = sha256_file(pp)
                except Exception:
                    rec["sha256"] = "error"
            records.append(rec)
        return records

    return {
        "created_at": now_utc_iso(),
        "inputs": _collect(inputs, hash_inputs),
        "outputs": _collect(outputs, hash_outputs),
        "extra": extra or {},
    }


def write_manifest(base_dir: str | Path, run_id: str, manifest: dict[str, Any], config_snapshot: str | None = None) -> dict[str, str]:
    base = _ensure_dir(Path(base_dir) / "runs" / run_id)
    man_path = base / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    paths = {"manifest": str(man_path)}
    if config_snapshot is not None:
        cfg_path = base / "config_snapshot.yml"
        cfg_path.write_text(config_snapshot, encoding="utf-8")
        paths["config_snapshot"] = str(cfg_path)
    return paths


def append_run(
    runs_csv: str | Path,
    run_id: str,
    commit: str,
    config_sha256: str,
    metrics: dict[str, Any] | None = None,
    extras: dict[str, Any] | None = None,
) -> None:
    path = Path(runs_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "run_id",
        "commit",
        "config_sha256",
    ]
    # Flatten metrics and extras with simple prefixes
    metrics = metrics or {}
    extras = extras or {}
    for k in metrics.keys():
        fieldnames.append(f"metric_{k}")
    for k in extras.keys():
        fieldnames.append(f"extra_{k}")

    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        row = {
            "timestamp": now_utc_iso(),
            "run_id": run_id,
            "commit": commit,
            "config_sha256": config_sha256,
        }
        for k, v in (metrics or {}).items():
            row[f"metric_{k}"] = v
        for k, v in (extras or {}).items():
            row[f"extra_{k}"] = v
        w.writerow(row)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

