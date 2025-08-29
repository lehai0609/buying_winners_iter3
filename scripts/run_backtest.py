from __future__ import annotations

from pathlib import Path
import sys
import yaml

# Ensure project root (containing `src/`) is importable when running as a script
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.backtest import compute_backtest


def main() -> None:
    cfg_path = Path("config/data.yml")
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    compute_backtest(cfg_dict=cfg, write=True)
    print("Backtest complete. Outputs written under data/clean/.")


if __name__ == "__main__":
    main()
