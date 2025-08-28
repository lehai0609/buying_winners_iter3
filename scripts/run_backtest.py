from __future__ import annotations

from pathlib import Path
import yaml

from src.backtest import compute_backtest


def main() -> None:
    cfg_path = Path("config/data.yml")
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    compute_backtest(cfg_dict=cfg, write=True)
    print("Backtest complete. Outputs written under data/clean/.")


if __name__ == "__main__":
    main()

