CLI Pipeline Overview (corrected)

Below is the end-to-end pipeline with one-liner commands. Flags and paths match the actual script interfaces in `scripts/`. Commands assume a Poetry shell; prefix with `poetry run` if your venv is not activated.

1) M1 — Ingest & Validate

```bash
poetry run python scripts/make_clean.py \
  --config config/data.yml \
  --hsx-dir HSX \
  --hnx-dir HNX \
  --indices-dir vn_indices \
  --out-parquet data/clean/ohlcv.parquet \
  --coverage-csv data/clean/coverage_summary.csv \
  --anomalies-csv data/clean/anomalies.csv \
  --verbose
```

2) M2 — Eligibility & Monthly Universe

```bash
poetry run python scripts/make_universe.py \
  --in-parquet data/clean/ohlcv.parquet \
  --indices-dir vn_indices \
  --anomalies-csv data/clean/anomalies.csv \
  --flags-parquet data/clean/eligibility.parquet \
  --universe-parquet data/clean/monthly_universe.parquet \
  --lookback-days 126 \
  --min-history-days 126 \
  --min-price-vnd 1000 \
  --min-adv-vnd 100000000 \
  --max-nontrading 15 \
  --calendar union \
  --price-scale 1000 \
  --verbose
```

3) M3 — Returns & Calendar Artifacts

```bash
poetry run python -m scripts.make_returns \
  --in-parquet data/clean/ohlcv.parquet \
  --indices-dir vn_indices \
  --calendar union \
  --out-daily data/clean/daily_returns.parquet \
  --out-forward data/clean/forward_returns.parquet \
  --forward-horizons 1 5 21 \
  --out-monthly data/clean/monthly_returns.csv \
  --monthly-flags data/clean/eligibility.parquet \
  --out-eligible-monthly data/clean/eligible_monthly_returns.parquet
```

Note

- `--monthly-flags` accepts either:
  - a monthly universe with columns `month_end,ticker,eligible` (e.g., `data/clean/monthly_universe.parquet`), or
  - daily eligibility flags indexed by `[date,ticker]` with an `eligible` column (e.g., `data/clean/eligibility.parquet`). The script will down-sample to month-ends automatically.
- Use module form (`-m scripts.make_returns`) so imports resolve reliably across environments.

4) M4 — Momentum Signals (J, skip, deciles)

```bash
poetry run python scripts/compute_momentum.py \
  -c config/data.yml \
  --indices-dir vn_indices \
  --out-parquet data/clean/momentum.parquet \
  --summary-csv data/clean/momentum_summary.csv
```

5) M5 — Portfolio Construction (overlapping K)

```bash
poetry run python scripts/compute_portfolio.py -c config/data.yml
```

6) M6 — Trading Costs (required before backtest)

Compute ADV-based fees/slippage and write `data/clean/portfolio_trades_costed.parquet` and `data/clean/costs_summary.csv`. There is no standalone script, so use a short Python one-liner:

```bash
# Uses defaults from code; to honor config, see the second example.
poetry run python -c "from src.costs import compute_costs; compute_costs(write=True)"

# Read config/data.yml explicitly (recommended)
poetry run python -c "import yaml; from pathlib import Path; from src.costs import compute_costs; cfg=yaml.safe_load(Path('config/data.yml').read_text()) or {}; compute_costs(cfg_dict=cfg, write=True)"
```

7) M7 — Backtest Engine (daily/monthly PnL)

```bash
poetry run python scripts/run_backtest.py
```

8) M8 — Metrics & Alpha Tests

```bash
poetry run python scripts/compute_metrics.py \
  --backtest-monthly data/clean/backtest_monthly.parquet \
  --indices-dir vn_indices \
  --benchmark VNINDEX \
  --use-net
```

9) M9 — Cross-Validation Grid & Robustness

```bash
poetry run python scripts/compute_cv.py -c config/data.yml
poetry run python scripts/compute_robustness.py -c config/data.yml --params 12 6
```

10) M10 — Report & Run Logging

```bash
# Report (saves summary artifacts under data/clean/)
poetry run python scripts/make_report.py -c config/data.yml --run-id momentum_run_YYYYMMDD

# Windows (PowerShell) example for run id
# poetry run python scripts/make_report.py -c config/data.yml --run-id momentum_run_$(Get-Date -Format yyyyMMdd)

# macOS/Linux (bash) example for run id
# poetry run python scripts/make_report.py -c config/data.yml --run-id momentum_run_$(date +%Y%m%d)

# Optional: log metrics to runs.csv
poetry run python scripts/log_experiment.py <run_id> -c config/data.yml --metrics sharpe=1.23 ret_ann=0.18
```

11) Fast Single J/K Run (skip grid search)

If you only want to evaluate one scenario with a chosen formation window J and holding window K (and still get metrics, figures, and a markdown report), use:

```bash
poetry run python scripts/run_single.py --params 12 6 -c config/data.yml
```

Notes

- This runs M4→M8 sequentially with J=12, K=6, writes standard outputs under `data/clean/`, and then assembles a markdown report under `data/clean/report/momentum_report.md`.
- You can optionally tag the run: `--run-id J12_K6`.
- It respects other config settings (e.g., calendar, gap, deciles, costs/backtest options) from `config/data.yml`.

Notes

- Use `--config` for `make_clean.py` (short `-c` is not supported there). Other scripts accept `-c` or `--config`.
- Step 6 is required: `run_backtest.py` expects `data/clean/portfolio_trades_costed.parquet` to exist.
- Index files under `vn_indices/` should include: `VNINDEX.csv`, `HNX-INDEX.csv`, and `VN30.csv`. Names are normalized internally.
- Outputs follow `config/data.yml` under `out.*`; CLI flags above explicitly set common paths for clarity.
