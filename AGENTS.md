# Repository Guidelines

## Project Structure & Module Organization
- `src/`: Source modules imported as `from src.<module> import ...` (e.g., `src/utils.py`). Add new code here (e.g., `src/data_io.py`).
- `tests/`: PyTest suite (e.g., `tests/test_data_io.py`, `tests/test_repro.py`). Name files `test_*.py` and functions `test_*`.
- `config/`: Runtime/config inputs (e.g., `config/data.yml`). Tests validate its schema via Pydantic (`data_dir`, `ohlcv_glob`, `min_volume >= 0`).
- `scripts/`: Optional helper scripts for local tasks.
- `HNX/`: Source of .parquet files for historical stock OHLCV on HNX exchange.
- `HSX/`: Source of .parquet files for historical stock OHLCV on HSX exchange.
- `vn_indices/`: Three CSV files for index OHLCV: `VNINDEX.csv`, `HNX-INDEX.csv`, `VN30.csv`.
- `data/clean/`: Outputs from M1 ingest: `ohlcv.parquet`, `coverage_summary.csv`, `anomalies.csv`, `hard_errors.csv`.
- `.venv/`: In-project Poetry virtualenv (not committed).

## Setup, Build, and Test Commands
- `poetry install`: Install dependencies into `.venv` (Python 3.11).
- `poetry run pytest -q`: Run the test suite quietly; use `-k <expr>` to filter.
- `pytest -q`: Same as above if the virtualenv is already activated (`poetry shell`).
- `poetry build` (optional): Build a distributable package; not required for local dev.

## Coding Style & Naming Conventions
- **Indentation**: 4 spaces; prefer type hints and short, clear docstrings.
- **Naming**: modules/functions/variables use `snake_case`; classes `PascalCase`; constants `UPPER_SNAKE_CASE`.
- **Imports**: Use absolute imports from `src` (e.g., `from src.utils import set_seed`).
- **Reproducibility**: Prefer `set_seed`, `make_rng`, and `fixed_seed` from `src/utils.py` over ad‑hoc RNG usage.

## Testing Guidelines
- **Framework**: PyTest. Keep tests deterministic and fast.
- **Layout**: Place tests under `tests/`; mirror module names where helpful.
- **Running**: `pytest tests/test_repro.py -q` (single file) or `pytest -k ohlcv -q` (filtered).
- **Coverage**: Aim for high coverage (≈80%+ when practical). Add tests with new features and bug fixes.

## Commit & Pull Request Guidelines
- **Commits**: Imperative, concise (<72 chars). Optionally include a scope/milestone prefix (e.g., `M0:`), as seen in history.
- **Pull Requests**: Provide summary, rationale, linked issues, test plan/output (`pytest -q` passing), and note any config changes. Small, focused PRs are preferred.

## Security & Configuration Tips
- Avoid committing data or secrets. `config/data.yml` should define: `data_dir`, `ohlcv_glob` (default `*.parquet`), and non‑negative `min_volume`.
- Keep environments isolated via `.venv`; rely on `poetry.lock` for repeatable installs.
- Outputs (config `out.*`):
  - `ohlcv_parquet`: clean, validated price data (keeps flagged rows).
  - `coverage_csv`: yearly counts and missing grid days.
  - `anomalies_csv`: flagged soft issues with OHLCV snapshot.
  - `hard_errors_csv`: hard failures (e.g., duplicates, schema issues) for audit.
