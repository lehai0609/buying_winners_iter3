Concise checklist (key subtasks)
- Stand up clean, versioned OHLCV data and eligibility filters for HOSE/HNX (2010–2025), with no look-ahead.
- Implement J-month formation with 5-day skip and K-month overlapping long-only winner portfolios; parameterize J,K.
- Build backtester with monthly rebalancing, T+2 execution convention, transaction costs and slippage; verify no leakage.
- Compute risk/return metrics, Newey-West alpha tests, and robustness (J×K grid, walk-forward, subperiods).
- Track experiments (configs, seeds, metrics) and produce reproducible reports.
- Document assumptions, validation at each step, and escalation for data/market edge cases.

Lean Implementation Plan (Quant Trading Prototype, TDD)

Status Summary (Current vs Plan)
- Completed: M0 Environment/Repro; M1 Data ingest & validation (OHLCV + indices), plus coverage/anomalies/hard-errors writers; M2 Eligibility & cleaning; M3 Returns & calendar utilities; M4 Momentum signal computation (J-month, deciles); M5 Portfolio construction (overlapping K-month, long-only); M6 Trading frictions (costs, slippage, impact); M7 Backtesting engine; M8 Metrics & statistical testing (perf summary, Newey–West alpha, bootstrap CIs, subperiods); M9 Robustness & cross-validation (grid search, walk-forward CV, cost sensitivity, subperiod metrics).
- Planned: M10 reporting/tracking remains to implement.
- Differences from initial plan:
  - Index loading implemented as `load_indices(dir_path, names)` returning a long DataFrame; use `get_index_series(indices_df, name)` for a single series (instead of a singular `load_index`).
  - Validation API returns a tuple `(clean_df, anomalies_df)` from `validate_ohlcv` (previously described as returning a single DataFrame).
  - Coverage/anomaly/hard-error CSV writers live in `src/reports.py` (not `report.py`).
  - Only `config/data.yml` exists today; `strategy.yml`, `backtest.yml`, and `report.yml` will be added when M4–M10 land.
  - Loader accepts CSV or Parquet and infers `ticker` from filenames under `HSX/` and `HNX/`; indices are loaded from `vn_indices/` with case/format-insensitive name normalization (e.g., `HNX-INDEX` vs `HNXINDEX`).
  - Update: With M4 complete, these configs will be added during M5–M10.
  - New (M4): Introduced `src/calendar.py` and calendar-aware `returns.monthly_returns` supporting `union` and `vnindex` grids with as-of month-end sampling.
  - New (M4): Added `signals.momentum.*` under `config/data.yml` with defaults (`lookback_months`, `gap_months`, `n_deciles`, `min_months_history`, `min_names_per_month`, `exclude_hard_errors`, optional `calendar`, `price_col`).
  - New (M4): Implemented `src/momentum.py` (`compute_momentum`, `assign_deciles`, `compute_momentum_signals`) and CLI `scripts/compute_momentum.py`; outputs `data/clean/momentum.parquet` (+ optional summary CSV).
  - New (M4): Deterministic ranking: `pct_rank` via average-tie ranks; deciles via `qcut(duplicates="drop")` with a rank-based fallback when unique values < bins.
  - New (M4): Optional exclusion of rows that appear in `hard_errors.csv` prior to monthly aggregation when configured.

0. Scope
- Goal: Replicate and adapt Jegadeesh & Titman (1993) price-momentum to Vietnam’s long-only constraints; test if past winners continue to outperform using only price data.
- Universe: HOSE & HNX common stocks; minimum 6 months trading history; average daily trading value > 100M VND; price > 1,000 VND; remove instruments with excessive non-trading days (>15 in formation windows).
- Horizon: Monthly formation and rebalancing; daily data used to compute returns and skip period.
- Success Gates (OOS 2020–2025):
  - Sharpe > 0.8; IR > 0.5 vs VN-Index; Max DD <> 25%; positive monthly hit-rate > 60%.
  - Statistical: Newey-West t-stat of monthly alpha ≥ 1.96 for top-decile long-only or for long-minus-benchmark excess returns.
- Baseline: Buy-and-hold VN-Index; equal-weight eligible universe (rebalanced monthly); naive 12-1 momentum without skip as a sanity baseline.

1. Minimal Project Structure
project/
- README.md
- requirements.txt / pyproject.toml
- config/
  - data.yml              # paths, calendars, filters
  - strategy.yml          # J,K, skip, deciles, sizing rules
  - backtest.yml          # costs, slippage, impact, capital
  - report.yml            # plots, tables
- data/{raw,clean,features,interim}
- src/
  - data_io.py            # load & validate OHLCV
  - filters.py            # eligibility & quality filters (VN-specific)
  - returns.py            # daily/monthly returns, skip logic
  - momentum.py           # J-month scores, ranking, deciles
  - portfolio.py          # overlapping K-month, weights, trades
  - costs.py              # fees, slippage, impact
  - backtest.py           # simulation engine
  - metrics.py            # perf metrics, drawdowns
  - stats.py              # alpha regressions, NW-SE, bootstrap
  - cv.py                 # J–K grid & walk-forward
  - report.py             # tables/plots/export
  - experiment_tracking.py# MLflow or CSV logger
  - utils.py              # calendars, helpers, config parsing
- tests/
  - test_data_io.py
  - test_filters.py
  - test_returns.py
  - test_momentum.py
  - test_portfolio.py
  - test_costs.py
  - test_backtest.py
  - test_metrics.py
  - test_stats.py
  - test_cv.py
- scripts/
  - make_clean.py
  - run_backtest.py
  - grid_search.py
  - make_report.py
- notebooks/               # EDA, sanity checks
2. Build Order (TDD Workflow with dependencies and validations)
M0. Environment and Repro
- Description: Set up environment; deterministic seeds; CI test harness (pytest).
- Dependencies: None.
- Validation:
  - Pytest discovers and runs tests locally and in CI in <60s.
  - Repro seed ensures identical metrics across runs given same config.
- Tests:
  - tests bootstrap; config schema loads and validates.
  - Deterministic sample function returns same output across runs with fixed seed.
 - Status: DONE (see `src/utils.py` and `tests/test_repro.py`).

M1. Data ingest & validation (OHLCV, index)
- Description: Load CSV files (2010–2025) with adjusted closing prices; VN-Index/HNX-Index series.
- Dependencies: M0.
- Validation:
  - No missing or non-positive OHLC; zero/NaN volume flagged; OHLC relationship sanity checks pass; date-sid index uniqueness enforced.
  - Extreme daily moves flagged; coverage summary matches expected ticker counts by year.
- Tests:
  - Missing values raise assert; duplicate (date,ticker) rejected.
  - Sample with known splits retains continuity due to adjusted close.
  - Date ranges return expected row counts.
 - Status: DONE.
 - Notes: Implemented `load_ohlcv(paths, start=None, end=None)`, `validate_ohlcv(df) -> (clean_df, anoms_df)`, `load_indices(dir_path, names) -> DataFrame[date,index,close]`, and `get_index_series(indices_df, name) -> Series`. Coverage/anomalies/hard-errors writers are in `src/reports.py`.

Current Project Structure (as built)
- Config: `config/data.yml` (paths, thresholds, outputs)
- Code: `src/data_io.py`, `src/reports.py`, `src/utils.py`, `src/filters.py`, `src/calendar.py`, `src/returns.py`, `src/momentum.py`
- Tests: `tests/test_config.py`, `tests/test_data_io.py`, `tests/test_validation.py`, `tests/test_index_loader.py`, `tests/test_reports.py`, `tests/test_repro.py`, `tests/test_filters.py`, `tests/test_returns.py`, `tests/test_momentum.py`
- Data layout: raw `HSX/` and `HNX/` per-ticker files; `vn_indices/` holds index CSVs
- Scripts: `scripts/compute_momentum.py` (signal computation)
- Planned (not yet present): `cv.py`, `report.py`, `experiment_tracking.py`; `strategy.yml`, `backtest.yml`, `report.yml`

M2. Eligibility & cleaning (Vietnam-specific)
- Description: Apply rules: min 6 months history; price ≥ 1,000 VND; ADV ≥ 100M VND; ≤15 non-trading days in formation windows; optional market cap if available; remove halted/suspended periods.
- Dependencies: M1.
- Validation:
  - Universe list per month generated; spot-check tickers pass/fail criteria correctly.
  - Survivorship-bias avoidance: universe is determined only using information up to each formation date.
- Tests:
  - Synthetic data cases confirm filters (e.g., ADV threshold) work; no forward-looking checks.
 - Status: DONE (see `src/filters.py` and `tests/test_filters.py`).

M3. Returns & calendar utilities
- Description: Compute daily simple/ln returns; monthly periodization; 5-day skip logic to avoid microstructure reversal; monthly aggregation schedule.
- Dependencies: M1, M2.
- Validation:
  - Monthly returns from daily data equal product of (1+daily)−1; skip window excludes last 5 trading days prior to ranking.
- Tests:
  - Known sequences yield exact monthly and skip-adjusted results; no leakage across month end./
 - Status: DONE (see `src/returns.py`, `src/calendar.py`, `tests/test_returns.py`).

M4. Momentum signal computation (J-month, deciles)
- Description: For each formation month t, compute cumulative return from t-J months up to t-5 trading days; rank into 10 deciles; for long-only take top decile (D10).
- Dependencies: M2, M3.
- Validation:
  - Distribution of momentum scores reasonable; deciles approximately balance counts; ties handled deterministically.
- Tests:
  - Toy example replicates J&V-style 12-1 momentum; skip implemented; ranks stable given same inputs.
 - Status: DONE (see `src/momentum.py`, `tests/test_momentum.py`, `scripts/compute_momentum.py`).

M5. Portfolio construction (overlapping K-month holding, long-only)
- Description: Implement K overlapping winner portfolios; equal-weight within cohort; monthly rebalance; T+2 execution assumption; enforce constraints (no short, sum weights ≤ 1, turnover bounded).
- Dependencies: M4.
- Validation:
  - Active weights equal average of K cohorts; weights sum to ≤ 1 and are non-negative.
- Tests:
  - Overlap math: with K=3, each cohort contributes 1/3; opening/closing flows match schedule; cash conserved.
 - Status: DONE (see `src/portfolio.py`, `tests/test_portfolio.py`, `scripts/compute_portfolio.py`). `exclude_on_missing_price` enforced via a monthly activity mask derived from `data/clean/ohlcv.parquet`; optional summary now includes `cash_weight`.

M6. Trading frictions (costs, slippage, impact)
- Description: Apply per-side transaction costs (default 25 bps), linear slippage as function of participation rate to ADV, optional market impact (e.g., 10 bps for large orders).
- Dependencies: M5.
- Validation:
  - Higher turnover increases costs; costless run equals gross returns; unit tests for corner cases (illiquid names).
- Tests:
  - Trades at zero not charged; doubling turnover roughly doubles costs; caps on slippage applied.
 - Status: DONE (see `src/costs.py`, `tests/test_costs.py`). Outputs: `data/clean/portfolio_trades_costed.parquet`, `data/clean/costs_summary.csv`. ADV-based slippage supported with `capital_vnd` and caps; threshold impact optional.

M7. Backtesting engine
- Description: Monthly loop: form ranks at month-end t, trade at t+1 open or VWAP proxy, hold K months with overlaps; incorporate T+2 where needed; update PnL daily/monthly; handle halts (no trade) and price limits (±7%) approximated via capped daily returns.
- Dependencies: M5, M6.
- Validation:
  - No look-ahead (signals computed only from data available at formation); equity curve matches hand-calculated toy example; cash ≥ 0 unlevered.
- Tests:
  - Trade timing validated using lag; halted stocks: orders skipped, weights renormalized; price limit saturation respected.
 - Status: DONE (see `src/backtest.py`, `tests/test_backtest.py`, `scripts/run_backtest.py`). Outputs include `data/clean/backtest_daily.parquet` and `data/clean/backtest_monthly.parquet`.

M8. Metrics & statistical testing
- Description: Compute CAGR, vol, Sharpe, IR vs VN-Index, max drawdown, turnover, VaR; regress excess returns on market (and optional factors) to estimate alpha with Newey-West SE; bootstrap CIs; subperiod stats.
- Dependencies: M7.
- Status: DONE (see `src/metrics.py`, `src/stats.py`, `scripts/compute_metrics.py`, `tests/test_metrics.py`, `tests/test_stats.py`). Outputs include `data/clean/metrics_summary.csv`, `data/clean/alpha_newey_west.csv`, `data/clean/metrics_subperiods.csv` (+ optional `data/clean/bootstrap_cis.csv`). [Verified via passing tests in repo.]
- Validation:
  - Metrics match numpy/pandas references on synthetic series; NW t-stats align with statsmodels on fixture data.
- Tests:
  - Max DD on known series; IR with benchmark; NW-OLS recovers known alpha in simulated AR(1) noise.

M9. Robustness & CV
- Status: Completed — implemented cross-validation and robustness per m_9.md (src/cv.py, src/robustness.py; CLIs; config defaults added).
- Description: Grid search J ∈ {3,6,9,12}, K ∈ {1,3,6,12}; walk-forward with 36m train, 12m validate; sensitivity to costs; subperiods (pre-COVID, COVID crash, recovery).
- Dependencies: M8.
- Validation:
  - Config space explored; best (J,K) stable across folds; sensitivity plots generated.
- Tests:
  - Reproducible selection given seed; parameter that overfits in-sample underperforms out-of-sample in synthetic test.

M10. Reporting & tracking
- Description: Log parameters/metrics (MLflow or CSV); export figures/tables; generate research report with assumptions and limitations.
- Dependencies: M8, M9.
- Validation:
  - Report includes all required sections; artifacts saved with run IDs and hashes.
- Tests:
  - Presence and schema of outputs; missing metric causes test failure.

3. Detailed Module Specs (TDD: inputs, outputs, logic, tests)

Module: data_io.py
- Purpose: Load and validate OHLCV and index data.
- Functions:
  - load_ohlcv(paths: list[str|Path], start: str|Timestamp|None, end: str|Timestamp|None) -> pd.DataFrame
    - Output: MultiIndex [date, ticker], columns [open, high, low, close, volume]; typed and sorted; duplicates rejected.
  - validate_ohlcv(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]
    - Output: (clean_df clipped to config date window, anomalies_df with rule/detail and OHLCV snapshot). Hard errors raise.
  - load_indices(dir_path: str|Path, names: list[str]|None = ["VNINDEX","HNX-INDEX","VN30"]) -> pd.DataFrame
    - Output: long DataFrame with columns [date, index, close]; duplicates on (date,index) rejected; accepts CSV/Parquet; name normalization tolerant to hyphens/underscores/case.
  - get_index_series(indices_df: pd.DataFrame, name: str) -> pd.Series
    - Output: one index close series (date-indexed), for downstream alignment/benchmarking.
- Error handling: If columns missing, escalate to data provider; if extreme moves cluster, flag possible adjustment issues.

Module: reports.py
- Purpose: Persist summary artifacts for auditing.
- Functions:
  - write_coverage(df: pd.DataFrame, out_path: str|Path) -> None — yearly rows/tickers/missing_days from OHLCV
  - write_anomalies(anoms: pd.DataFrame, out_path: str|Path) -> None — schema [date,ticker,rule,...]
  - write_hard_errors(errors: pd.DataFrame|list[dict], out_path: str|Path) -> None — audit trail for hard failures

Module: utils.py
- Purpose: Reproducibility helpers (seed control).
- Functions: `set_seed`, `make_rng`, `fixed_seed` (see `tests/test_repro.py`).

Module: filters.py
- Purpose: Apply Vietnam-specific eligibility filters.
- Functions:
  - min_history(df, min_days: int=126) -> pd.Index (eligible tickers by date)
  - price_floor(df, min_price: int=1000) -> mask
  - adv_filter(df, min_adv_vnd: float=1e8, window: int=60) -> mask
  - non_trading_days_filter(df, max_ntd: int=15, window_days: int=J*21) -> mask
  - compose_universe(df, rules: dict, monthly: bool=True) -> pd.DataFrame[bool]
- Outputs: Monthly boolean eligibility matrix indexed by [month_end, ticker].
- Logic: Compute rolling stats only up to formation dates; combine masks with AND.
- Tests:
  - Synthetic ADV and price cases; ensure no look-ahead (use shift).

Module: returns.py
- Purpose: Compute returns and calendar transforms.
- Functions:
  - daily_returns(df) -> pd.DataFrame[ret_d]
  - monthly_calendar(df) -> pd.Series[month_end_flags]
  - cum_return_skip(df, J_months: int, skip_days: int=5) -> pd.DataFrame[mom_window_ret]
    - Logic: For each formation month-end t, compute product(1+ret_d) from (t-J months end − skip_days) to (t − skip_days) − 1.
  - month_returns_from_daily(df) -> pd.DataFrame[ret_m]
- Tests:
  - Product-of-dailies equals monthly; skip excludes last 5 trading days; off-by-one boundaries verified.

Module: momentum.py
- Purpose: Compute momentum scores and ranks.
- Functions:
  - momentum_scores(ret_d, universe_mask, J, skip_days) -> pd.DataFrame[score]
  - decile_ranks(scores, by_month=True, q=10, method="ordinal") -> pd.DataFrame[rank 0..9]
  - top_decile_mask(ranks, decile=9) -> pd.DataFrame[bool]
- Logic: Score computed only where universe==True; ranks computed cross-sectionally per month.
- Tests:
  - Known toy example yields expected ranks; ties stable; no scores for ineligible assets; no leakage.

Module: portfolio.py
- Purpose: Build overlapping K-month winner portfolios and trade lists.
- Functions:
  - build_cohorts(top_mask, K) -> dict[cohort_start_month -> set[tickers]]
  - target_weights_from_cohorts(cohorts, K, constraints: dict) -> pd.DataFrame[weights]
    - Constraints: long-only, sum(weights) ≤ 1, per-name cap optional.
  - generate_trades(prev_weights, target_weights, prices, calendar, t_plus=1, settlement="T+2") -> pd.DataFrame[trades]
- Logic: Each month, open new cohort at 1/K weight, hold previous K−1 cohorts; equal-weight within cohort; renormalize for halts.
- Tests:
  - Weight sums; cohort aging; trade directions; T+2 scheduling unit test.
- Error handling: If halts or price limit days prevent fills, log and adjust weights next day.

Module: costs.py
- Purpose: Apply fees, slippage, and impact.
- Functions:
  - apply_costs(trades, adv, fee_bps=25, slip_model="linear", slip_params: dict) -> pd.Series[costs_vnd]
    - Logic: cost = abs(notional) * (fee_bps + slippage(participation)) + impact if applicable.
- Tests:
  - Zero trades â†’ zero cost; monotonic in turnover; caps respected.

Module: backtest.py
- Purpose: Simulate PnL with correct timing and constraints.
- Functions:
  - run_backtest(prices_d, universe_m, J, K, skip_days, config) -> dict
    - Output keys: equity_curve, monthly_returns, weights, trades, turnover, costs, gross/net returns.
  - simulate_daily(prices_d, weights, trades, costs, limits_model) -> pd.DataFrame[pnl]
- Logic: At each rebalance month-end, compute ranks using data up to t−skip; trade at t+1 (or per config); daily PnL accrues; respect price limit approximations (cap ret_d at ±7%).
- Tests:
  - No look-ahead: shift checks; cash non-negative; reconciliation between gross − costs = net.

Module: metrics.py
- Purpose: Compute performance and risk metrics.
- Functions:
  - perf_summary(returns_m, benchmark_m, rf=0) -> dict{CAGR, vol, Sharpe, IR, maxDD, beta, VaR}
  - drawdown_stats(returns_m) -> dict{mdd, duration}
  - turnover_stats(trades) -> dict{avg_turnover}
- Tests:
  - Validate against known arrays; edge cases (flat series).

Module: stats.py
- Purpose: Statistical inference.
- Functions:
  - alpha_newey_west(returns_m, benchmark_m, lags=6) -> dict{alpha, t_alpha, p_value}
  - bootstrap_cis(returns_m, n=1000, block_size=6) -> dict{Sharpe_CI, alpha_CI}
  - subperiod_tests(returns_m, periods: list) -> pd.DataFrame
- Tests:
  - Recover known alpha in simulated data; NW-robust SEs match statsmodels.

Module: cv.py
- Purpose: Parameter search and walk-forward.
- Functions:
  - grid_search(prices_d, universe_m, J_list, K_list, config) -> pd.DataFrame
  - walk_forward(prices_d, universe_m, train_months=36, valid_months=12, step=1) -> pd.DataFrame
- Tests:
  - Deterministic results with fixed seed/config; sanity check that extreme costs penalize high turnover combos.

Module: report.py
- Purpose: Produce research outputs.
- Functions:
  - generate_report(results: dict, config: dict, out_dir: str) -> None
    - Outputs: HTML/Markdown summary, CSV of monthly returns, plots (equity, drawdown, bar charts), tables (J×K heatmap).
- Tests:
  - All required files produced; missing metrics triggers fail.

Module: experiment_tracking.py
- Purpose: Lightweight logging.
- Functions:
  - start_run(config) -> run_id
  - log_params_metrics(run_id, params: dict, metrics: dict) -> None
  - end_run(run_id) -> None
- Tests:
  - CSV/MLflow entries exist; run_id uniqueness.

Module: utils.py
- Purpose: Calendars and helpers.
- Functions:
  - month_ends_from_daily(dates) -> pd.DatetimeIndex
  - vn_trading_calendar_holidays() -> set[dates]
  - enforce_types(df) -> df
- Tests:
  - Calendar correctness on fixture periods.

4. Workflow, module and milestone dependencies
- Data path: data_io -> filters -> returns -> momentum -> portfolio -> costs -> backtest -> metrics/stats -> report.
- Dependencies:
  - M1 â†’ M2 (eligibility uses loaded data).
  - M2 â†’ M3 (returns computed for eligible tickers per period).
  - M3 â†’ M4 (momentum scores depend on returns and skip).
  - M4 â†’ M5 (deciles to cohorts to weights).
  - M5 â†’ M6 (trades determine costs).
  - M6 â†’ M7 (net returns require costs).
  - M7 â†’ M8/M9 (metrics, tests).
  - M8/M9 â†’ M10 (reporting).
- Build sequencing: Write failing tests for each module; implement minimal pass; refactor; re-run higher-level tests to ensure no regression.
- Validation gates:
  - Gate A (M3): Monthly vs product of dailies within 1e-8.
  - Gate B (M5): Overlap weights sum ≤ 1 and match cohort math to 1e-12.
  - Gate C (M7): Toy backtest matches hand-calculated returns ±1bp.
  - Gate D (M8): Achieved — NW alpha on synthetic series within tolerance.

5. Vietnam-specific considerations (embedded in modules)
- Long-only: No short positions; negative weights disallowed (portfolio.py tests).
- Price limits ±7%: Approximate by capping daily returns and skipping trades on limit days (backtest.py).
- T+2 settlement: Execute rebalances at t+1, consider availability constraints; skip if halted.
- Liquidity: ADV filters and participation caps (e.g., ≤10% ADV) in costs.py; capacity analysis function optional.
- Data: Use only price/volume; no fundamentals; survivorship avoidance via monthly universe recompute using only past info.

6. TDD examples (per function: inputs, outputs, logic, tests)
Example: returns.cum_return_skip
- Inputs: ret_d DataFrame, J=6, skip_days=5; monthly formation dates.
- Outputs: score_m DataFrame aligned to formation months.
- Logic: For each ticker and formation month-end t, compute product (1+ret_d) from window [t-6m-5d, t-5d) − 1.
- Focused tests:
  - Construct daily returns = 1% for 6 months, last week random â†’ score ≈ (1.01^N)−1 excluding last 5 days.
  - Off-by-one: increasing-window checks against manual computation.

Example: portfolio.target_weights_from_cohorts
- Inputs: cohorts dict, K, constraints.
- Outputs: weights DataFrame [month_end×ticker].
- Logic: Equal-weight within cohort, each cohort 1/K; renormalize if ineligible/halts.
- Focused tests:
  - With K=3 and 3 cohorts of 10 names each, each name weight=1/K * 1/10; sums ≤ 1.
  - Dropping a name renormalizes remaining names without exceeding sum ≤ 1.

7. Backtesting configuration (defaults; configurable)
- backtest.yml defaults:
  - initial_capital: 1_000_000_000 VND
  - rebalance: monthly
  - J_list: [3,6,9,12]; K_list: [1,3,6,12]; skip_days: 5
  - transaction_cost_bps: 25 per side; impact_bps: 10 beyond 10% ADV
  - slippage: linear with slope 2 bps per 1% participation; cap at 50 bps
  - execution_price: next day open (proxy)
  - price_limit_model: cap daily ret at ±7%; skip trades on limit
- Validation:
  - Costless runs match gross; increasing cost penalizes high-turnover combos; execution lag shifts returns as expected.

8. Evaluation framework and statistical tests
- Performance:
  - CAGR, annualized vol, Sharpe, IR vs VN-Index, max DD, DD duration, downside dev, VaR(95/99), beta.
- Statistical inference:
  - Alpha via OLS of strategy excess on market excess; NW lags=6 (monthly).
  - Bootstrap block resampling for Sharpe/alpha CIs; subperiod t-stats (2010–2014, 2015–2019, 2020–2025; COVID crash window).
- Validation:
  - Compare with benchmark: D10 long-only vs VN-Index; equal-weight universe baseline.

9. Robustness and parameter selection
- Grid search J×K; evaluate OOS using walk-forward (36m train / 12m valid).
- Robustness:
  - Costs sensitivity: 25–75 bps; participation caps 5–15% ADV.
  - Alternative constructions: value-weighted within D10; volatility-scaled weights (optional).
  - Signal variants (optional, post-baseline): 52-week high proximity; vol-scaled momentum.
- Validation:
  - Stability of preferred (J,K) across folds; consistent rank of top combos OOS.

10. Reporting and documentation
- Outputs:
  - Strategy overview with logic flow diagram (text/plot), assumptions, data lineage.
  - Tables: J×K heatmap (Sharpe, IR), monthly return tables, subperiod stats, alpha regression summary with NW t-stats.
  - Plots: equity curve vs VN-Index, drawdown, turnover, capacity scenarios.
- Reproducibility:
  - Log commit hash, config hashes, environment; export run artifacts.

11. Error handling, assumptions, and escalation
- Assumptions:
  - Prices are pre-adjusted (splits/dividends) as per data note.
  - No shorting; borrow costs irrelevant.
  - Execution at next-day open approximates achievable fill; price limits approximated via return caps.
- Unanswered questions (to be confirmed; defaults applied):
  - Exact fee schedule per broker/exchange (default 25 bps/side).
  - Granular halt/auction calendars; currently approximated by missing/zero volume days.
  - Shares outstanding for value-weighting; omitted in baseline unless provided.
- Escalation paths:
  - Data anomalies (massive spikes, persistent NaNs): open data-quality ticket; quarantine symbols; rerun with exclusion and document impact.
  - Execution feasibility (thin liquidity): tighten ADV thresholds, reduce participation caps; run capacity analysis and report sensitivity.
  - Statistical instability: expand sample, adjust NW lags, or re-express in subperiods; document limits.

12. Phase plan and clean build sequencing (mapped to checklist)
- Phase 1 (Weeks 1–2): M0–M3
  - Deliverables: Clean OHLCV, eligibility universe, validated returns; tests all green.
  - Validation: Gate A achieved.
- Phase 2 (Weeks 3–4): M4–M6
  - Deliverables: Momentum scores, deciles, overlapping portfolio, costs module.
  - Validation: Gate B; costs monotonicity tests pass.
- Phase 3 (Weeks 5-6): M7-M8
  - Deliverables: Backtester with full metrics and NW alpha tests.
  - Validation: Gate C and D achieved.
- Phase 4 (Weeks 7–8): M9
  - Deliverables: Grid and walk-forward analyses; robustness pack.
  - Validation: Stable (J,K) identified; sensitivity plots.
- Phase 5 (Weeks 9–10): M10
  - Deliverables: Research report; experiment logs; reproducibility checks.

Appendix A: Mapping to Jegadeesh & Titman (1993) methodology
- Formation: J-month cumulative return excluding most recent week (skip 5 trading days).
- Holding: K months with overlapping portfolios; monthly rebalancing.
- Ranks: Cross-sectional deciles; we implement long-only top decile (winners) tailored to Vietnam (no shorting).
- Data: Price/return only; no fundamentals; aligns with core sections of J&T.
- Testing: Report mean returns, t-stats with NW SEs; decompose performance vs market.

Appendix B: Sample configs (abbrev.)
- config/strategy.yml
  J_list: [3,6,9,12]
  K_list: [1,3,6,12]
  skip_days: 5
  deciles: 10
  weighting: equal
- config/backtest.yml
  initial_capital: 1000000000
  transaction_cost_bps: 25
  slippage:
    model: linear
    bps_per_1pct_participation: 2
    cap_bps: 50
  impact_bps_large_trade: 10
  adv_participation_cap: 0.10
  execution_price: next_open
  price_limit_cap: 0.07
- Validation:
  - Loading these configs yields reproducible backtest runs; changing costs shifts IR monotonically downward, confirming intended effect.
