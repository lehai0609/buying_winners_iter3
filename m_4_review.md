Short answer: Mostly aligned. Key gaps vs TRD and the project statement: skip-period definition, API shape, config completeness, and explicit universe use.

### What matches

* Goal and scope: J-month momentum, cross-sectional deciles, long-only top cohort; outputs and CLI paths consistent. &#x20;
* Deterministic ranking and handling of duplicates; optional exclusion using `hard_errors.csv`. &#x20;

### Deviations to fix

1. **Skip period**

* TRD and project statement require a 5-trading-day skip; your spec encodes a **monthly gap `G`** (12-1 by month), not a 5-day daily skip. Replace the monthly gap with daily skip logic from M3 when forming signals. &#x20;

2. **API shape for `momentum.py`**

* TRD’s detailed module spec shows a daily-returns API: `momentum_scores(ret_d, universe_mask, J, skip_days)`; your spec defines a monthly-pipeline API (`compute_monthly_prices/returns`, then `compute_momentum`). Choose one and make TRD and code consistent. If you keep the monthly orchestrator, still compute the **score from daily returns with 5-day skip**. &#x20;

3. **Config completeness**

* TRD adds `signals.momentum` plus optional `calendar` and `price_col`. Your spec omits the latter two. Add them. &#x20;

4. **Explicit universe mask**

* TRD momentum functions accept a **universe/ineligibility mask** from M2; your spec doesn’t state using it. Compute scores only for eligible names at each formation month.&#x20;

5. **Decile fallback detail**

* TRD notes a **rank-based fallback** when unique values < bins. Your spec reduces bins and records `n_deciles_used`. Add the rank-based fallback for full alignment. &#x20;

### Tests to add or adjust

* Windowing on **daily grid with `skip_days=5`**; off-by-one checks.
* Respect of **universe mask** at formation.
* Config parse for `calendar` and `price_col`.&#x20;

### Verdict

* Keep: decile mechanics, outputs, CLI, hard-error exclusion.
* Change: compute momentum from **daily returns with 5-day skip**, align `momentum.py` API with TRD, include `calendar`/`price_col`, and enforce the M2 universe. After these edits, M4 will strictly conform to TRD and the project statement. &#x20;
