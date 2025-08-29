Short answer: **Mostly aligned.** Four fixes needed.

**Where M5 matches TRD + project statement**

* Long-only winners, equal-weight within cohort, K-month overlaps, monthly rebalance.
* Weights sum ≤1 with implicit cash when cohorts are fewer/empty; turnover reported.
* Inputs from M4 signals and same monthly calendar; dependency chain M4→M5→M6 respected.
* Config lives in `config/data.yml` now, with option to mirror into `strategy.yml` later.

**Deviations to correct**

1. **Trade timing (T+2) missing.**
   TRD M5 expects trade generation with a T+2 assumption; current M5 only computes weight deltas and declares execution out of scope. Add a `generate_trades(..., t_plus=1, settlement="T+2")` in M5 or amend TRD to move T+2 fully to M7.

**Action items**

* Add T+2 trade generator or revise TRD to push timing to M7.
* Expose `max_weight_per_name` and optional `turnover_cap` in config and tests.

If you want, I can rewrite M5 to match TRD strictly (mask-in API + T+2 trades) or patch TRD to match the current M5 module plan.
