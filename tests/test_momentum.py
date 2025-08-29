from __future__ import annotations
import numpy as np
import pandas as pd


def _mk_daily(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])  # ensure datetime type
    return df.set_index(["date", "ticker"]).sort_index()


def test_momentum_windowing_12m_with_5d_skip():
    from src.momentum import momentum_scores

    # Daily constant return over ~15 months
    dates = pd.date_range("2020-01-01", periods=330, freq="B")
    r_d = 0.001  # 0.1% daily
    rows = [{"date": d, "ticker": "AAA", "ret_1d": r_d} for d in dates]
    ret = pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()

    out = momentum_scores(ret_d=ret, universe_mask=None, J=12, skip_days=5, calendar="union", indices_df=None)
    # Last formation month
    if len(out) == 0:
        assert False, "expected non-empty output for sufficient history"
    last_month = pd.to_datetime(out["month_end"]).max()
    row = out[(out["month_end"] == last_month) & (out["ticker"] == "AAA")].iloc[0]
    # Expected: product over d with dates in (s_anchor, e_excl), where s_anchor is month-end 12 months prior
    # Since ret is constant, the momentum equals (1+r)^N - 1, for N selected days.
    # Count days used by re-deriving the mask
    from src.calendar import build_trading_grid, month_ends, shift_trading_days
    dummy = pd.DataFrame(index=ret.index)
    grid = build_trading_grid(dummy, calendar="union", indices_df=None)
    me = month_ends(grid)
    s_per = last_month.to_period("M") - 12
    s_anchor = pd.to_datetime(pd.Series(me, index=me.to_period("M")).loc[s_per])
    e_excl = shift_trading_days(grid, pd.DatetimeIndex([last_month]), -5)[0]
    mask = (ret.index.get_level_values("date") > s_anchor) & (ret.index.get_level_values("date") < e_excl)
    N = int(mask.sum())
    expected = (1.0 + r_d) ** N - 1.0
    assert abs(float(row["momentum"]) - float(expected)) < 1e-12
    assert bool(row["valid"]) is True


def test_assign_deciles_basic_and_sparse_threshold():
    from src.momentum import assign_deciles

    month = pd.Timestamp("2021-01-31")
    # Ten names with distinct momentum
    sig = pd.DataFrame({
        "month_end": [month] * 10,
        "ticker": [f"T{i}" for i in range(10)],
        "momentum": np.arange(10, dtype=float),
        "n_months_used": [12] * 10,
        "valid": [True] * 10,
    })
    out = assign_deciles(sig.copy(), n_deciles=10, min_names_per_month=1)
    dec = set(int(x) for x in out["decile"].dropna().unique())
    assert dec == set(range(1, 11))
    # Sparse case: threshold higher than available names -> NaNs
    out2 = assign_deciles(sig.copy(), n_deciles=10, min_names_per_month=50)
    assert out2["decile"].isna().all() and out2["pct_rank"].isna().all()


def test_insufficient_history_marks_invalid():
    from src.momentum import momentum_scores

    # Only ~60 business days (~3 months), but J=6 -> invalid
    dates = pd.date_range("2020-01-01", periods=60, freq="B")
    ret = pd.DataFrame({
        "date": np.repeat(dates, 1),
        "ticker": ["AAA"] * len(dates),
        "ret_1d": [0.001] * len(dates),
    }).set_index(["date", "ticker"]).sort_index()
    out = momentum_scores(ret_d=ret, universe_mask=None, J=6, skip_days=5, calendar="union", indices_df=None)
    # With insufficient calendar history, output may be empty or all invalid
    if len(out) == 0:
        assert True
    else:
        assert not out["valid"].any()


def test_pct_rank_deterministic():
    from src.momentum import assign_deciles

    month = pd.Timestamp("2021-06-30")
    vals = [0.0, 0.5, 1.0]
    df = pd.DataFrame({
        "month_end": [month] * 3,
        "ticker": ["A", "B", "C"],
        "momentum": vals,
        "n_months_used": [12, 12, 12],
        "valid": [True, True, True],
    })
    out = assign_deciles(df, n_deciles=3, min_names_per_month=3)
    # Expect pct ranks 0, 0.5, 1.0 in sorted order of momentum
    o = out.sort_values("momentum").reset_index(drop=True)
    assert abs(float(o.loc[0, "pct_rank"]) - 0.0) < 1e-12
    assert abs(float(o.loc[1, "pct_rank"]) - 0.5) < 1e-12
    assert abs(float(o.loc[2, "pct_rank"]) - 1.0) < 1e-12


def test_end_to_end_compute_momentum_signals_small(tmp_path):
    from src.momentum import compute_momentum_signals

    # Construct small daily OHLCV for 2 tickers over 18 months
    dates = pd.date_range("2020-01-01", periods=550, freq="B")  # ~2 years business days
    # pick last business day of each month
    month_ends = pd.date_range(dates.min(), dates.max(), freq="ME")
    px_a = []
    px_b = []
    pa = 10.0
    pb = 20.0
    for d in dates:
        # monotonic upward drift
        pa *= 1.0005
        pb *= 1.0008
        px_a.append({"date": d, "ticker": "AAA", "open": pa, "high": pa, "low": pa, "close": pa, "volume": 100})
        px_b.append({"date": d, "ticker": "BBB", "open": pb, "high": pb, "low": pb, "close": pb, "volume": 200})
    df = _mk_daily(px_a + px_b)

    cfg = {
        "signals": {"momentum": {
            "lookback_months": 12,
            "skip_days": 5,
            "n_deciles": 5,
            "min_names_per_month": 1,
            "exclude_hard_errors": True,
            "calendar": "union",
            "price_col": "close",
        }},
        "out": {
            "ohlcv_parquet": str(tmp_path / "ohlcv.parquet"),
            "monthly_universe_parquet": str(tmp_path / "monthly_universe.parquet"),
            "hard_errors_csv": str(tmp_path / "hard_errors.csv"),
        },
    }
    # Persist an input parquet to mimic production path
    df.to_parquet(tmp_path / "ohlcv.parquet")

    # Build a simple monthly universe on the trading grid month-ends
    from src.calendar import build_trading_grid, month_ends
    grid = build_trading_grid(df, calendar="union")
    all_me = month_ends(grid)
    uni = pd.DataFrame({
        "month_end": list(all_me) * 2,
        "ticker": ["AAA"] * len(all_me) + ["BBB"] * len(all_me),
        "eligible": [True] * (2 * len(all_me)),
    })
    uni.to_parquet(tmp_path / "monthly_universe.parquet", index=False)

    out = compute_momentum_signals(df=None, cfg_dict=cfg, clean_parquet_path=None, write=False)

    # Columns present
    for c in ["month_end", "ticker", "momentum", "valid"]:
        assert c in out.columns
    # No duplicate (month_end, ticker)
    assert not out.duplicated(["month_end", "ticker"]).any()
    # For final month, names should be valid with enough history
    last_month = pd.to_datetime(out["month_end"]).max()
    sub = out[out["month_end"] == last_month]
    # Deciles assigned since min_names_per_month=1 and signals valid
    assert sub["valid"].all()
    assert sub["decile"].notna().all()


def test_universe_mask_enforced_drops_ineligible_rows():
    from src.momentum import momentum_scores

    # Two tickers, ~ 9 months of daily data
    dates = pd.date_range("2020-01-01", periods=190, freq="B")
    r_d_a = 0.0008
    r_d_b = 0.0010
    rows = []
    for d in dates:
        rows.append({"date": d, "ticker": "AAA", "ret_1d": r_d_a})
        rows.append({"date": d, "ticker": "BBB", "ret_1d": r_d_b})
    ret = pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()

    # Build a monthly universe where BBB is ineligible at the last month
    from src.calendar import build_trading_grid, month_ends
    dummy = pd.DataFrame(index=ret.index)
    grid = build_trading_grid(dummy, calendar="union", indices_df=None)
    me = month_ends(grid)
    last_me = me.max()
    uni_rows = []
    for m in me:
        uni_rows.append({"month_end": m, "ticker": "AAA", "eligible": True})
        uni_rows.append({"month_end": m, "ticker": "BBB", "eligible": bool(m != last_me)})
    uni = pd.DataFrame(uni_rows)

    out = momentum_scores(ret_d=ret, universe_mask=uni, J=6, skip_days=5, calendar="union", indices_df=None)
    # Ensure that for last formation month, no row exists for BBB
    if len(out) == 0:
        assert False, "unexpected empty output"
    sub_last = out[out["month_end"] == last_me]
    assert set(sub_last["ticker"]) == {"AAA"}
