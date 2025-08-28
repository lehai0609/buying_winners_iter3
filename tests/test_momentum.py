from __future__ import annotations
import numpy as np
import pandas as pd


def _mk_daily(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])  # ensure datetime type
    return df.set_index(["date", "ticker"]).sort_index()


def test_momentum_windowing_12_1():
    from src.momentum import compute_momentum

    # Build monthly returns directly for 15 consecutive months
    months = pd.date_range("2020-01-31", periods=15, freq="ME")
    # define simple pattern of returns, avoid -1
    r = np.linspace(0.01, 0.03, num=15)
    ret = pd.DataFrame({
        "month_end": np.tile(months, 1),
        "ticker": ["AAA"] * len(months),
        "ret_1m": r,
    })

    out = compute_momentum(ret, lookback=12, gap=1)
    # formation month is the last month in out; window excludes the most recent 1 month
    last_month = months[-1]
    row = out[(out["month_end"] == last_month) & (out["ticker"] == "AAA")].iloc[0]
    # Window should be months[-2] back 12 entries: months[2:14]
    expected = np.prod(1.0 + r[2:14]) - 1.0
    assert abs(float(row["momentum"]) - float(expected)) < 1e-12
    assert int(row["n_months_used"]) >= 12
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
    from src.momentum import compute_momentum

    months = pd.date_range("2020-01-31", periods=5, freq="ME")
    ret = pd.DataFrame({
        "month_end": months,
        "ticker": ["AAA"] * len(months),
        "ret_1m": [0.01] * len(months),
    })
    out = compute_momentum(ret, lookback=6, gap=1)
    # No row can be valid due to insufficient lookback history
    assert not out["valid"].any()
    # momentum should be NaN for rows without full window
    assert out["momentum"].isna().any()


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
            "gap_months": 1,
            "n_deciles": 5,
            "min_names_per_month": 1,
            "exclude_hard_errors": True,
            "calendar": "union",
            "price_col": "close",
        }},
        "out": {
            "ohlcv_parquet": str(tmp_path / "ohlcv.parquet"),
            "hard_errors_csv": str(tmp_path / "hard_errors.csv"),
        },
    }
    # Persist an input parquet to mimic production path
    df.to_parquet(tmp_path / "ohlcv.parquet")

    out = compute_momentum_signals(df=None, cfg_dict=cfg, clean_parquet_path=None, write=False)

    # Columns present
    for c in ["month_end", "ticker", "momentum", "n_months_used", "valid"]:
        assert c in out.columns
    # No duplicate (month_end, ticker)
    assert not out.duplicated(["month_end", "ticker"]).any()
    # For final month, names should be valid with enough history
    last_month = pd.to_datetime(out["month_end"]).max()
    sub = out[out["month_end"] == last_month]
    assert (sub["n_months_used"] >= 12).all()
    # Deciles assigned since min_names_per_month=1
    assert sub["decile"].notna().all()
