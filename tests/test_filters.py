from __future__ import annotations
import pandas as pd
import numpy as np


def _mk_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])  # ensure datetime type
    return df.set_index(["date", "ticker"]).sort_index()


def test_compute_turnover_and_rolling_adv_basic():
    from src.filters import compute_turnover, rolling_adv

    rows = [
        {"date": "2020-01-01", "ticker": "AAA", "close": 10.0, "volume": 100},
        {"date": "2020-01-02", "ticker": "AAA", "close": 11.0, "volume": 0},
        {"date": "2020-01-03", "ticker": "AAA", "close": 12.0, "volume": 200},
    ]
    df = _mk_df(rows)

    trn = compute_turnover(df)
    assert trn.loc[(pd.Timestamp("2020-01-01"), "AAA")] == 10.0 * 100
    assert trn.loc[(pd.Timestamp("2020-01-02"), "AAA")] == 11.0 * 0

    adv = rolling_adv(df, window=2, min_periods=1)
    # Day 1: avg over [1000] = 1000; Day 2: avg over [0,1000] = 500; Day 3: avg over [0, 2400] = 1200
    got = [
        adv.loc[(pd.Timestamp("2020-01-01"), "AAA")],
        adv.loc[(pd.Timestamp("2020-01-02"), "AAA")],
        adv.loc[(pd.Timestamp("2020-01-03"), "AAA")],
    ]
    assert np.allclose(got, [1000.0, 500.0, 1200.0])


def test_eligibility_flags_components_and_no_lookahead():
    from src.filters import eligibility_flags

    # Two tickers across 6 days; use small windows to keep it tight
    rows = [
        # AAA trades 4 of last 5 grid days, price above threshold at end
        {"date": "2020-01-01", "ticker": "AAA", "open": 10, "high": 10.5, "low": 9.5, "close": 10, "volume": 100},
        {"date": "2020-01-02", "ticker": "AAA", "open": 10, "high": 10.5, "low": 9.5, "close": 10, "volume": 0},  # zero volume (non-trading)
        {"date": "2020-01-04", "ticker": "AAA", "open": 12, "high": 12.5, "low": 11.5, "close": 12, "volume": 50},
        {"date": "2020-01-05", "ticker": "AAA", "open": 12, "high": 12.5, "low": 11.5, "close": 12, "volume": 50},
        # BBB is illiquid and below price threshold at end
        {"date": "2020-01-01", "ticker": "BBB", "open": 5, "high": 5.2, "low": 4.8, "close": 5, "volume": 10},
        {"date": "2020-01-03", "ticker": "BBB", "open": 5, "high": 5.2, "low": 4.8, "close": 5, "volume": 0},
        {"date": "2020-01-05", "ticker": "BBB", "open": 5, "high": 5.2, "low": 4.8, "close": 5, "volume": 0},
    ]
    df = _mk_df(rows)

    flags = eligibility_flags(
        df,
        lookback_days=5,
        min_history_days=3,
        min_price_vnd=10.0,
        min_adv_vnd=400.0,  # AAA meets with given data; BBB fails
        max_nontrading_days=1,
        calendar="union",
        indices_df=None,
        anomalies_df=None,
    )

    # Evaluate at last grid day 2020-01-05
    end = pd.Timestamp("2020-01-05")
    aa = flags.loc[(end, "AAA")]
    bb = flags.loc[(end, "BBB")]

    assert bool(aa["price_ok"]) is True
    assert bool(aa["adv_ok"]) is True
    # In last 5 days (1,2,3,4,5), AAA has one zero-volume day and one missing day (3rd) counted -> nontrading_days <=1? there is 1 missing (3) and 1 zero (2) => 2 > 1 -> fails
    assert int(aa["nontrading_days"]) >= 2
    assert bool(aa["nontrading_ok"]) is False
    # AAA history days in window >= 3 -> min_history_ok True
    assert bool(aa["min_history_ok"]) is True
    assert bool(aa["quality_ok"]) is True
    assert bool(aa["eligible"]) is False  # nontrading fails

    assert bool(bb["price_ok"]) is False  # price below 10
    assert bool(bb["adv_ok"]) is False
    assert bool(bb["eligible"]) is False

    # No look-ahead: altering a future day beyond end-1 must not affect eligibility at end-1
    # Compute flags up to 2020-01-04, then change data on 2020-01-05 and ensure eligibility at 2020-01-04 is unchanged
    df2 = df.copy()
    f_all = eligibility_flags(df2, lookback_days=5, min_history_days=3, min_price_vnd=10.0, min_adv_vnd=600.0, max_nontrading_days=1)
    day4 = pd.Timestamp("2020-01-04")
    aa_before = f_all.loc[(day4, "AAA"), "eligible"]
    # Modify future (2020-01-05) massively
    df2.loc[(pd.Timestamp("2020-01-05"), "AAA"), "close"] = 1000.0
    df2.loc[(pd.Timestamp("2020-01-05"), "AAA"), "volume"] = 1_000_000
    f_mod = eligibility_flags(df2, lookback_days=5, min_history_days=3, min_price_vnd=10.0, min_adv_vnd=600.0, max_nontrading_days=1)
    aa_after = f_mod.loc[(day4, "AAA"), "eligible"]
    assert bool(aa_before) == bool(aa_after)


def test_quality_screen_uses_anomalies_df():
    from src.filters import eligibility_flags

    rows = [
        {"date": "2020-02-01", "ticker": "AAA", "open": 10, "high": 10.5, "low": 9.5, "close": 10, "volume": 100},
        {"date": "2020-02-02", "ticker": "AAA", "open": 10, "high": 10.5, "low": 9.5, "close": 10, "volume": 100},
        {"date": "2020-02-03", "ticker": "AAA", "open": 10, "high": 10.5, "low": 9.5, "close": 10, "volume": 100},
    ]
    df = _mk_df(rows)
    # An anomaly inside the window should flip quality_ok to False
    anoms = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-02-02"]),
            "ticker": ["AAA"],
            "rule": ["ohlc_ordering_violation"],
        }
    )

    flags = eligibility_flags(
        df,
        lookback_days=3,
        min_history_days=3,
        min_price_vnd=1.0,
        min_adv_vnd=1.0,
        max_nontrading_days=3,
        anomalies_df=anoms,
    )

    end = pd.Timestamp("2020-02-03")
    rec = flags.loc[(end, "AAA")]
    # At end date, trailing window includes 02-01..02-03; anomaly on 02-02 exists => quality_ok False
    assert bool(rec["quality_ok"]) is False
    # Also ensure 2020-02-01 has quality_ok True (no prior anomaly)
    rec_prior = flags.loc[(pd.Timestamp("2020-02-01"), "AAA")]
    assert bool(rec_prior["quality_ok"]) is True
    # And 2020-02-02 (anomaly day) has quality_ok False
    rec_mid = flags.loc[(pd.Timestamp("2020-02-02"), "AAA")]
    assert bool(rec_mid["quality_ok"]) is False


def test_monthly_universe_selects_last_grid_day():
    from src.filters import eligibility_flags, monthly_universe

    rows = [
        {"date": "2020-03-30", "ticker": "AAA", "open": 10, "high": 10.5, "low": 9.5, "close": 10, "volume": 100},
        {"date": "2020-03-31", "ticker": "AAA", "open": 10, "high": 10.5, "low": 9.5, "close": 10, "volume": 100},
        {"date": "2020-04-01", "ticker": "AAA", "open": 10, "high": 10.5, "low": 9.5, "close": 10, "volume": 100},
        {"date": "2020-04-30", "ticker": "AAA", "open": 10, "high": 10.5, "low": 9.5, "close": 10, "volume": 100},
    ]
    df = _mk_df(rows)
    flags = eligibility_flags(
        df, lookback_days=2, min_history_days=1, min_price_vnd=1.0, min_adv_vnd=1.0, max_nontrading_days=10
    )
    uni = monthly_universe(flags)
    # Expect 2020-03-31 and 2020-04-30 as month ends
    assert set(pd.to_datetime(uni["month_end"])) == {pd.Timestamp("2020-03-31"), pd.Timestamp("2020-04-30")}
    # Eligible should mirror flags at those dates
    for d in ["2020-03-31", "2020-04-30"]:
        f_ok = bool(flags.loc[(pd.Timestamp(d), "AAA"), "eligible"])
        u_ok = bool(uni.loc[(uni["month_end"] == pd.Timestamp(d)) & (uni["ticker"] == "AAA"), "eligible"].iloc[0])
        assert f_ok == u_ok
