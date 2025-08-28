from __future__ import annotations
import numpy as np
import pandas as pd


def _mk_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])  # ensure datetime type
    return df.set_index(["date", "ticker"]).sort_index()


def test_daily_simple_and_log_returns_per_ticker():
    from src.returns import daily_simple_returns, daily_log_returns

    rows = [
        {"date": "2020-01-01", "ticker": "AAA", "close": 10},
        {"date": "2020-01-02", "ticker": "AAA", "close": 11},
        {"date": "2020-01-01", "ticker": "BBB", "close": 20},
        {"date": "2020-01-03", "ticker": "BBB", "close": 22},
    ]
    df = _mk_df(rows)
    r = daily_simple_returns(df)
    # First per ticker is NaN
    assert np.isnan(r.loc[(pd.Timestamp("2020-01-01"), "AAA")])
    assert np.isnan(r.loc[(pd.Timestamp("2020-01-01"), "BBB")])
    # AAA: 11/10 - 1 = 0.1
    assert abs(r.loc[(pd.Timestamp("2020-01-02"), "AAA")] - 0.1) < 1e-9
    # BBB next observed day is 2020-01-03 with previous 2020-01-01
    assert abs(r.loc[(pd.Timestamp("2020-01-03"), "BBB")] - (22/20 - 1.0)) < 1e-9

    lr = daily_log_returns(df)
    # AAA day 2 log return
    assert abs(lr.loc[(pd.Timestamp("2020-01-02"), "AAA")] - np.log(11/10)) < 1e-12


def test_forward_returns_multiple_horizons():
    from src.returns import forward_returns

    rows = [
        {"date": "2020-01-01", "ticker": "AAA", "close": 10},
        {"date": "2020-01-02", "ticker": "AAA", "close": 11},
        {"date": "2020-01-03", "ticker": "AAA", "close": 11},
        {"date": "2020-01-04", "ticker": "AAA", "close": 12},
    ]
    df = _mk_df(rows)
    out = forward_returns(df, horizons=[1, 2])
    # h=1 at 2020-01-02: 11/11 - 1 = 0
    assert abs(out.loc[(pd.Timestamp("2020-01-02"), "AAA"), "fwd_ret_1d"] - 0.0) < 1e-12
    # h=2 at 2020-01-01: 11/10 - 1 = 0.1
    assert abs(out.loc[(pd.Timestamp("2020-01-01"), "AAA"), "fwd_ret_2d"] - 0.1) < 1e-12


def test_monthly_returns_close_to_close_and_eligibility_filter():
    from src.returns import monthly_returns, eligible_monthly_returns

    # Two months, sparse observations
    rows = [
        {"date": "2020-01-01", "ticker": "AAA", "close": 10},
        {"date": "2020-01-31", "ticker": "AAA", "close": 12},  # Jan end
        {"date": "2020-02-28", "ticker": "AAA", "close": 15},  # Feb end
        {"date": "2020-01-15", "ticker": "BBB", "close": 20},
        {"date": "2020-02-27", "ticker": "BBB", "close": 22},
    ]
    df = _mk_df(rows)
    mret = monthly_returns(df, calendar="union")
    # Rows for both tickers at month-ends (Jan, Feb)
    jan = pd.Timestamp("2020-01-31")
    feb = pd.Timestamp("2020-02-28")
    # AAA first month NaN, second (15/12 - 1)
    r_aaa_jan = mret.loc[(mret["month_end"] == jan) & (mret["ticker"] == "AAA"), "ret_1m"].iloc[0]
    assert pd.isna(r_aaa_jan)
    r_aaa_feb = mret.loc[(mret["month_end"] == feb) & (mret["ticker"] == "AAA"), "ret_1m"].iloc[0]
    assert abs(r_aaa_feb - (15/12 - 1.0)) < 1e-12

    # BBB: at Jan-end last close on/before is 20 (Jan 15), next month-end last close is 22 (Feb 27)
    r_bbb_jan = mret.loc[(mret["month_end"] == jan) & (mret["ticker"] == "BBB"), "ret_1m"].iloc[0]
    assert pd.isna(r_bbb_jan)
    r_bbb_feb = mret.loc[(mret["month_end"] == feb) & (mret["ticker"] == "BBB"), "ret_1m"].iloc[0]
    assert abs(r_bbb_feb - (22/20 - 1.0)) < 1e-12

    # Eligibility filter: only keep AAA in Feb
    monthly_flags = pd.DataFrame({
        "month_end": [jan, feb, jan, feb],
        "ticker": ["AAA", "AAA", "BBB", "BBB"],
        "eligible": [False, True, False, False],
    })
    filt = eligible_monthly_returns(df, monthly_flags=monthly_flags, calendar="union")
    assert len(filt) == 1
    assert filt.iloc[0]["month_end"] == feb and filt.iloc[0]["ticker"] == "AAA"


def test_monthly_returns_with_vnindex_calendar_alignment():
    from src.returns import monthly_returns

    rows = [
        {"date": "2020-01-30", "ticker": "AAA", "close": 10},
        {"date": "2020-02-27", "ticker": "AAA", "close": 11},
    ]
    df = _mk_df(rows)
    # VNINDEX month-ends
    indices = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-31", "2020-02-28"]),
        "index": ["VNINDEX", "VNINDEX"],
        "close": [900.0, 905.0],
    })
    mret = monthly_returns(df, calendar="vnindex", indices_df=indices)
    # Month-ends must follow indices dates and use last close on/before each
    assert set(pd.to_datetime(mret["month_end"]).unique()) == {pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-28")}
    # First month NaN, second is 11/10 - 1
    jan_row = mret.loc[(mret["month_end"] == pd.Timestamp("2020-01-31")) & (mret["ticker"] == "AAA")]
    feb_row = mret.loc[(mret["month_end"] == pd.Timestamp("2020-02-28")) & (mret["ticker"] == "AAA")]
    assert pd.isna(jan_row["ret_1m"].iloc[0])
    assert abs(feb_row["ret_1m"].iloc[0] - (11/10 - 1.0)) < 1e-12
