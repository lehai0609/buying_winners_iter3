from __future__ import annotations
import pandas as pd


def _mk_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])  # ensure datetime type
    return df.set_index(["date", "ticker"]).sort_index()


def test_build_trading_grid_union_and_vnindex():
    from src.calendar import build_trading_grid

    rows = [
        {"date": "2020-01-01", "ticker": "AAA", "close": 10, "open": 10, "high": 10.2, "low": 9.8, "volume": 1},
        {"date": "2020-01-03", "ticker": "AAA", "close": 10, "open": 10, "high": 10.2, "low": 9.8, "volume": 1},
        {"date": "2020-02-28", "ticker": "BBB", "close": 9,  "open": 9,  "high": 9.1,  "low": 8.9, "volume": 2},
    ]
    df = _mk_df(rows)
    grid_union = build_trading_grid(df, calendar="union")
    assert list(grid_union) == list(sorted(df.index.get_level_values("date").unique()))

    # vnindex grid
    indices = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-01-31", "2020-02-28"]),
        "index": ["VNINDEX", "VNINDEX", "VNINDEX"],
        "close": [900.0, 905.0, 910.0],
    })
    grid_idx = build_trading_grid(df, calendar="vnindex", indices_df=indices)
    assert list(grid_idx) == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-28")]


def test_month_ends_and_nth_trading_day_and_shift_and_align():
    from src.calendar import month_ends, nth_trading_day_in_month, shift_trading_days, align_to_grid

    grid = pd.DatetimeIndex(pd.to_datetime([
        "2020-01-01", "2020-01-03", "2020-01-31",
        "2020-02-03", "2020-02-28",
    ]))
    me = month_ends(grid)
    assert list(me) == [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-28")]

    nth2 = nth_trading_day_in_month(grid, n=2)
    assert nth2.loc[pd.Period("2020-01")] == pd.Timestamp("2020-01-03")
    assert nth2.loc[pd.Period("2020-02")] == pd.Timestamp("2020-02-28")

    # shift
    shifted = shift_trading_days(grid, pd.Series([pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-01")]), +1)
    assert list(shifted) == [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-01-03")]
    # clipping at bounds
    shifted2 = shift_trading_days(grid, pd.Series([pd.Timestamp("2020-01-01")]), -10)
    assert list(shifted2) == [pd.Timestamp("2020-01-01")]
    # date not in grid -> NaT
    not_in = shift_trading_days(grid, pd.Series([pd.Timestamp("2020-01-02")]), 1)
    assert pd.isna(not_in.iloc[0])

    # align_to_grid
    rows = [
        {"date": "2020-01-01", "ticker": "AAA", "open": 10, "high": 10.1, "low": 9.9, "close": 10, "volume": 1},
        {"date": "2020-01-31", "ticker": "AAA", "open": 11, "high": 11.2, "low": 10.8, "close": 11, "volume": 2},
        {"date": "2020-02-28", "ticker": "BBB", "open": 9,  "high": 9.1,  "low": 8.9,  "close": 9,  "volume": 3},
    ]
    df = _mk_df(rows)
    left = align_to_grid(df, grid=grid, how="left")
    # left should contain len(grid) * 2 tickers rows
    assert len(left) == len(grid) * 2
    inner = align_to_grid(df, grid=grid, how="inner")
    # inner keeps only dates present in grid and input
    dates_in = set(df.index.get_level_values("date").unique()) & set(grid)
    assert set(inner.index.get_level_values("date").unique()) == dates_in

