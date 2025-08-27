from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import pytest


def _df_from_rows(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])  # ensure datetime type
    df = df.set_index(["date", "ticker"]).sort_index()
    return df


def test_soft_rules_flag_but_do_not_drop_rows():
    # Prices include zero/negative and OHLC ordering violations; volume has NaN/zero.
    rows = [
        {"date": "2020-01-01", "ticker": "AAA", "open": -1.0, "high": 9.0, "low": 8.0, "close": 9.0, "volume": 0},
        {"date": "2020-01-02", "ticker": "AAA", "open": 10.0, "high": 9.5, "low": 9.9, "close": 10.1, "volume": np.nan},  # high < open
        {"date": "2020-01-03", "ticker": "AAA", "open": 10.0, "high": 10.5, "low": 10.1, "close": 10.2, "volume": 100},  # low > open
    ]
    df = _df_from_rows(rows)

    from src.data_io import validate_ohlcv

    clean, anoms = validate_ohlcv(df)

    # Soft rules: rows retained in clean df
    assert len(clean) == len(df)
    # Anomalies include price and/or OHLC ordering and volume flags
    rules = " ".join(str(r).lower() for r in anoms.get("rule", []))
    assert any(k in rules for k in ["price", "posit", "non-positive"])  # price positivity
    assert any(k in rules for k in ["ohlc", "order"])  # OHLC ordering
    assert "volume" in rules  # zero/NaN volume flagged


def test_extreme_daily_move_flag_no_lookahead_and_per_ticker():
    # AAA jumps by +70% on day 2 (flag); BBB stays modest
    rows = [
        {"date": "2020-01-01", "ticker": "AAA", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 10},
        {"date": "2020-01-02", "ticker": "AAA", "open": 170.0, "high": 171.0, "low": 169.0, "close": 170.0, "volume": 10},
        {"date": "2020-01-01", "ticker": "BBB", "open": 50.0, "high": 50.5, "low": 49.5, "close": 50.0, "volume": 5},
        {"date": "2020-01-02", "ticker": "BBB", "open": 52.0, "high": 52.5, "low": 51.5, "close": 52.0, "volume": 5},
    ]
    df = _df_from_rows(rows)

    from src.data_io import validate_ohlcv

    clean, anoms = validate_ohlcv(df)
    # First day cannot be flagged for extreme move (no prior close)
    assert not any(
        (row["ticker"] == "AAA" and pd.to_datetime(row["date"]) == pd.Timestamp("2020-01-01") and "extreme" in str(row.get("rule", "")).lower())
        for _, row in anoms.iterrows()
    )
    # Second day for AAA flagged as extreme; BBB not flagged
    assert any(
        (row["ticker"] == "AAA" and pd.to_datetime(row["date"]) == pd.Timestamp("2020-01-02") and "extreme" in str(row.get("rule", "")).lower())
        for _, row in anoms.iterrows()
    )
    assert not any(row["ticker"] == "BBB" and "extreme" in str(row.get("rule", "")).lower() for _, row in anoms.iterrows())


def test_duplicate_date_ticker_is_hard_error():
    rows = [
        {"date": "2020-01-02", "ticker": "AAA", "open": 10.0, "high": 10.5, "low": 9.5, "close": 10.0, "volume": 100},
        {"date": "2020-01-02", "ticker": "AAA", "open": 10.0, "high": 10.5, "low": 9.5, "close": 10.0, "volume": 100},
    ]
    df = _df_from_rows(rows)

    from src.data_io import validate_ohlcv

    with pytest.raises(Exception) as exc:
        validate_ohlcv(df)
    assert any(k in str(exc.value).lower() for k in ["duplicate", "duplicated", "unique"])


def test_missing_required_columns_raise():
    # Drop "open"
    rows = [
        {"date": "2020-01-01", "ticker": "AAA", "high": 10.0, "low": 9.8, "close": 9.9, "volume": 100}
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])  # ensure datetime type
    df = df.set_index(["date", "ticker"]).sort_index()

    from src.data_io import validate_ohlcv

    with pytest.raises(Exception) as exc:
        validate_ohlcv(df)
    assert any(k in str(exc.value).lower() for k in ["missing", "required", "column"])


def test_date_range_clipped_to_config_window():
    # Include out-of-range dates; expect clipping to [2010-01-01, 2025-08-31] from config
    rows = [
        {"date": "2009-12-31", "ticker": "AAA", "open": 10, "high": 10.5, "low": 9.5, "close": 10.1, "volume": 1},
        {"date": "2010-01-01", "ticker": "AAA", "open": 10, "high": 10.5, "low": 9.5, "close": 10.1, "volume": 1},
        {"date": "2025-08-31", "ticker": "AAA", "open": 10, "high": 10.5, "low": 9.5, "close": 10.1, "volume": 1},
        {"date": "2026-01-01", "ticker": "AAA", "open": 10, "high": 10.5, "low": 9.5, "close": 10.1, "volume": 1},
    ]
    df = _df_from_rows(rows)

    from src.data_io import validate_ohlcv

    clean, anoms = validate_ohlcv(df)
    dates = clean.index.get_level_values("date")
    assert dates.min() >= pd.Timestamp("2010-01-01")
    assert dates.max() <= pd.Timestamp("2025-08-31")
    # Out-of-range dates removed
    assert pd.Timestamp("2009-12-31") not in dates
    assert pd.Timestamp("2026-01-01") not in dates
