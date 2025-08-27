from __future__ import annotations
from pathlib import Path
import pandas as pd
import pytest


def _ohlcv_fixture() -> pd.DataFrame:
    # Year 2020: AAA has 2 dates, BBB has 1 date → union dates=2, tickers=2 → rows=3 → missing_days = 2*2 - 3 = 1
    rows = [
        {"date": "2020-01-01", "ticker": "AAA", "open": 9.0, "high": 9.2, "low": 8.8, "close": 9.0, "volume": 100},
        {"date": "2020-01-02", "ticker": "AAA", "open": 10.0, "high": 10.2, "low": 9.8, "close": 10.0, "volume": 110},
        {"date": "2020-01-01", "ticker": "BBB", "open": 18.0, "high": 18.5, "low": 17.9, "close": 18.2, "volume": 200},
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])  # ensure datetime type
    return df.set_index(["date", "ticker"]).sort_index()


def test_write_coverage_computes_yearly_counts_and_missing_days(tmp_path: Path):
    from src.reports import write_coverage

    df = _ohlcv_fixture()
    out = tmp_path / "coverage.csv"
    write_coverage(df, str(out))

    assert out.exists()
    cov = pd.read_csv(out)
    # Required columns
    for col in ["year", "tickers", "rows", "missing_days"]:
        assert col in cov.columns
    # Single year present with expected numbers
    row = cov.loc[cov["year"] == 2020].iloc[0]
    assert int(row["tickers"]) == 2
    assert int(row["rows"]) == 3
    assert int(row["missing_days"]) == 1


def test_write_anomalies_persists_expected_schema(tmp_path: Path):
    from src.reports import write_anomalies

    anoms = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
            "ticker": ["AAA", "BBB"],
            "rule": ["extreme_move", "zero_or_nan_volume"],
            "detail": ["ret_d=0.70", "volume=0"],
        }
    )
    out = tmp_path / "anoms.csv"
    write_anomalies(anoms, str(out))

    assert out.exists()
    df = pd.read_csv(out)
    for col in ["date", "ticker", "rule"]:
        assert col in df.columns
    assert len(df) == 2
