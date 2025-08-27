from pathlib import Path
import pandas as pd
import pytest


# M1: Data ingest adapted to current layout: per-ticker files in `HSX/` and
# `HNX/` use Parquet in production, but the loader must accept CSV or Parquet.
# Files contain columns: time, open, high, low, close, volume (no ticker).
# Ticker is inferred from filename and normalized to uppercase. Returned frame
# is indexed by [date, ticker], sorted, and typed. Date filters are inclusive.


def _write_file(p: Path, df: pd.DataFrame, ext: str) -> Path:
    if ext == "csv":
        df.to_csv(p, index=False)
    elif ext == "parquet":
        df.to_parquet(p, index=False)
    else:
        raise ValueError(f"unsupported ext: {ext}")
    return p


@pytest.mark.parametrize("ext", ["csv", "parquet"])
def test_loads_files_multiindex_typed_sorted_ticker_normalized(tmp_path: Path, ext: str):
    from src.data_io import load_ohlcv

    aaa = pd.DataFrame(
        {
            "time": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "open": [9.0, 9.5, 10.1],
            "high": [9.2, 10.2, 10.6],
            "low": [8.8, 9.3, 9.9],
            "close": [9.0, 10.0, 10.5],
            "volume": [120, 110, 100],
        }
    )
    bbb = pd.DataFrame(
        {
            "time": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "open": [18.5, 19.0, 19.5],
            "high": [19.2, 19.7, 20.3],
            "low": [18.4, 18.9, 18.9],
            "close": [19.0, 19.5, 20.0],
            "volume": [220, 210, 200],
        }
    )
    p1 = _write_file(tmp_path / f"AAA.{ext}", aaa, ext)
    # Lowercase filename should still yield uppercase ticker
    p2 = _write_file(tmp_path / f"bbb.{ext}", bbb, ext)

    df = load_ohlcv(paths=[p1, p2])

    assert isinstance(df.index, pd.MultiIndex)
    assert list(df.index.names) == ["date", "ticker"]
    assert pd.api.types.is_datetime64_any_dtype(df.index.get_level_values("date"))
    assert df.index.is_monotonic_increasing
    assert set(df.index.get_level_values("ticker").unique()) == {"AAA", "BBB"}

    required = ["open", "high", "low", "close", "volume"]
    for c in required:
        assert c in df.columns
        assert pd.api.types.is_numeric_dtype(df[c])


@pytest.mark.parametrize("ext", ["csv", "parquet"])
def test_duplicate_date_ticker_raises(tmp_path: Path, ext: str):
    from src.data_io import load_ohlcv

    df = pd.DataFrame(
        {
            "time": ["2020-01-02", "2020-01-02"],
            "open": [10.0, 10.0],
            "high": [10.5, 10.5],
            "low": [9.5, 9.5],
            "close": [10.0, 10.0],
            "volume": [100, 100],
        }
    )
    p = _write_file(tmp_path / f"DUP.{ext}", df, ext)

    with pytest.raises(Exception) as exc:
        load_ohlcv(paths=[p])
    assert any(k in str(exc.value).lower() for k in ["duplicate", "duplicated", "unique"])


@pytest.mark.parametrize("ext", ["csv", "parquet"])
def test_missing_required_column_raises(tmp_path: Path, ext: str):
    from src.data_io import load_ohlcv

    df = pd.DataFrame(
        {
            "time": ["2020-01-02"],
            "open": [10.0],
            "high": [10.1],
            "low": [9.9],
            # "close" missing
            "volume": [100],
        }
    )
    p = _write_file(tmp_path / f"MISS.{ext}", df, ext)

    with pytest.raises(Exception) as exc:
        load_ohlcv(paths=[p])
    assert any(k in str(exc.value).lower() for k in ["missing", "required", "column"])


@pytest.mark.parametrize("ext", ["csv", "parquet"])
def test_date_filter_returns_inclusive_range(tmp_path: Path, ext: str):
    from src.data_io import load_ohlcv

    aaa = pd.DataFrame(
        {
            "time": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "open": [9.0, 9.5, 10.1],
            "high": [9.2, 10.2, 10.6],
            "low": [8.8, 9.3, 9.9],
            "close": [9.0, 10.0, 10.5],
            "volume": [120, 110, 100],
        }
    )
    bbb = pd.DataFrame(
        {
            "time": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "open": [18.5, 19.0, 19.5],
            "high": [19.2, 19.7, 20.3],
            "low": [18.4, 18.9, 18.9],
            "close": [19.0, 19.5, 20.0],
            "volume": [220, 210, 200],
        }
    )
    p1 = _write_file(tmp_path / f"AAA.{ext}", aaa, ext)
    p2 = _write_file(tmp_path / f"BBB.{ext}", bbb, ext)

    df = load_ohlcv(paths=[p1, p2], start="2020-01-02", end="2020-01-03")
    assert len(df) == 4  # 2 tickers × 2 dates
    dates = df.index.get_level_values("date")
    assert dates.min() == pd.Timestamp("2020-01-02")
    assert dates.max() == pd.Timestamp("2020-01-03")


@pytest.mark.parametrize("ext", ["csv", "parquet"])
def test_known_split_fixture_preserves_adjusted_close_continuity(tmp_path: Path, ext: str):
    from src.data_io import load_ohlcv

    df = pd.DataFrame(
        {
            "time": ["2020-01-01", "2020-01-02"],
            "open": [50.0, 50.0],
            "high": [50.5, 50.6],
            "low": [49.5, 49.6],
            "close": [50.0, 50.0],
            "volume": [1000, 2000],
        }
    )
    p = _write_file(tmp_path / f"SPLT.{ext}", df, ext)

    out = load_ohlcv(paths=[p])
    sp = out.xs("SPLT", level="ticker").sort_index()
    assert pytest.approx(sp["close"].iloc[0]) == pytest.approx(sp["close"].iloc[1])
