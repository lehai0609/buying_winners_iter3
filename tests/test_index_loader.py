from pathlib import Path
import pandas as pd
import pytest


def _write_index_csv(path: Path, dates: list[str], closes: list[float]) -> None:
    df = pd.DataFrame(
        {
            "time": dates,
            "open": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
            "volume": [0 for _ in closes],
        }
    )
    df.to_csv(path, index=False)


def test_load_indices_returns_long_typed_sorted_unique(tmp_path: Path):
    from src.data_io import load_indices

    d = tmp_path / "vn_indices"
    d.mkdir()

    # Create three simple index CSVs; give HNX name with hyphen to match repo convention
    _write_index_csv(d / "VNINDEX.csv", ["2020-01-01", "2020-01-02"], [900.0, 905.0])
    _write_index_csv(d / "HNX-INDEX.csv", ["2020-01-01", "2020-01-03"], [100.0, 101.0])
    _write_index_csv(d / "VN30.csv", ["2020-01-02", "2020-01-03"], [800.0, 802.0])

    out = load_indices(str(d), names=["VNINDEX", "HNX-INDEX", "VN30"])  # explicit canonical names

    # Schema and types
    assert list(out.columns) == ["date", "index", "close"]
    assert pd.api.types.is_datetime64_any_dtype(out["date"])  # parsed dates
    assert out["index"].dtype == object
    assert pd.api.types.is_numeric_dtype(out["close"])

    # Sorted by date then index; and no duplicate (date,index)
    assert out.sort_values(["date", "index"]).equals(out)
    assert not out.duplicated(["date", "index"]).any()

    # Correct index names present and only expected rows (no forward-fill of missing days)
    assert set(out["index"].unique()) == {"VNINDEX", "HNX-INDEX", "VN30"}
    # Row count equals sum of per-file rows (2 + 2 + 2 = 6)
    assert len(out) == 6

    # Per-index dates match exactly the input sets
    vn_dates = set(pd.to_datetime(["2020-01-01", "2020-01-02"]))
    hnx_dates = set(pd.to_datetime(["2020-01-01", "2020-01-03"]))
    vn30_dates = set(pd.to_datetime(["2020-01-02", "2020-01-03"]))
    got_vn = set(out.loc[out["index"] == "VNINDEX", "date"])
    got_hnx = set(out.loc[out["index"] == "HNX-INDEX", "date"])
    got_vn30 = set(out.loc[out["index"] == "VN30", "date"])
    assert got_vn == vn_dates and got_hnx == hnx_dates and got_vn30 == vn30_dates


def test_missing_required_index_file_raises(tmp_path: Path):
    from src.data_io import load_indices

    d = tmp_path / "vn_indices"
    d.mkdir()

    # Only two indices present
    _write_index_csv(d / "VNINDEX.csv", ["2020-01-01"], [900.0])
    _write_index_csv(d / "VN30.csv", ["2020-01-01"], [800.0])

    with pytest.raises(Exception) as exc:
        load_indices(str(d), names=["VNINDEX", "HNX-INDEX", "VN30"])
    assert any(k in str(exc.value).lower() for k in ["missing", "required", "hnx"])


def test_duplicate_date_index_raises(tmp_path: Path):
    from src.data_io import load_indices

    d = tmp_path / "vn_indices"
    d.mkdir()

    # VNINDEX contains a duplicated date
    _write_index_csv(d / "VNINDEX.csv", ["2020-01-01", "2020-01-01"], [900.0, 901.0])
    _write_index_csv(d / "HNX-INDEX.csv", ["2020-01-01"], [100.0])
    _write_index_csv(d / "VN30.csv", ["2020-01-01"], [800.0])

    with pytest.raises(Exception) as exc:
        load_indices(str(d), names=["VNINDEX", "HNX-INDEX", "VN30"])
    assert any(k in str(exc.value).lower() for k in ["duplicate", "duplicated", "unique"])


def test_name_normalization_from_filename_case_insensitive(tmp_path: Path):
    from src.data_io import load_indices

    d = tmp_path / "vn_indices"
    d.mkdir()

    # Lowercase filenames should be recognized; output index names normalized to uppercase basenames
    _write_index_csv(d / "vnindex.csv", ["2020-01-01"], [900.0])
    _write_index_csv(d / "hnx-index.csv", ["2020-01-01"], [100.0])
    _write_index_csv(d / "vn30.csv", ["2020-01-01"], [800.0])

    out = load_indices(str(d), names=["VNINDEX", "HNX-INDEX", "VN30"])
    assert set(out["index"].unique()) == {"VNINDEX", "HNX-INDEX", "VN30"}
