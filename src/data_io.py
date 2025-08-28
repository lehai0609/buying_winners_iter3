from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple
import pandas as pd
import numpy as np
import yaml


def _load_config() -> dict:
    p = Path("config/data.yml")
    if p.exists():
        return yaml.safe_load(p.read_text()) or {}
    return {}


def _ensure_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.MultiIndex) and list(df.index.names) == ["date", "ticker"]:
        return df.sort_index()
    cols = df.columns
    rename = {}
    if "time" in cols and "date" not in cols:
        rename["time"] = "date"
    if rename:
        df = df.rename(columns=rename)
    if "date" not in df.columns or "ticker" not in df.columns:
        raise ValueError("missing required index columns 'date' and 'ticker'")
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    return df.set_index(["date", "ticker"]).sort_index()


def load_ohlcv(paths: Iterable[Path | str], start: str | pd.Timestamp | None = None, end: str | pd.Timestamp | None = None) -> pd.DataFrame:
    required = ["open", "high", "low", "close", "volume"]
    frames = []
    for p in paths:
        pth = Path(p)
        if not pth.exists():
            raise FileNotFoundError(f"input file missing: {pth}")
        ticker = pth.stem.upper()
        if pth.suffix.lower() == ".csv":
            df = pd.read_csv(pth)
        elif pth.suffix.lower() == ".parquet":
            df = pd.read_parquet(pth)
        else:
            raise ValueError(f"unsupported file extension: {pth.suffix}")
        if "time" not in df.columns:
            raise ValueError("missing required column 'time'")
        for c in required:
            if c not in df.columns:
                raise ValueError(f"missing required column '{c}'")
        df = df.copy()
        df["date"] = pd.to_datetime(df["time"], errors="raise")
        for c in required:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["ticker"] = ticker
        frames.append(df[["date", "ticker", *required]])

    if not frames:
        return pd.DataFrame(columns=required).set_index(pd.MultiIndex.from_arrays([[], []], names=["date", "ticker"]))

    out = pd.concat(frames, axis=0, ignore_index=True)
    out = out.set_index(["date", "ticker"]).sort_index()

    if out.index.duplicated().any():
        dups = out.index[out.index.duplicated()].unique()
        raise ValueError(f"duplicate (date,ticker) found: {dups[:5].tolist()} ...")

    if start is not None or end is not None:
        if start is not None:
            start_ts = pd.to_datetime(start)
        else:
            start_ts = out.index.get_level_values("date").min()
        if end is not None:
            end_ts = pd.to_datetime(end)
        else:
            end_ts = out.index.get_level_values("date").max()
        mask = (out.index.get_level_values("date") >= start_ts) & (
            out.index.get_level_values("date") <= end_ts
        )
        out = out.loc[mask].sort_index()

    return out


def validate_ohlcv(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cfg = _load_config()
    required = ["open", "high", "low", "close", "volume"]

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a DataFrame")
    # Ensure index and required columns
    if not isinstance(df.index, pd.MultiIndex) or list(df.index.names) != ["date", "ticker"]:
        df = _ensure_multiindex(df)
    for c in required:
        if c not in df.columns:
            raise ValueError(f"missing required column '{c}'")

    # Coerce dtypes
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop out-of-range per config window
    dr = cfg.get("date_range", {})
    start = pd.to_datetime(dr.get("start", "2010-01-01"))
    end = pd.to_datetime(dr.get("end", "2025-08-31"))
    idx_dates = df.index.get_level_values("date")
    in_range = (idx_dates >= start) & (idx_dates <= end)
    df = df.loc[in_range].sort_index()

    # Hard: uniqueness
    if df.index.duplicated().any():
        raise ValueError("duplicate (date,ticker) entries detected")

    # Build anomaly flags
    anoms_parts = []

    # HARD: Non-positive prices -> remove from clean output
    price_nonpos = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    if price_nonpos.any():
        part = df.loc[price_nonpos, ["open", "high", "low", "close", "volume"]].copy()
        part["rule"] = "price_non_positive"
        part["detail"] = "one or more of open,high,low,close <= 0"
        anoms_parts.append(part)
        # Drop hard failures from clean df
        df = df.loc[~price_nonpos].sort_index()

    # SOFT: Other quality flags
    if not df.empty:
        ohlc_order = (df["high"] < df[["open", "close"]].max(axis=1)) | (
            df["low"] > df[["open", "close"]].min(axis=1)
        ) | (df["high"] < df["low"])
        if ohlc_order.any():
            part = df.loc[ohlc_order, ["open", "high", "low", "close", "volume"]].copy()
            part["rule"] = "ohlc_ordering_violation"
            part["detail"] = "high<max(open,close) or low>min(open,close) or high<low"
            anoms_parts.append(part)

        vol_flag = df["volume"].isna() | (df["volume"] == 0)
        if vol_flag.any():
            part = df.loc[vol_flag, ["open", "high", "low", "close", "volume"]].copy()
            part["rule"] = "volume_zero_or_nan"
            part["detail"] = "volume is NaN or 0"
            anoms_parts.append(part)

        thr = float(cfg.get("extreme_move_abs", 0.50))
        prev_close = df.groupby(level="ticker")["close"].shift(1)
        ret_d = df["close"] / prev_close - 1.0
        extreme = ret_d.abs() > thr
        extreme = extreme & prev_close.notna()
        if extreme.any():
            part = df.loc[extreme, ["open", "high", "low", "close", "volume"]].copy()
            part["rule"] = "extreme_move"
            part["detail"] = ("ret_d=" + ret_d.loc[extreme].round(6).astype(str))
            anoms_parts.append(part)

    if anoms_parts:
        anoms = pd.concat(anoms_parts, axis=0).reset_index()
        # Ensure OHLCV columns are present in the output alongside metadata
        for c in ["open", "high", "low", "close", "volume"]:
            if c not in anoms.columns:
                anoms[c] = np.nan
        anoms = anoms[[
            "date", "ticker", "rule", "detail", "open", "high", "low", "close", "volume"
        ]]
        anoms = anoms.sort_values(["date", "ticker", "rule"]).reset_index(drop=True)
    else:
        anoms = pd.DataFrame(columns=["date", "ticker", "rule", "detail", "open", "high", "low", "close", "volume"])

    return df.sort_index(), anoms


def _norm_index_key(name: str) -> str:
    return name.strip().lower().replace("_", "").replace("-", "")


def load_indices(dir_path: str | Path, names: list[str] | None = None) -> pd.DataFrame:
    dirp = Path(dir_path)
    if not dirp.exists() or not dirp.is_dir():
        raise FileNotFoundError(f"indices dir not found: {dirp}")
    if names is None:
        names = ["VNINDEX", "HNX-INDEX", "VN30"]

    files = {}
    for p in dirp.iterdir():
        if p.is_file() and p.suffix.lower() in {".csv", ".parquet"}:
            files[_norm_index_key(p.stem)] = p

    frames = []
    for nm in names:
        key = _norm_index_key(nm)
        if key not in files:
            raise FileNotFoundError(f"missing required index file for {nm}")
        p = files[key]
        if p.suffix.lower() == ".csv":
            df = pd.read_csv(p)
        else:
            df = pd.read_parquet(p)
        if "time" not in df.columns or "close" not in df.columns:
            raise ValueError(f"index file schema invalid: {p}")
        d = pd.DataFrame({
            "date": pd.to_datetime(df["time"], errors="raise"),
            "index": nm,
            "close": pd.to_numeric(df["close"], errors="coerce"),
        })
        frames.append(d)

    out = pd.concat(frames, axis=0, ignore_index=True)
    out = out.sort_values(["date", "index"]).reset_index(drop=True)
    if out.duplicated(["date", "index"]).any():
        raise ValueError("duplicate (date,index) detected in indices")
    return out


def get_index_series(indices_df: pd.DataFrame, name: str) -> pd.Series:
    mask = indices_df["index"] == name
    if not mask.any():
        raise KeyError(f"index not found: {name}")
    s = indices_df.loc[mask, ["date", "close"]].sort_values("date")
    s = s.set_index("date")["close"]
    return s
