from __future__ import annotations
from pathlib import Path
import pandas as pd


def write_coverage(df: pd.DataFrame, out_path: str | Path) -> None:
    idx = df.index
    if not isinstance(idx, pd.MultiIndex) or list(idx.names) != ["date", "ticker"]:
        raise ValueError("expected df indexed by [date, ticker]")
    tmp = df.reset_index()[["date", "ticker"]].copy()
    tmp["year"] = tmp["date"].dt.year
    rows = tmp.groupby("year", as_index=False).size().rename(columns={"size": "rows"})
    tickers = (
        tmp.groupby("year")["ticker"].nunique().reset_index().rename(columns={"ticker": "tickers"})
    )
    days = (
        tmp.groupby("year")["date"].nunique().reset_index().rename(columns={"date": "unique_dates"})
    )
    cov = rows.merge(tickers, on="year").merge(days, on="year")
    cov["missing_days"] = cov["unique_dates"] * cov["tickers"] - cov["rows"]
    cov = cov[["year", "tickers", "rows", "missing_days"]].sort_values("year").reset_index(drop=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cov.to_csv(out_path, index=False)


def write_anomalies(anoms: pd.DataFrame, out_path: str | Path) -> None:
    req = {"date", "ticker", "rule"}
    missing = req - set(anoms.columns)
    if missing:
        raise ValueError(f"anomalies missing required columns: {sorted(missing)}")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    anoms.to_csv(out_path, index=False)


def write_hard_errors(errors: pd.DataFrame | list[dict], out_path: str | Path) -> None:
    """Persist hard validation errors for auditing.

    Accepts either a DataFrame with columns like [stage, error, ...] and optional
    context columns (e.g., date, ticker, open, high, low, close, volume), or a
    list of dicts. Writes to CSV and ensures parent directory exists.
    """
    if isinstance(errors, list):
        if not errors:
            df = pd.DataFrame(columns=["stage", "error"])  # empty
        else:
            df = pd.DataFrame(errors)
    else:
        df = errors
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
