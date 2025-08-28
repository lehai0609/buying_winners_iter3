from __future__ import annotations
from pathlib import Path
from typing import Any
import pandas as pd
import json
import datetime as _dt

# NOTE: Existing helpers (write_coverage, write_anomalies, write_hard_errors)
# remain unchanged for backward compatibility and tests.


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


# ----- M10: Reporting assembly (lightweight, dependency-safe) -----

def _ensure_dir(p: str | Path) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _default_report_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    report = {
        "out_dir": "data/clean/report",
        "figs_dir": "data/clean/report/figs",
        "tables_dir": "data/clean/report/tables",
        "include": {
            "equity_curve": True,
            "drawdown": True,
            "turnover": True,
            "jk_heatmap": True,
            "subperiods": True,
            "alpha_regression": True,
            "capacity_curve": False,
        },
        "figure_dpi": 150,
        "figsize": [9, 5],
    }
    user = (cfg or {}).get("report", {})
    # shallow merge for simple structure
    report.update({k: v for k, v in user.items() if k != "include"})
    if isinstance(user.get("include"), dict):
        report["include"].update(user["include"])  # type: ignore[index]
    return report


def _try_plot_equity(backtest_daily: Path, figs_dir: Path, dpi: int = 150, figsize: tuple[int, int] = (9, 5)) -> list[Path]:
    created: list[Path] = []
    if not backtest_daily.exists():
        return created
    try:
        import matplotlib.pyplot as plt  # local import, optional dep
    except Exception:
        return created
    try:
        df = pd.read_parquet(backtest_daily)
    except Exception:
        return created
    # Use either `nav` or cumulative of ret_net_d
    if "nav" in df.columns:
        nav = df["nav"].astype(float)
    else:
        ret = df.get("ret_net_d")
        if ret is None:
            return created
        nav = (1.0 + ret.astype(float).fillna(0)).cumprod()
    fig1 = figs_dir / "equity_curve.png"
    fig2 = figs_dir / "drawdown.png"
    # Equity
    plt.figure(figsize=figsize, dpi=dpi)
    nav.plot()
    plt.title("Equity Curve (Net)")
    plt.xlabel("Date")
    plt.ylabel("NAV")
    plt.tight_layout()
    plt.savefig(fig1)
    plt.close()
    created.append(fig1)
    # Drawdown
    dd = nav / nav.cummax() - 1.0
    plt.figure(figsize=figsize, dpi=dpi)
    dd.plot(color="red")
    plt.title("Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(fig2)
    plt.close()
    created.append(fig2)
    return created


def _try_table_monthly(backtest_monthly: Path, tables_dir: Path) -> list[Path]:
    created: list[Path] = []
    if not backtest_monthly.exists():
        return created
    try:
        df = pd.read_parquet(backtest_monthly)
    except Exception:
        return created
    out = tables_dir / "monthly_returns.csv"
    cols = [c for c in df.columns if c in ("ret_net_m", "ret_gross_m")]
    if not cols:
        return created
    tmp = df[cols].copy()
    tmp.to_csv(out)
    created.append(out)
    return created


def assemble_report(cfg: dict[str, Any] | None = None, run_id: str | None = None) -> dict[str, Any]:
    """Assemble a lightweight Markdown report.

    Returns a dict with created artifact paths useful for experiment tracking.
    This function is dependency-safe: plotting is skipped if matplotlib missing.
    """
    cfg = cfg or {}
    report_cfg = _default_report_cfg(cfg)
    out_dir = _ensure_dir(report_cfg["out_dir"])  # type: ignore[index]
    figs_dir = _ensure_dir(report_cfg["figs_dir"])  # type: ignore[index]
    tables_dir = _ensure_dir(report_cfg["tables_dir"])  # type: ignore[index]
    dpi = int(report_cfg.get("figure_dpi", 150))
    fs = report_cfg.get("figsize", [9, 5])
    figsize = (int(fs[0]), int(fs[1])) if isinstance(fs, (list, tuple)) and len(fs) == 2 else (9, 5)

    # Attempt to locate standard inputs from cfg
    out_cfg = cfg.get("out", {}) if isinstance(cfg, dict) else {}
    backtest_daily = Path("data/clean/backtest_daily.parquet")
    backtest_monthly = Path("data/clean/backtest_monthly.parquet")
    # Prefer explicit config keys if provided
    if isinstance(out_cfg, dict):
        backtest_daily = Path(out_cfg.get("backtest_daily", backtest_daily))
        backtest_monthly = Path(out_cfg.get("backtest_monthly", backtest_monthly))

    figs: list[Path] = []
    tables: list[Path] = []

    # Plots and tables (best-effort)
    figs += _try_plot_equity(backtest_daily, figs_dir, dpi=dpi, figsize=figsize)  # equity + dd
    tables += _try_table_monthly(backtest_monthly, tables_dir)

    # Compose Markdown (include links to created artifacts)
    report_md = out_dir / "momentum_report.md"
    created_at = _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
    lines = [
        "# Momentum Strategy Report",
        "",
        f"Generated: {created_at}",
        f"Run ID: {run_id or 'N/A'}",
        "",
        "## Overview",
        "This report summarizes the backtest results, equity curve, drawdowns, and monthly returns.",
    ]
    if figs:
        lines.append("\n## Figures")
        for p in figs:
            rel = p.relative_to(out_dir)
            lines.append(f"- {rel.as_posix()}")
    if tables:
        lines.append("\n## Tables")
        for p in tables:
            rel = p.relative_to(out_dir)
            lines.append(f"- {rel.as_posix()}")
    report_md.write_text("\n".join(lines), encoding="utf-8")

    return {
        "report_md": str(report_md),
        "figs": [str(p) for p in figs],
        "tables": [str(p) for p in tables],
        "inputs": {
            "backtest_daily": str(backtest_daily),
            "backtest_monthly": str(backtest_monthly),
        },
        "out_dirs": {
            "out_dir": str(out_dir),
            "figs_dir": str(figs_dir),
            "tables_dir": str(tables_dir),
        },
        "meta": {"created_at": created_at, "run_id": run_id},
    }
