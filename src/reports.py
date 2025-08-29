from __future__ import annotations
from pathlib import Path
from typing import Any
import pandas as pd
import json
import datetime as _dt
from pathlib import Path

from .data_io import load_indices, get_index_series

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


def _try_plot_monthly_returns(
    backtest_monthly: Path,
    figs_dir: Path,
    date_range: dict | None = None,
    dpi: int = 150,
    figsize: tuple[int, int] = (10, 4),
) -> list[Path]:
    created: list[Path] = []
    if not backtest_monthly.exists():
        return created
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return created
    try:
        df = pd.read_parquet(backtest_monthly)
    except Exception:
        return created
    col = "ret_net_m" if "ret_net_m" in df.columns else ("ret_gross_m" if "ret_gross_m" in df.columns else None)
    if col is None:
        return created
    ser = pd.to_numeric(df[col], errors="coerce")
    ser.index = pd.to_datetime(df.index if df.index.name == "month_end" else df.get("month_end", df.index))
    ser = ser.sort_index()
    # Clip to configured date_range if provided
    if isinstance(date_range, dict):
        start = date_range.get("start")
        end = date_range.get("end")
        if start is not None:
            ser = ser[ser.index >= pd.to_datetime(start)]
        if end is not None:
            ser = ser[ser.index <= pd.to_datetime(end)]
    figp = figs_dir / "monthly_returns.png"
    plt.figure(figsize=figsize, dpi=dpi)
    colors = ser.apply(lambda x: "#2ca02c" if x >= 0 else "#d62728")
    plt.bar(ser.index, ser.values, color=colors, width=20)
    plt.title("Monthly Returns (Strategy)")
    plt.xlabel("Month")
    plt.ylabel("Return")
    plt.tight_layout()
    plt.savefig(figp)
    plt.close()
    created.append(figp)
    return created


def _try_plot_benchmark_compare(
    backtest_monthly: Path,
    indices_dir: Path,
    figs_dir: Path,
    dpi: int = 150,
    figsize: tuple[int, int] = (10, 5),
    names: list[str] | None = None,
) -> list[Path]:
    created: list[Path] = []
    if not backtest_monthly.exists() or not indices_dir.exists():
        return created
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return created
    try:
        dfm = pd.read_parquet(backtest_monthly)
    except Exception:
        return created
    col = "ret_net_m" if "ret_net_m" in dfm.columns else ("ret_gross_m" if "ret_gross_m" in dfm.columns else None)
    if col is None:
        return created
    strat = pd.to_numeric(dfm[col], errors="coerce").rename("Strategy")
    # Align to month-end index
    me = pd.to_datetime(dfm.index if dfm.index.name == "month_end" else dfm.get("month_end", dfm.index))
    me = pd.DatetimeIndex(me)
    strat.index = me
    # Load indices and compute monthly returns for each benchmark
    try:
        idx_df = load_indices(indices_dir, names=names or ["VNINDEX", "HNX-INDEX", "VN30"])
    except Exception:
        return created
    bench_series = {}
    for nm in (names or ["VNINDEX", "HNX-INDEX", "VN30"]):
        try:
            close = get_index_series(idx_df, nm).sort_index()
            aligned = close.reindex(close.index.union(me)).ffill().reindex(me)
            bench_series[nm] = aligned.pct_change().rename(nm)
        except Exception:
            continue
    if not bench_series:
        return created
    # Build cumulative returns
    cr = pd.DataFrame({k: (1.0 + v.fillna(0.0)).cumprod() - 1.0 for k, v in bench_series.items()})
    cr = cr.join(((1.0 + strat.fillna(0.0)).cumprod() - 1.0).rename("Strategy"))
    cr = cr.dropna(how="all")
    if cr.empty:
        return created
    figp = figs_dir / "cumret_vs_benchmarks.png"
    plt.figure(figsize=figsize, dpi=dpi)
    for col in cr.columns:
        plt.plot(cr.index, cr[col], label=col)
    plt.legend()
    plt.title("Cumulative Return vs Benchmarks")
    plt.xlabel("Month")
    plt.ylabel("Cumulative Return")
    plt.tight_layout()
    plt.savefig(figp)
    plt.close()
    created.append(figp)
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

    # Additional figures: monthly returns bar chart and benchmark comparison
    # Resolve indices dir and date_range if present in cfg
    raw_dirs = (cfg.get("raw_dirs", {}) if isinstance(cfg, dict) else {}) or {}
    indices_dir = Path(raw_dirs.get("indices", "vn_indices"))
    date_range = (cfg.get("date_range", {}) if isinstance(cfg, dict) else {}) or {}
    figs += _try_plot_monthly_returns(backtest_monthly, figs_dir, date_range=date_range, dpi=dpi, figsize=(10, 4))
    figs += _try_plot_benchmark_compare(backtest_monthly, indices_dir, figs_dir, dpi=dpi, figsize=(10, 5), names=["VNINDEX", "HNX-INDEX", "VN30"])

    # Compose Markdown (include links to created artifacts and metrics if present)
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
    # Metrics section (best-effort from metrics CSVs)
    try:
        metrics_csv = Path((cfg.get("out", {}) or {}).get("metrics_summary", "data/clean/metrics_summary.csv"))  # type: ignore[union-attr]
        alpha_csv = Path((cfg.get("out", {}) or {}).get("alpha_newey_west", "data/clean/alpha_newey_west.csv"))  # type: ignore[union-attr]
        if metrics_csv.exists():
            mdf = pd.read_csv(metrics_csv)
            if not mdf.empty:
                m = mdf.iloc[0].to_dict()
                def _fmt_pct(x: float) -> str:
                    try:
                        return f"{100.0*float(x):.2f}%"
                    except Exception:
                        return "nan"
                def _fmt(x: float) -> str:
                    try:
                        return f"{float(x):.3f}"
                    except Exception:
                        return "nan"
                lines.extend([
                    "",
                    "## Metrics",
                    f"- N months: {int(m.get('N_months', 0))}",
                    f"- CAGR: {_fmt_pct(m.get('CAGR', float('nan')))}",
                    f"- Vol (ann): {_fmt(m.get('vol_ann', float('nan')))}",
                    f"- Sharpe: {_fmt(m.get('Sharpe', float('nan')))}",
                    f"- IR (vs benchmark): {_fmt(m.get('IR', float('nan')))}",
                    f"- Max Drawdown: {_fmt_pct(m.get('maxDD', float('nan')))}",
                    f"- DD duration (months): {int(m.get('dd_duration', 0))}",
                    f"- Beta: {_fmt(m.get('beta', float('nan')))}",
                ])
        if alpha_csv.exists():
            adf = pd.read_csv(alpha_csv)
            if not adf.empty:
                a = adf.iloc[0].to_dict()
                def _fmt_pct2(x: float) -> str:
                    try:
                        return f"{100.0*float(x):.2f}%"
                    except Exception:
                        return "nan"
                def _fmt2(x: float) -> str:
                    try:
                        return f"{float(x):.3f}"
                    except Exception:
                        return "nan"
                lines.extend([
                    "",
                    "## Alpha (Newey–West)",
                    f"- Alpha (ann.): {_fmt_pct2(a.get('alpha_ann', float('nan')))}",
                    f"- t(alpha): {_fmt2(a.get('t_alpha', float('nan')))}",
                    f"- p-value: {_fmt2(a.get('p_value', float('nan')))}",
                    f"- Beta: {_fmt2(a.get('beta', float('nan')))}",
                ])
    except Exception:
        # metrics optional; ignore errors
        pass
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
