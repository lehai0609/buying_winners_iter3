import pandas as pd
import numpy as np

from src import robustness as rb
from src.cv import cross_validate


def _mk_daily_ohlcv(start: str, months: int, tickers=("AAA", "BBB")) -> pd.DataFrame:
    dates = pd.date_range(start, periods=months * 21, freq="B")
    rows = []
    px = {tk: 10.0 + i * 2.0 for i, tk in enumerate(tickers)}
    for d in dates:
        for i, tk in enumerate(tickers):
            # slightly different drifts
            drift = 1.0005 + i * 0.0001
            px[tk] *= drift
            rows.append({
                "date": d,
                "ticker": tk,
                "open": px[tk],
                "high": px[tk],
                "low": px[tk],
                "close": px[tk],
                "volume": 1000 + i * 100,
            })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])  # ensure datetime
    return df.set_index(["date", "ticker"]).sort_index()


def test_cost_sensitivity_monotone(tmp_path):
    ohlcv = _mk_daily_ohlcv("2020-01-01", months=24)
    p = tmp_path / "ohlcv.parquet"
    ohlcv.to_parquet(p)
    # Write a simple monthly universe marking all tickers eligible
    from src.calendar import build_trading_grid, month_ends
    grid = build_trading_grid(ohlcv, calendar="union")
    me = month_ends(grid)
    uni = pd.DataFrame({
        "month_end": list(me) * ohlcv.index.get_level_values("ticker").nunique(),
        "ticker": sum([[tk] * len(me) for tk in ohlcv.index.get_level_values("ticker").unique()], []),
        "eligible": [True] * (len(me) * ohlcv.index.get_level_values("ticker").nunique()),
    })
    pu = tmp_path / "monthly_universe.parquet"
    uni.to_parquet(pu, index=False)

    cfg = {
        "out": {"ohlcv_parquet": str(p), "monthly_universe_parquet": str(pu)},
        "signals": {"momentum": {"lookback_months": 6, "gap_months": 1, "n_deciles": 5, "min_names_per_month": 1}},
        "portfolio": {"k_months": 3},
        "backtest": {"lag_days": 1},
        "costs": {"per_side_bps": 0.0, "use_adv": False, "slippage_per_turnover_bps": 0.0},
    }
    params = {"J": 6, "K": 3}
    grid = [0.0, 10.0, 50.0]
    out = rb.cost_sensitivity(params, grid, cfg)
    assert set(["J", "K", "cost_bps", "sharpe", "ret_ann", "alpha_nw", "vol_ann", "turnover"]) <= set(out.columns)
    # monotone: higher costs should not improve Sharpe/CAGR
    out = out.sort_values("cost_bps")
    assert np.all(np.diff(out["ret_ann"].values) <= 1e-12)
    assert np.all(np.diff(out["sharpe"].values) <= 1e-12)


def test_subperiod_metrics_shapes(tmp_path):
    ohlcv = _mk_daily_ohlcv("2020-01-01", months=18)
    p = tmp_path / "ohlcv.parquet"
    ohlcv.to_parquet(p)
    # Write a simple monthly universe marking all tickers eligible
    from src.calendar import build_trading_grid, month_ends
    grid = build_trading_grid(ohlcv, calendar="union")
    me = month_ends(grid)
    uni = pd.DataFrame({
        "month_end": list(me) * ohlcv.index.get_level_values("ticker").nunique(),
        "ticker": sum([[tk] * len(me) for tk in ohlcv.index.get_level_values("ticker").unique()], []),
        "eligible": [True] * (len(me) * ohlcv.index.get_level_values("ticker").nunique()),
    })
    pu = tmp_path / "monthly_universe.parquet"
    uni.to_parquet(pu, index=False)

    cfg = {
        "out": {"ohlcv_parquet": str(p), "monthly_universe_parquet": str(pu)},
        "signals": {"momentum": {"lookback_months": 6, "gap_months": 1, "n_deciles": 5, "min_names_per_month": 1}},
        "portfolio": {"k_months": 3},
        "backtest": {"lag_days": 1},
        "costs": {"per_side_bps": 10.0, "use_adv": False, "slippage_per_turnover_bps": 0.0},
    }
    params = {"J": 6, "K": 3}
    subperiods = [
        {"name": "pre", "start": "2020-01-31", "end": "2020-12-31"},
        {"name": "post", "start": "2021-01-31", "end": "2021-12-31"},
    ]
    out = rb.subperiod_metrics(params, subperiods, cfg)
    assert set(out["name"]) == {"pre", "post"}
    assert out["N_months"].ge(0).all()
