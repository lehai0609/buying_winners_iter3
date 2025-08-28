import pandas as pd
import pytest

from src.cv import rolling_splits, param_grid, select_best


def test_rolling_splits_36_12_basic():
    # 60 consecutive month-ends
    idx = pd.date_range("2015-01-31", periods=60, freq="ME")
    folds = rolling_splits(idx, train_months=36, valid_months=12)
    # Anchors at 36 and 48 -> two folds
    assert len(folds) == 2
    (tr0_s, tr0_e, va0_s, va0_e) = folds[0]
    assert (tr0_e + pd.offsets.MonthEnd(1)) == va0_s  # contiguous windows
    assert (va0_s + pd.offsets.MonthEnd(11)).normalize() == va0_e.normalize()


def test_param_grid_cartesian():
    combos = param_grid([3, 6], [1, 3])
    assert set(combos) == {(3, 1), (3, 3), (6, 1), (6, 3)}


def test_select_best_with_tie_breakers():
    rows = [
        {"J": 12, "K": 6, "metric": "sharpe", "value": 1.0, "turnover": 1.2, "ret_ann": 0.20},
        {"J": 12, "K": 3, "metric": "sharpe", "value": 1.0, "turnover": 1.0, "ret_ann": 0.18},
        {"J": 9,  "K": 3, "metric": "sharpe", "value": 0.9, "turnover": 0.9, "ret_ann": 0.22},
    ]
    # Same metric value for first two; lower_turnover should pick K=3
    best = select_best(rows, metric="sharpe", tie_breaker="lower_turnover")
    assert best["K"] == 3


@pytest.mark.skip(reason="Cross-validation orchestration will be implemented in full M9 work")
def test_cross_validate_placeholder():
    from src.cv import cross_validate  # noqa
    assert callable(cross_validate)


def _mk_daily_ohlcv(start: str, months: int, tickers=("AAA", "BBB")) -> pd.DataFrame:
    dates = pd.date_range(start, periods=months * 21, freq="B")
    rows = []
    px = {tk: 10.0 + i * 2.0 for i, tk in enumerate(tickers)}
    for d in dates:
        for i, tk in enumerate(tickers):
            drift = 1.0006 + i * 0.0001
            px[tk] *= drift
            rows.append({
                "date": d,
                "ticker": tk,
                "open": px[tk],
                "high": px[tk],
                "low": px[tk],
                "close": px[tk],
                "volume": 1000 + 10 * i,
            })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])  # ensure datetime
    return df.set_index(["date", "ticker"]).sort_index()


def test_cross_validate_end_to_end_small(tmp_path):
    from src.cv import cross_validate

    ohlcv = _mk_daily_ohlcv("2020-01-01", months=24)
    p = tmp_path / "ohlcv.parquet"
    ohlcv.to_parquet(p)

    cfg = {
        "out": {"ohlcv_parquet": str(p)},
        "signals": {"momentum": {
            "lookback_months": 6,
            "gap_months": 1,
            "n_deciles": 5,
            "min_names_per_month": 1,
            "exclude_hard_errors": True,
            "calendar": "union",
            "price_col": "close",
        }},
        "portfolio": {"k_months": 3, "selection": "top_decile", "long_decile": 5},
        "backtest": {"lag_days": 1},
        "costs": {"per_side_bps": 0.0, "use_adv": False, "slippage_per_turnover_bps": 0.0},
        "cv": {"train_months": 12, "valid_months": 6, "j_grid": [3, 6], "k_grid": [3]},
    }

    res, sel, summ = cross_validate(cfg)
    # Basic schema checks
    assert {"fold_id", "J", "K", "metric", "value"}.issubset(res.columns)
    assert not sel.empty and {"J", "K", "fold_id"}.issubset(sel.columns)
    assert not summ.empty and {"policy", "metric", "value_mean"}.issubset(summ.columns)
