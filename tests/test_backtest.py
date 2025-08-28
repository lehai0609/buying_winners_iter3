from __future__ import annotations

import numpy as np
import pandas as pd


def _mk_ohlcv(days: pd.DatetimeIndex, ticker: str, closes: list[float], vols: list[float] | float = 1000.0) -> pd.DataFrame:
    if isinstance(vols, (int, float)):
        vols = [vols] * len(days)
    df = pd.DataFrame({
        "date": np.repeat(days, 1),
        "ticker": [ticker] * len(days),
        "open": closes,
        "high": [c * 1.01 for c in closes],
        "low": [c * 0.99 for c in closes],
        "close": closes,
        "volume": vols,
    }).set_index(["date", "ticker"]).sort_index()
    return df


def test_exec_timing_and_costs_tplus1(tmp_path):
    from src.backtest import compute_backtest

    # Trading days spanning Jan month-end into Feb
    days = pd.date_range("2021-01-25", periods=7, freq="B")  # 25,26,27,28,29, Feb1, Feb2
    # Close path: flat in Jan, then +2% on Feb1, +1% on Feb2
    closes = [100.0, 100.0, 100.0, 100.0, 100.0, 102.0, 103.02]
    ohlcv = _mk_ohlcv(days, "AAA", closes)

    # Holdings: target 100% at Jan month-end (will execute on Feb1 with lag_days=1)
    holdings = pd.DataFrame({
        "month_end": [pd.Timestamp("2021-01-31")],
        "ticker": ["AAA"],
        "weight": [1.0],
    })

    # Trades costed: 10 bps total cost (already scaled by |dW|)
    trades_costed = pd.DataFrame({
        "month_end": [pd.Timestamp("2021-01-31")],
        "ticker": ["AAA"],
        "prev_weight": [0.0],
        "target_weight": [1.0],
        "trade_dW": [1.0],
        "total_cost_bps": [10.0],
    })

    cfg = {
        "backtest": {
            "lag_days": 1,
            "exec_price": "open",
            "calendar": "union",
            "skip_on_halt": True,
            "skip_on_limit": True,
            "price_limit_pct": 0.07,
        }
    }

    daily, monthly, _, ex = compute_backtest(
        cfg_dict=cfg,
        ohlcv_df=ohlcv,
        holdings_df=holdings,
        trades_costed_df=trades_costed,
        write=False,
    )

    # Exec at Feb 1 (T+1 from Jan month-end)
    feb1 = pd.Timestamp("2021-02-01")
    # Gross return = +2%; Net = 2% - 10bps = 1.9%
    rg = float(daily.loc[feb1, "ret_gross_d"])  # type: ignore[index]
    rn = float(daily.loc[feb1, "ret_net_d"])  # type: ignore[index]
    assert np.isclose(rg, 0.02, atol=1e-12)
    assert np.isclose(rn, 0.019, atol=1e-12)
    # Prior trading day, still zero exposure
    jan29 = pd.Timestamp("2021-01-29")
    assert np.isclose(float(daily.loc[jan29, "ret_gross_d"]), 0.0)  # type: ignore[index]
    # NAV after Feb1
    assert np.isclose(float(daily.loc[feb1, "nav"]), 1.019, atol=1e-12)
    # Executions logged
    assert ex is not None and len(ex) == 1
    assert ex.iloc[0]["skipped_reason"] == "none"
    assert np.isclose(float(ex.iloc[0]["cost_bps_applied"]), 10.0)


def test_halt_skips_and_renormalizes(tmp_path):
    from src.backtest import compute_backtest

    days = pd.date_range("2021-01-25", periods=6, freq="B")  # through Feb1
    closes = [100, 100, 100, 100, 100, 110]  # big move on exec day, irrelevant due to halt
    # Volume zero on exec day -> halt
    vols = [1000, 1000, 1000, 1000, 1000, 0]
    ohlcv = _mk_ohlcv(days, "AAA", closes, vols)

    holdings = pd.DataFrame({
        "month_end": [pd.Timestamp("2021-01-31")],
        "ticker": ["AAA"],
        "weight": [1.0],
    })
    trades_costed = pd.DataFrame({
        "month_end": [pd.Timestamp("2021-01-31")],
        "ticker": ["AAA"],
        "prev_weight": [0.0],
        "target_weight": [1.0],
        "trade_dW": [1.0],
        "total_cost_bps": [50.0],
    })

    cfg = {"backtest": {"lag_days": 1, "skip_on_halt": True}}
    daily, monthly, _, ex = compute_backtest(
        cfg_dict=cfg, ohlcv_df=ohlcv, holdings_df=holdings, trades_costed_df=trades_costed, write=False
    )

    feb1 = days[-1]
    # No trade executed -> zero gross/net; nav stays 1.0
    assert np.isclose(float(daily.loc[feb1, "ret_gross_d"]), 0.0)
    assert np.isclose(float(daily.loc[feb1, "ret_net_d"]), 0.0)
    assert np.isclose(float(daily.loc[feb1, "nav"]), 1.0)
    # Execution shows halt skip and zero applied costs
    assert ex is not None and ex.iloc[0]["skipped_reason"] == "halt"
    assert np.isclose(float(ex.iloc[0]["cost_bps_applied"]), 0.0)


def test_price_limit_skip_behaviour():
    from src.backtest import compute_backtest

    days = pd.date_range("2021-01-28", periods=3, freq="B")  # 28,29, Feb1
    # Feb1: +8% move, with limit at 7% and skip_on_limit=True -> skip
    closes = [100.0, 100.0, 108.0]
    ohlcv = _mk_ohlcv(days, "AAA", closes)
    holdings = pd.DataFrame({
        "month_end": [pd.Timestamp("2021-01-31")],
        "ticker": ["AAA"],
        "weight": [1.0],
    })
    trades_costed = pd.DataFrame({
        "month_end": [pd.Timestamp("2021-01-31")],
        "ticker": ["AAA"],
        "prev_weight": [0.0],
        "target_weight": [1.0],
        "trade_dW": [1.0],
        "total_cost_bps": [25.0],
    })
    cfg = {"backtest": {"lag_days": 1, "skip_on_limit": True, "price_limit_pct": 0.07}}
    daily, monthly, _, ex = compute_backtest(cfg, ohlcv, holdings, trades_costed, write=False)

    feb1 = pd.Timestamp("2021-02-01")
    # Trade skipped -> zero exposure -> zero return
    assert np.isclose(float(daily.loc[feb1, "ret_gross_d"]), 0.0)
    assert ex is not None and ex.iloc[0]["skipped_reason"] == "limit"

