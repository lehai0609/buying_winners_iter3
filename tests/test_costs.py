from __future__ import annotations

import numpy as np
import pandas as pd


def _mk_trades(months: list[pd.Timestamp], tickers: list[str], dW: float) -> pd.DataFrame:
    rows = []
    for m in months:
        for tk in tickers:
            rows.append({
                "month_end": m,
                "ticker": tk,
                "prev_weight": 0.0,
                "target_weight": dW,
                "trade_dW": dW,
                "side": "buy" if dW > 0 else ("sell" if dW < 0 else "none"),
            })
    return pd.DataFrame(rows)


def test_zero_trades_no_costs():
    from src.costs import apply_trading_costs, CostsConfig

    m = pd.Timestamp("2021-01-31")
    trades = _mk_trades([m], ["AAA", "BBB"], dW=0.0)
    cfg = CostsConfig(per_side_bps=25.0, use_adv=False, slippage_per_turnover_bps=0.0)
    costed, summary = apply_trading_costs(trades, adv=None, config=cfg)
    assert (costed[["fees_bps", "slippage_bps", "impact_bps", "total_cost_bps"]] == 0.0).all().all()
    assert float(summary.loc[0, "gross_turnover"]) == 0.0 and float(summary.loc[0, "total_cost_bps"]) == 0.0


def test_doubling_turnover_doubles_costs_adv_off():
    from src.costs import apply_trading_costs, CostsConfig

    m = pd.Timestamp("2021-01-31")
    cfg = CostsConfig(per_side_bps=10.0, use_adv=False, slippage_per_turnover_bps=30.0)

    t1 = _mk_trades([m], ["AAA", "BBB"], dW=0.10)
    c1, s1 = apply_trading_costs(t1, adv=None, config=cfg)
    total1 = float(s1.loc[0, "total_cost_bps"])  # already in bps-return units

    t2 = _mk_trades([m], ["AAA", "BBB"], dW=0.20)
    c2, s2 = apply_trading_costs(t2, adv=None, config=cfg)
    total2 = float(s2.loc[0, "total_cost_bps"])

    # Doubling |dW| should roughly double costs (fees + slippage scale linearly in |dW|)
    assert np.isclose(total2, 2.0 * total1, rtol=1e-12, atol=1e-12)


def test_adv_slippage_linearity_and_cap():
    from src.costs import apply_trading_costs, CostsConfig

    m = pd.Timestamp("2021-02-28")
    trades = pd.DataFrame({
        "month_end": [m, m],
        "ticker": ["AAA", "BBB"],
        "trade_dW": [0.05, 0.50],  # 5% and 50% of NAV
    })
    adv = pd.DataFrame({
        "month_end": [m, m],
        "ticker": ["AAA", "BBB"],
        "adv_value": [50_000_000.0, 50_000_000.0],  # 50m VND
    })
    # NAV = 100m VND, slope 2 bps per 1% ADV, cap 10 bps
    cfg = CostsConfig(
        per_side_bps=0.0,
        use_adv=True,
        capital_vnd=100_000_000.0,
        slippage_bps_per_1pct_adv=2.0,
        slippage_cap_bps=10.0,
        impact_model="none",
    )
    costed, _ = apply_trading_costs(trades, adv=adv, config=cfg)
    # AAA: notional=5m; participation=10%; base slippage=2*10=20 bps, capped at 10 bps; scaled by |dW|=0.05 -> 0.5 bps
    a = float(costed.loc[costed["ticker"] == "AAA", "slippage_bps"].iloc[0])
    assert np.isclose(a, 0.10 * 0.05 * 100.0 / 100.0) or np.isclose(a, 0.5)  # explicitly 0.5 bps
    # BBB: notional=50m; participation=100%; base=200 bps; capped at 10; scaled by 0.50 -> 5 bps
    b = float(costed.loc[costed["ticker"] == "BBB", "slippage_bps"].iloc[0])
    assert np.isclose(b, 5.0, atol=1e-12)


def test_impact_threshold_applies_only_above_threshold():
    from src.costs import apply_trading_costs, CostsConfig

    m = pd.Timestamp("2021-03-31")
    trades = pd.DataFrame({
        "month_end": [m, m],
        "ticker": ["LOW", "HIGH"],
        "trade_dW": [0.05, 0.05],
    })
    # ADV such that participation is 4% and 10%
    capital = 100_000_000.0
    notional = 0.05 * capital
    adv_low = notional / 0.04  # 4%
    adv_high = notional / 0.10  # 10%
    adv = pd.DataFrame({
        "month_end": [m, m],
        "ticker": ["LOW", "HIGH"],
        "adv_value": [adv_low, adv_high],
    })
    cfg = CostsConfig(
        per_side_bps=0.0,
        use_adv=True,
        capital_vnd=capital,
        impact_model="threshold",
        impact_threshold_pct_adv=10.0,
        impact_bps=8.0,
        slippage_bps_per_1pct_adv=0.0,
        slippage_cap_bps=100.0,
    )
    costed, _ = apply_trading_costs(trades, adv=adv, config=cfg)
    # LOW below threshold -> no impact; HIGH >= threshold -> impact applied scaled by |dW|
    imp_low = float(costed.loc[costed["ticker"] == "LOW", "impact_bps"].iloc[0])
    imp_high = float(costed.loc[costed["ticker"] == "HIGH", "impact_bps"].iloc[0])
    assert np.isclose(imp_low, 0.0)
    assert np.isclose(imp_high, 8.0 * 0.05)


def test_compute_monthly_adv_alignment_and_min_days():
    from src.costs import compute_monthly_adv

    # Build 10 trading days in Jan for one ticker
    days = pd.date_range("2021-01-04", periods=10, freq="B")
    df = pd.DataFrame({
        "date": np.repeat(days, 1),
        "ticker": ["AAA"] * len(days),
        "close": np.linspace(10, 19, len(days)),
        "volume": np.linspace(1_000, 2_000, len(days)),
    }).set_index(["date", "ticker"]).sort_index()
    mes = [pd.Timestamp("2021-01-31")]  # month-end
    # With min_days greater than available, ADV missing (NaN)
    adv = compute_monthly_adv(df, mes, window_days=21, min_days=15)
    assert pd.isna(float(adv.loc[0, "adv_value"]))
    # With min_days <= available, ADV present
    adv2 = compute_monthly_adv(df, mes, window_days=10, min_days=5)
    assert not pd.isna(float(adv2.loc[0, "adv_value"]))


def test_integration_enrich_trades_and_summary(tmp_path):
    from src.costs import compute_costs

    # Minimal trades and OHLCV
    mes = [pd.Timestamp("2021-01-31"), pd.Timestamp("2021-02-28")]
    trades = pd.DataFrame({
        "month_end": [mes[0], mes[1]],
        "ticker": ["AAA", "AAA"],
        "prev_weight": [0.0, 0.5],
        "target_weight": [0.5, 0.0],
        "trade_dW": [0.5, -0.5],
        "side": ["buy", "sell"],
    })
    # Daily OHLCV for ADV (won't be used if use_adv=False)
    days = pd.date_range("2021-01-01", periods=25, freq="B")
    ohlcv = pd.DataFrame({
        "date": np.repeat(days, 1),
        "ticker": ["AAA"] * len(days),
        "close": np.linspace(10, 12.4, len(days)),
        "volume": np.linspace(1000, 1500, len(days)),
    }).set_index(["date", "ticker"]).sort_index()

    cfg = {
        "costs": {
            "per_side_bps": 25.0,
            "use_adv": False,
            "slippage_per_turnover_bps": 10.0,
        }
    }

    out_costed = tmp_path / "portfolio_trades_costed.parquet"
    out_summary = tmp_path / "costs_summary.csv"

    costed, summary = compute_costs(
        cfg_dict=cfg,
        trades_df=trades,
        ohlcv_df=ohlcv,
        write=True,
        out_trades_costed=str(out_costed),
        out_summary=str(out_summary),
    )
    # Files written
    assert out_costed.exists() and out_summary.exists()
    # Basic schema checks
    for col in [
        "fees_bps",
        "slippage_bps",
        "impact_bps",
        "total_cost_bps",
        "participation_pct_adv",
    ]:
        assert col in costed.columns
    # Summary contains expected columns
    for col in ["gross_turnover", "fees_bps", "slippage_bps", "total_cost_bps", "n_trades"]:
        assert col in summary.columns

