from __future__ import annotations

import numpy as np
import pandas as pd


def _mk_signals_one_month(month: pd.Timestamp, n: int = 10) -> pd.DataFrame:
    # momentum increasing 0..n-1, valid True; deciles 1..10 repeated or capped
    vals = np.arange(n, dtype=float)
    decs = np.linspace(1, 10, num=n).round().astype(int)
    pct = np.linspace(0.0, 1.0, num=n)
    return pd.DataFrame(
        {
            "month_end": [month] * n,
            "ticker": [f"T{i:02d}" for i in range(n)],
            "momentum": vals,
            "valid": [True] * n,
            "decile": decs,
            "pct_rank": pct,
        }
    )


def test_select_winners_modes():
    from src.portfolio import select_winners

    month = pd.Timestamp("2021-01-31")
    sig = _mk_signals_one_month(month, n=10)

    w_dec = select_winners(sig, selection="top_decile", long_decile=10)
    assert (w_dec["month_end"] == month).all() and set(w_dec["ticker"]).issubset(set(sig[sig.decile == 10]["ticker"]))

    w_q = select_winners(sig, selection="top_quantile", top_quantile=0.2)
    # top 20% -> pct_rank >= 0.8
    expect = set(sig[sig.pct_rank >= 0.8]["ticker"])
    assert set(w_q["ticker"]) == expect


def test_build_overlapping_portfolio_overlap_and_sums():
    from src.portfolio import build_overlapping_portfolio

    # Winners for 3 consecutive months, 3 names each
    months = pd.date_range("2021-01-31", periods=3, freq="ME")
    rows = []
    for i, m in enumerate(months):
        for j in range(3):
            rows.append({"month_end": m, "ticker": f"N{i}{j}"})
    winners = pd.DataFrame(rows)

    K = 3
    hold = build_overlapping_portfolio(winners, k_months=K, renorm_within=True)
    # At first month, only 1 cohort active, weight sum should be 1/K
    m0 = months[0]
    sub0 = hold[hold["month_end"] == m0]
    assert abs(sub0["weight"].sum() - (1.0 / K)) < 1e-12
    # Each name in cohort: equal split of 1/K across 3 names
    expected_each = (1.0 / K) / 3.0
    assert np.allclose(sub0["weight"].values, np.full(3, expected_each))
    # At third month, 3 cohorts active, total weight ~ 1
    m2 = months[2]
    sub2 = hold[hold["month_end"] == m2]
    assert abs(sub2.groupby("ticker")["weight"].sum().sum() - 1.0) < 1e-12


def test_renormalization_on_drop_with_active_mask():
    from src.portfolio import build_overlapping_portfolio

    # One cohort at t0 with 4 names, K=3
    t0 = pd.Timestamp("2021-01-31")
    winners = pd.DataFrame({"month_end": [t0] * 4, "ticker": ["A", "B", "C", "D"]})
    # Drop one name (D) at t1 via active_mask
    t1 = pd.Timestamp("2021-02-28")
    mask = pd.DataFrame(
        {
            "month_end": [t0, t0, t0, t0, t1, t1, t1, t1],
            "ticker": ["A", "B", "C", "D", "A", "B", "C", "D"],
            "active": [True, True, True, True, True, True, True, False],
        }
    )
    K = 3
    hold = build_overlapping_portfolio(winners, k_months=K, renorm_within=True, active_mask=mask)
    # At t1, only 3 names alive in cohort -> each gets (1/K)/3
    sub1 = hold[hold["month_end"] == t1]
    assert len(sub1) == 3
    expected = (1.0 / K) / 3.0
    assert np.allclose(sorted(sub1["weight"].tolist()), [expected] * 3)
    # Cohort sum remains 1/K
    assert abs(sub1["weight"].sum() - (1.0 / K)) < 1e-12


def test_holdings_to_trades_basic():
    from src.portfolio import holdings_to_trades

    # Simple 2 months, one name increases from 0 to 0.5 then to 0.0
    m0, m1 = pd.to_datetime(["2021-01-31", "2021-02-28"]) 
    holdings = pd.DataFrame(
        {
            "month_end": [m0, m1],
            "ticker": ["AAA", "AAA"],
            "weight": [0.5, 0.0],
        }
    )
    trades = holdings_to_trades(holdings)
    # First month buy 0.5, second month sell 0.5
    t0 = trades[trades["month_end"] == m0].iloc[0]
    t1 = trades[trades["month_end"] == m1].iloc[0]
    assert abs(float(t0["trade_dW"]) - 0.5) < 1e-12 and t0["side"] == "buy"
    assert abs(float(t1["trade_dW"]) + 0.5) < 1e-12 and t1["side"] == "sell"


def test_summary_cash_weight_basic(tmp_path):
    # Minimal winners in Jan only, K=2 => invested=0.5 each month until cohort expires
    me_jan = pd.Timestamp("2021-01-31")
    me_feb = pd.Timestamp("2021-02-28")
    sig = pd.DataFrame(
        {
            "month_end": [me_jan, me_jan],
            "ticker": ["A", "B"],
            "momentum": [1.0, 2.0],
            "valid": [True, True],
            "decile": [10, 10],
        }
    )

    from src.portfolio import compute_portfolio

    cfg = {
        "portfolio": {
            "k_months": 2,
            "selection": "top_decile",
            "long_decile": 10,
            "renormalize_within_cohort": True,
            "exclude_on_missing_price": False,
            "redistribute_across_cohorts": False,
        }
    }

    out_sum = tmp_path / "portfolio_summary.csv"
    out_hold = tmp_path / "holdings.parquet"
    out_trd = tmp_path / "trades.parquet"

    holdings = compute_portfolio(
        cfg_dict=cfg,
        signals_df=sig,
        write=True,
        out_holdings=str(out_hold),
        out_trades=str(out_trd),
        out_summary=str(out_sum),
    )
    # Summary CSV contains expected cash weights
    s = pd.read_csv(out_sum)
    s["month_end"] = pd.to_datetime(s["month_end"]) if "month_end" in s.columns else s["month_end"]
    row_j = s.loc[s["month_end"] == me_jan].iloc[0]
    row_f = s.loc[s["month_end"] == me_feb].iloc[0]
    assert abs(float(row_j["cash_weight"]) - 0.5) < 1e-12
    assert abs(float(row_f["cash_weight"]) - 0.5) < 1e-12


def test_generate_trades_caps_and_tplus(tmp_path):
    # Build simple holdings for two months for one name; month-end dates
    m0, m1 = pd.to_datetime(["2021-01-31", "2021-02-28"]) 
    holdings = pd.DataFrame(
        {
            "month_end": [m0, m1, m1],
            "ticker": ["AAA", "AAA", "BBB"],
            # AAA grows to 0.8 then trimmed by cap; BBB enters 0.4
            "weight": [0.6, 0.8, 0.4],
        }
    )
    # Build tiny OHLCV grid spanning next-month first trading days
    dates = pd.to_datetime([
        "2021-01-29",  # last trading day of Jan (Fri)
        "2021-02-01",  # next trading day (Mon)
        "2021-02-26",  # last trading day of Feb (Fri)
        "2021-03-01",  # next trading day (Mon)
    ])
    ohlcv = pd.DataFrame(
        {
            "date": dates.tolist() * 2,
            "ticker": ["AAA"] * 4 + ["BBB"] * 4,
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1000,
        }
    )
    from src.portfolio import generate_trades

    # Apply caps: max_weight_per_name=0.5, turnover_cap=0.2
    trades = generate_trades(
        holdings,
        t_plus=1,
        settlement="T+2",
        ohlcv_df=ohlcv,
        calendar="union",
        max_weight_per_name=0.5,
        turnover_cap=0.2,
    )
    # Check schema
    for c in ["month_end", "ticker", "prev_weight", "target_weight", "trade_dW", "side", "trade_date", "settlement", "settlement_date"]:
        assert c in trades.columns
    # Month 0: AAA capped to 0.5 then scaled by turnover cap (0.2 / 0.25 = 0.8) => 0.4
    t0 = trades[trades["month_end"] == m0].set_index("ticker").loc["AAA"]
    assert abs(float(t0["target_weight"]) - 0.4) < 1e-12
    assert abs(float(t0["trade_dW"]) - 0.4) < 1e-12
    assert t0["side"] == "buy"
    # Trade date is next trading day after true month-end (2021-01-29 -> 2021-02-01)
    assert pd.to_datetime(t0["trade_date"]).date() == pd.Timestamp("2021-02-01").date()
    # Settlement is two trading days after trade_date (2021-02-03 on the grid)
    # Our grid only includes 2021-02-01 and 2021-02-26; shift caps to last available (implementation clips)
    assert trades["settlement"].iloc[0] == "T+2"
    # Month 1: compute gross turnover and ensure scaled to cap
    sub1 = trades[trades["month_end"] == m1]
    gross_turnover = 0.5 * np.abs(sub1["trade_dW"]).sum()
    assert gross_turnover <= 0.2000000001
