import numpy as np
import pandas as pd
import pytest

from src.metrics import perf_summary, drawdown_stats, subperiod_metrics
from src.utils import fixed_seed, make_rng


def _mk_index(n: int, start: str = "2010-01-31") -> pd.DatetimeIndex:
    start_ts = pd.to_datetime(start)
    return pd.date_range(start_ts, periods=n, freq="ME")


def test_perf_summary_constant_returns():
    n = 24
    idx = _mk_index(n)
    r = pd.Series(0.01, index=idx)
    s = perf_summary(r)
    assert s["N_months"] == n
    expected_cagr = (1.01 ** 12) - 1.0
    assert s["CAGR"] == pytest.approx(expected_cagr, rel=1e-12, abs=1e-12)
    assert s["vol_ann"] == pytest.approx(0.0, abs=1e-12)
    # Sharpe undefined (std=0) -> NaN
    assert np.isnan(s["Sharpe"]) or s["Sharpe"] == pytest.approx(0.0, abs=1e-12)
    assert s["hit_rate_m"] == pytest.approx(1.0, abs=1e-12)
    assert s["VaR95"] == pytest.approx(0.01, abs=1e-12)
    assert s["VaR99"] == pytest.approx(0.01, abs=1e-12)
    dd = drawdown_stats(r)
    assert dd["maxDD"] == pytest.approx(0.0, abs=1e-12)
    assert dd["dd_duration_months"] == 0


def test_perf_summary_ir_and_beta_against_benchmark():
    n = 120
    idx = _mk_index(n)
    with fixed_seed(123):
        rng = make_rng(123)
        b = pd.Series(0.01 + 0.02 * rng.standard_normal(n), index=idx)
        noise = 0.01 * rng.standard_normal(n)
        r = pd.Series(0.002 + 1.5 * b.values + noise, index=idx)
    s = perf_summary(r, benchmark_m=b, rf=0.0)
    # Information ratio equals Sharpe of excess returns
    ex = (r - b)
    mu, sd = ex.mean(), ex.std(ddof=1)
    ir = mu / sd * np.sqrt(12.0)
    assert s["IR"] == pytest.approx(ir, rel=1e-6, abs=1e-6)
    # Beta close to 1.5
    assert s["beta"] == pytest.approx(1.5, rel=0.05)


def test_drawdown_known_path():
    # +10%, then -50%, then four months of +20% recover; recovery on month 6
    r = pd.Series(
        [0.10, -0.50, 0.20, 0.20, 0.20, 0.20],
        index=_mk_index(6),
    )
    dd = drawdown_stats(r)
    assert dd["maxDD"] == pytest.approx(-0.5, rel=1e-6)
    # Underwater during months 2..5 -> 4 months max duration
    assert dd["dd_duration_months"] == 4


def test_subperiod_metrics_rollup_counts():
    n = 24
    idx = _mk_index(n, start="2015-01-31")
    r = pd.Series(0.01, index=idx)
    periods = [
        (pd.Timestamp("2015-01-31"), pd.Timestamp("2015-12-31")),
        (pd.Timestamp("2016-01-31"), pd.Timestamp("2016-12-31")),
    ]
    df = subperiod_metrics(r, benchmark_m=None, periods=periods)
    assert df["N_months"].sum() == 24
