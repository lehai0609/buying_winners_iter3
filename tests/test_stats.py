import numpy as np
import pandas as pd
import pytest

from src.stats import alpha_newey_west, bootstrap_cis
from src.utils import fixed_seed, make_rng


def _mk_index(n: int, start: str = "2010-01-31") -> pd.DatetimeIndex:
    start_ts = pd.to_datetime(start)
    return pd.date_range(start_ts, periods=n, freq="ME")


def test_alpha_newey_west_recovers_known_alpha():
    n = 240
    idx = _mk_index(n)
    alpha_true = 0.005  # 0.5% per month
    beta_true = 1.2
    with fixed_seed(7):
        rng = make_rng(7)
        b = pd.Series(0.01 + 0.02 * rng.standard_normal(n), index=idx)
        # AR(1) residuals to justify HAC
        eps = np.zeros(n)
        phi = 0.3
        z = 0.005 * rng.standard_normal(n)
        for t in range(1, n):
            eps[t] = phi * eps[t - 1] + z[t]
        y = alpha_true + beta_true * b.values + eps
        r = pd.Series(y, index=idx)

    res = alpha_newey_west(r, b, rf=0.0, lags=6, intercept=True)
    # Alpha within tolerance; allow some slack due to noise
    assert res["alpha_m"] == pytest.approx(alpha_true, abs=3e-3)
    assert res["beta"] == pytest.approx(beta_true, rel=0.1)
    assert res["n_obs"] == n
    # t_alpha should be positive and reasonably large
    assert np.isfinite(res["t_alpha"]) and res["t_alpha"] > 1.0


def test_bootstrap_cis_deterministic():
    n = 120
    idx = _mk_index(n)
    with fixed_seed(99):
        rng = make_rng(99)
        b = pd.Series(0.01 + 0.02 * rng.standard_normal(n), index=idx)
        r = pd.Series(0.005 + 1.0 * b.values + 0.01 * rng.standard_normal(n), index=idx)

    ci1 = bootstrap_cis(r, b, metric="Sharpe", n_boot=500, block_size=6, seed=123)
    ci2 = bootstrap_cis(r, b, metric="Sharpe", n_boot=500, block_size=6, seed=123)
    assert ci1 == ci2

    ca1 = bootstrap_cis(r, b, metric="alpha_ann", n_boot=500, block_size=6, seed=123)
    ca2 = bootstrap_cis(r, b, metric="alpha_ann", n_boot=500, block_size=6, seed=123)
    assert ca1 == ca2
