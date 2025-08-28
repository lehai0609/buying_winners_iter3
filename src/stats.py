from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from .metrics import perf_summary
from .utils import make_rng


def _align_xy(
    returns_m: pd.Series, benchmark_m: pd.Series, rf: float | pd.Series = 0.0
) -> tuple[np.ndarray, np.ndarray, int, pd.DatetimeIndex]:
    r = pd.Series(returns_m).dropna()
    b = pd.Series(benchmark_m).dropna()
    r.index = pd.to_datetime(r.index)
    b.index = pd.to_datetime(b.index)
    idx = r.index.intersection(b.index)
    r = r.loc[idx]
    b = b.loc[idx]
    if isinstance(rf, pd.Series):
        rf = pd.Series(rf).reindex(idx).fillna(0.0)
        y = (r - rf).values
        x = (b - rf).values
    else:
        rf_m = float(rf)
        y = (r - rf_m).values
        x = (b - rf_m).values
    return y, x, len(idx), idx


def _nw_covariance(U: np.ndarray, lags: int) -> np.ndarray:
    """Newey–West HAC covariance of summed score vectors.

    U: Txp matrix where each row t is the score u_t = e_t * X_t (p-vector).
    Returns: p x p covariance estimate (without dividing by T in X'X), to be
    used inside (X'X)^{-1} S (X'X)^{-1}.
    """
    T, p = U.shape
    if T == 0:
        return np.full((p, p), np.nan)
    # Bartlett weights
    L = int(max(0, lags))
    S = U.T @ U  # lag 0
    for k in range(1, L + 1):
        w = 1.0 - k / (L + 1.0)
        # sum_{t=k..T-1} u_t u_{t-k}'
        S_k = U[k:].T @ U[:-k]
        S += w * (S_k + S_k.T)
    return S


def alpha_newey_west(
    returns_m: pd.Series,
    benchmark_m: pd.Series,
    rf: float | pd.Series = 0.0,
    lags: int = 6,
    intercept: bool = True,
) -> dict:
    """Estimate alpha with HAC (Newey–West) SEs on monthly data.

    Model: r_p - rf = alpha + beta * (r_b - rf) + eps.
    Returns dict {alpha_m, alpha_ann, beta, t_alpha, p_value, lags, n_obs}.
    """
    y, x, n, idx = _align_xy(returns_m, benchmark_m, rf=rf)
    if n == 0:
        return {
            "alpha_m": np.nan,
            "alpha_ann": np.nan,
            "beta": np.nan,
            "t_alpha": np.nan,
            "p_value": np.nan,
            "lags": lags,
            "n_obs": 0,
        }
    x = x.reshape(-1, 1)
    if intercept:
        X = np.column_stack([np.ones_like(x), x])
    else:
        X = x
    # OLS coefficients
    XtX = X.T @ X
    XtY = X.T @ y
    beta_hat = np.linalg.solve(XtX, XtY)
    y_hat = X @ beta_hat
    resid = y - y_hat
    # HAC covariance for coefficients
    U = resid.reshape(-1, 1) * X  # Txp
    S = _nw_covariance(U, lags=lags)
    XtX_inv = np.linalg.inv(XtX)
    cov = XtX_inv @ S @ XtX_inv
    # Extract alpha and beta
    if intercept:
        alpha_m = float(beta_hat[0])
        beta_coef = float(beta_hat[1])
        var_alpha = float(cov[0, 0])
    else:
        alpha_m = 0.0
        beta_coef = float(beta_hat[0])
        var_alpha = np.nan
    se_alpha = float(np.sqrt(var_alpha)) if np.isfinite(var_alpha) and var_alpha >= 0 else np.nan
    t_alpha = alpha_m / se_alpha if se_alpha and se_alpha > 0 else np.nan
    # Two-sided p-value using normal approximation
    try:
        from math import erf, sqrt

        def _norm_cdf(z: float) -> float:
            return 0.5 * (1.0 + erf(z / sqrt(2.0)))

        if np.isfinite(t_alpha):
            p_value = 2.0 * (1.0 - _norm_cdf(abs(t_alpha)))
        else:
            p_value = np.nan
    except Exception:
        p_value = np.nan

    alpha_ann = (1.0 + alpha_m) ** 12 - 1.0 if np.isfinite(alpha_m) else np.nan
    return {
        "alpha_m": alpha_m,
        "alpha_ann": float(alpha_ann) if np.isfinite(alpha_ann) else np.nan,
        "beta": beta_coef,
        "t_alpha": float(t_alpha) if np.isfinite(t_alpha) else np.nan,
        "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
        "lags": int(lags),
        "n_obs": int(n),
    }


def _block_bootstrap_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    b = max(1, int(block_size))
    if n <= 0:
        return np.array([], dtype=int)
    if b > n:
        b = n
    starts = rng.integers(0, n - b + 1, size=1 + n // b)
    idx = []
    for s in starts:
        idx.extend(range(int(s), int(s) + b))
        if len(idx) >= n:
            break
    return np.array(idx[:n], dtype=int)


def bootstrap_cis(
    returns_m: pd.Series,
    benchmark_m: pd.Series | None,
    metric: str,
    n_boot: int = 1000,
    block_size: int = 6,
    seed: int | None = None,
    lags: int = 6,
) -> dict:
    """Block-bootstrap confidence intervals for Sharpe or alpha.

    metric: "Sharpe" or "alpha_ann".
    Returns dict {low, high, level, n_boot, block_size}.
    """
    r = pd.Series(returns_m).dropna()
    r.index = pd.to_datetime(r.index)
    if benchmark_m is not None:
        b = pd.Series(benchmark_m).dropna()
        b.index = pd.to_datetime(b.index)
        idx = r.index.intersection(b.index)
        r = r.loc[idx]
        b = b.loc[idx]
    else:
        b = None
    n = len(r)
    if n == 0:
        return {"low": np.nan, "high": np.nan, "level": 0.95, "n_boot": 0, "block_size": block_size}

    rng = make_rng(seed if seed is not None else 42)
    vals = []
    for _ in range(int(n_boot)):
        idx = _block_bootstrap_indices(n, block_size, rng)
        rr = pd.Series(r.values[idx])
        if metric.lower().startswith("sharpe"):
            mu = rr.mean()
            sd = rr.std(ddof=1)
            s = mu / sd * np.sqrt(12.0) if sd and sd > 0 else np.nan
            vals.append(s)
        elif metric.lower().startswith("alpha"):
            if b is None:
                vals.append(np.nan)
            else:
                bb = pd.Series(b.values[idx])
                res = alpha_newey_west(rr, bb, rf=0.0, lags=lags, intercept=True)
                vals.append(res.get("alpha_ann", np.nan))
        else:
            raise ValueError("unsupported metric for bootstrap: use 'Sharpe' or 'alpha_ann'")
    arr = np.asarray(vals, dtype=float)
    low = float(np.nanquantile(arr, 0.025))
    high = float(np.nanquantile(arr, 0.975))
    return {"low": low, "high": high, "level": 0.95, "n_boot": int(n_boot), "block_size": int(block_size)}

