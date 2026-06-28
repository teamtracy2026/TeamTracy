"""Kalman-filtered dynamic hedge ratio for a pair of price series.

The classic pairs-trading spread assumes a static hedge ratio estimated once
by OLS: ``spread = y - beta * x``. In practice the relationship drifts, so the
spread computed from a stale beta looks non-stationary even when the pair is
genuinely cointegrated.

We instead model the hedge ratio (and intercept) as a slowly varying hidden
state and recover it with a Kalman filter — a time-varying linear regression:

    Observation:  y_t = beta_t * x_t + alpha_t + e_t,   e_t ~ N(0, R)
    State:        [alpha_t, beta_t] = [alpha_{t-1}, beta_{t-1}] + w_t,  w_t ~ N(0, Q)

The filtered residual ``e_t`` is the smoothed spread. Because the state adapts,
the residual stays mean-reverting and its z-score is a clean trading signal.

The implementation is a self-contained two-state Kalman filter — no external
filtering dependency — so it stays easy to read and to test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class KalmanResult:
    """Output of :func:`kalman_hedge_ratio`.

    All series share the index of the input prices.
    """

    alpha: pd.Series          # filtered intercept over time
    beta: pd.Series           # filtered hedge ratio over time
    spread: pd.Series         # filtered residual (smoothed spread)
    zscore: pd.Series         # rolling z-score of the spread


def kalman_hedge_ratio(
    y: pd.Series,
    x: pd.Series,
    delta: float = 1e-4,
    obs_cov: float = 1e-3,
    z_window: int = 60,
) -> KalmanResult:
    """Estimate a time-varying hedge ratio and smoothed spread via Kalman filter.

    Parameters
    ----------
    y, x:
        Price series for the two legs. ``y`` is regressed on ``x``; the hedge
        ratio ``beta_t`` is "units of x per unit of y". They must share an index.
    delta:
        Controls the state transition covariance ``Q = delta/(1-delta) * I``.
        Larger ``delta`` lets the hedge ratio adapt faster (more responsive,
        noisier); smaller makes it stiffer. ``1e-4`` is a common default.
    obs_cov:
        Observation noise variance ``R``. Larger values trust the model state
        more and the latest observation less, smoothing the spread further.
    z_window:
        Lookback (in bars) for the rolling mean/std used to z-score the spread.

    Returns
    -------
    KalmanResult with the filtered ``alpha``, ``beta``, ``spread`` and ``zscore``.
    """
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(df) < 2:
        raise ValueError("Need at least two aligned observations for the filter.")

    yv = df["y"].to_numpy(dtype=float)
    xv = df["x"].to_numpy(dtype=float)
    n = len(df)

    # State: theta = [alpha, beta]. Observation matrix at t is H_t = [1, x_t].
    trans_cov = delta / (1.0 - delta) * np.eye(2)   # Q
    R = float(obs_cov)

    theta = np.zeros(2)              # state mean, initialised flat
    P = np.eye(2) * 1.0             # state covariance, diffuse-ish prior

    alpha_out = np.empty(n)
    beta_out = np.empty(n)
    spread_out = np.empty(n)

    for t in range(n):
        # --- Predict ---
        # State transition is identity, so predicted mean == previous mean.
        P = P + trans_cov

        # --- Update ---
        H = np.array([1.0, xv[t]])           # 1x2 observation row
        y_hat = H @ theta                    # predicted observation
        resid = yv[t] - y_hat                # innovation == smoothed spread
        S = H @ P @ H.T + R                  # innovation variance (scalar)
        K = (P @ H) / S                      # Kalman gain (2-vector)

        theta = theta + K * resid
        P = P - np.outer(K, H) @ P

        alpha_out[t] = theta[0]
        beta_out[t] = theta[1]
        spread_out[t] = resid

    idx = df.index
    alpha = pd.Series(alpha_out, index=idx, name="alpha")
    beta = pd.Series(beta_out, index=idx, name="beta")
    spread = pd.Series(spread_out, index=idx, name="spread")

    win = max(2, min(z_window, n))
    roll = spread.rolling(win, min_periods=max(2, win // 2))
    zscore = ((spread - roll.mean()) / roll.std(ddof=0)).rename("zscore")

    return KalmanResult(alpha=alpha, beta=beta, spread=spread, zscore=zscore)


def half_life(spread: pd.Series) -> float:
    """Mean-reversion half-life of a spread, in bars.

    Fits the Ornstein-Uhlenbeck discretisation
    ``d_spread_t = lambda * spread_{t-1} + c`` by OLS and converts the decay
    coefficient to a half-life ``-ln(2)/lambda``. Returns ``inf`` when the
    spread shows no mean reversion (``lambda >= 0``).
    """
    s = spread.dropna()
    if len(s) < 3:
        return float("inf")
    lag = s.shift(1).dropna()
    delta = (s - s.shift(1)).dropna()
    lag = lag.loc[delta.index]
    # OLS of delta on lag with intercept.
    X = np.column_stack([np.ones(len(lag)), lag.to_numpy()])
    beta_hat, *_ = np.linalg.lstsq(X, delta.to_numpy(), rcond=None)
    lam = beta_hat[1]
    if lam >= 0:
        return float("inf")
    return float(-np.log(2) / lam)
