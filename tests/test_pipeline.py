"""Tests for the Kalman filter and cointegration screening.

All tests run on synthetic data so they need no network access. We construct a
genuinely cointegrated pair (a shared random-walk factor plus stationary noise)
and an independent pair, and assert the pipeline tells them apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.cointegration import screen_pairs
from src.kalman import half_life, kalman_hedge_ratio


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=n)


def _cointegrated_pair(n: int = 500, beta: float = 1.5, seed: int = 0):
    """y = beta * x + stationary noise, with x a random walk -> cointegrated."""
    rng = np.random.default_rng(seed)
    x = 50 + np.cumsum(rng.normal(0, 1, n))
    noise = rng.normal(0, 1.0, n)  # stationary residual
    y = beta * x + 10 + noise
    idx = _dates(n)
    return pd.Series(y, index=idx, name="Y"), pd.Series(x, index=idx, name="X")


def test_kalman_recovers_hedge_ratio():
    y, x = _cointegrated_pair(beta=1.5)
    res = kalman_hedge_ratio(y, x)
    # The filtered beta should converge near the true 1.5 by the end of the series.
    assert res.beta.iloc[-1] == pytest.approx(1.5, abs=0.2)
    # Spread should be the residual: roughly zero-mean and much smaller than prices.
    assert abs(res.spread.iloc[100:].mean()) < 1.0
    assert res.spread.std() < y.std()


def test_kalman_output_shapes_align():
    y, x = _cointegrated_pair(n=300)
    res = kalman_hedge_ratio(y, x)
    for s in (res.alpha, res.beta, res.spread, res.zscore):
        assert len(s) == 300
        assert s.index.equals(y.index)


def test_half_life_finite_for_mean_reverting_spread():
    # AR(1) with positive persistence < 1 mean-reverts -> finite half-life.
    rng = np.random.default_rng(1)
    n = 400
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = 0.9 * s[t - 1] + rng.normal(0, 1)
    hl = half_life(pd.Series(s, index=_dates(n)))
    assert np.isfinite(hl) and hl > 0


def test_screen_ranks_cointegrated_above_random():
    y, x = _cointegrated_pair(n=400, seed=2)
    rng = np.random.default_rng(3)
    # Two independent random walks: not cointegrated.
    r1 = pd.Series(100 + np.cumsum(rng.normal(0, 1, 400)), index=y.index)
    r2 = pd.Series(100 + np.cumsum(rng.normal(0, 1, 400)), index=y.index)

    prices = pd.DataFrame({"AAA": y, "BBB": x, "CCC": r1, "DDD": r2})

    # within_sector_only=False so every combination is tested on synthetic names.
    results = screen_pairs(prices, within_sector_only=False, max_pvalue=1.0)
    assert results, "expected at least one screened pair"

    # The cointegrated AAA/BBB pair should be the strongest (lowest p-value).
    best = results[0]
    assert {best.y, best.x} == {"AAA", "BBB"}
    assert best.pvalue < 0.05
