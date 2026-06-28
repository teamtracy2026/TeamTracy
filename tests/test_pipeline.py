"""Tests for the Kalman filter and cointegration screening.

All tests run on synthetic data so they need no network access. We construct a
genuinely cointegrated pair (a shared random-walk factor plus stationary noise)
and an independent pair, and assert the pipeline tells them apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import backtest_pair
from src.cointegration import correlation_candidates, screen_pairs
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

    # Low correlation floor so every combination is a candidate on synthetic data.
    results = screen_pairs(prices, min_correlation=0.0, max_pvalue=1.0)
    assert results, "expected at least one screened pair"

    # The cointegrated AAA/BBB pair should be the strongest (lowest p-value).
    best = results[0]
    assert {best.y, best.x} == {"AAA", "BBB"}
    assert best.pvalue < 0.05


def test_correlation_prefilter_keeps_correlated_pair():
    y, x = _cointegrated_pair(n=300, seed=5)
    rng = np.random.default_rng(6)
    indep = pd.Series(100 + np.cumsum(rng.normal(0, 1, 300)), index=y.index)
    prices = pd.DataFrame({"AAA": y, "BBB": x, "CCC": indep})
    # AAA/BBB move together (high return correlation); the prefilter should keep
    # them and is allowed to drop the uncorrelated pairs.
    cands = correlation_candidates(prices, min_correlation=0.5, max_candidates=10)
    assert ("AAA", "BBB") in cands or ("BBB", "AAA") in cands


def test_backtest_starts_at_capital_and_compounds():
    y, x = _cointegrated_pair(n=400, seed=7)
    kf = kalman_hedge_ratio(y, x)
    bt = backtest_pair(y, x, kf.zscore, entry=1.5, exit=0.5, initial_capital=100_000)
    assert len(bt.equity) == len(bt.returns)
    # Equity compounds from the starting capital.
    expected_first = 100_000 * (1 + bt.returns.iloc[0])
    assert bt.equity.iloc[0] == pytest.approx(expected_first, rel=1e-9)
    assert bt.initial_capital == 100_000
    assert bt.num_trades >= 0
    assert -1.0 <= bt.max_drawdown <= 0.0
