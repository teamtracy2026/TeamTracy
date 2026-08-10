"""Tests for the Anglo-Teck merger analysis and projection (network-free)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.merger import (
    MergerTerms,
    annualized_arb_return,
    merger_spread,
    project_gbm,
    scenario_forecast,
    spread_summary,
)


def _series(n: int = 400, start: float = 40.0, seed: int = 0) -> pd.Series:
    idx = pd.bdate_range("2024-01-01", periods=n)
    rng = np.random.default_rng(seed)
    return pd.Series(start * np.exp(np.cumsum(rng.normal(0.0002, 0.02, n))), index=idx)


def test_merger_spread_and_summary():
    teck = _series(seed=1)
    anglo = _series(seed=2, start=30.0)
    ratio = 1.33
    df = merger_spread(teck, anglo, ratio)
    assert {"teck", "anglo", "implied", "spread_pct"} <= set(df.columns)
    # Implied value is ratio * anglo.
    assert df["implied"].iloc[-1] == pytest.approx(ratio * anglo.iloc[-1])
    summ = spread_summary(df)
    # Upside to deal and spread are consistent: teck/implied - 1 == spread,
    # implied/teck - 1 == upside.
    assert summ["spread_pct"] == pytest.approx(df["spread_pct"].iloc[-1])
    assert summ["upside_to_deal"] == pytest.approx(summ["implied"] / summ["teck"] - 1)


def test_annualized_arb_return():
    # 10% upside realised over half a year annualises to ~21%.
    r = annualized_arb_return(0.10, 183)
    assert r == pytest.approx((1.10) ** (365.25 / 183) - 1, rel=1e-6)
    assert np.isnan(annualized_arb_return(0.1, 0))


def test_project_gbm_shapes_and_bands():
    close = _series(n=500)
    proj = project_gbm(close, horizon_days=60, n_paths=2000, seed=7)
    assert len(proj.dates) == 60
    assert list(proj.bands.columns) == ["p05", "p25", "p50", "p75", "p95"]
    # Percentile bands are monotonically ordered at every horizon step.
    b = proj.bands
    assert (b["p05"] <= b["p25"]).all()
    assert (b["p25"] <= b["p50"]).all()
    assert (b["p50"] <= b["p75"]).all()
    assert (b["p75"] <= b["p95"]).all()
    assert 0.0 <= proj.prob_gain <= 1.0
    assert proj.terminal.shape == (2000,)


def test_random_walk_drift_is_zero():
    close = _series(n=300)
    proj = project_gbm(close, horizon_days=30, n_paths=500, drift_mode="random_walk")
    assert proj.drift_daily == 0.0
    hist = project_gbm(close, horizon_days=30, n_paths=500, drift_mode="historical")
    # Historical drift uses the sample mean (generally non-zero).
    assert hist.drift_daily != 0.0


def test_scenario_forecast_blends_toward_deal():
    teck = _series(seed=3, start=40.0)
    # Anglo priced so that ratio*anglo is well above Teck -> deal implies upside.
    anglo = _series(seed=4, start=45.0)
    ratio = 1.33

    low = scenario_forecast(teck, anglo,
                            MergerTerms(exchange_ratio=ratio, completion_prob=0.0,
                                        horizon_days=120), n_paths=3000)
    high = scenario_forecast(teck, anglo,
                             MergerTerms(exchange_ratio=ratio, completion_prob=1.0,
                                         horizon_days=120), n_paths=3000)
    # Deal value here is above Teck, so weighting the deal branch more heavily
    # raises the blended expected price.
    assert high.blended_exp_price > low.blended_exp_price
    assert high.deal_value_now == pytest.approx(ratio * anglo.iloc[-1])
    assert high.implied_upside == pytest.approx(ratio * anglo.iloc[-1] / teck.iloc[-1] - 1)
