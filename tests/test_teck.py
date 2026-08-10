"""Tests for the Teck Resources module (parsing + predictor analysis).

Network-free: exercises the CSV parser, performance-stat computation and the
predictor regression on synthetic data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.teck import (
    _parse_stooq_csv,
    compute_performance_stats,
    predictor_analysis,
)


def _synthetic_close(n: int = 800, seed: int = 0) -> pd.Series:
    idx = pd.bdate_range("2021-01-01", periods=n)
    rng = np.random.default_rng(seed)
    return pd.Series(
        30 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, n))), index=idx, name="Close"
    )


def test_parse_stooq_csv():
    csv = (
        "Date,Open,High,Low,Close,Volume\n"
        "2024-01-03,40.5,42.0,40.0,41.8,1200000\n"
        "2024-01-02,40.0,41.0,39.5,40.5,1000000\n"  # out of order on purpose
    )
    df = _parse_stooq_csv(csv)
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    # Parser sorts ascending by date.
    assert df.index.is_monotonic_increasing
    assert df["Close"].iloc[-1] == pytest.approx(41.8)


def test_compute_performance_stats_keys():
    close = _synthetic_close()
    ohlcv = pd.DataFrame({"Close": close, "Volume": np.full(len(close), 1_000_000)})
    stats = compute_performance_stats(ohlcv)
    for key in ("Last close", "1-year", "YTD", "52-week range", "Annualised volatility"):
        assert key in stats
    assert stats["Last close"].startswith("$")


def test_predictor_analysis_identifies_dominant_driver():
    teck = _synthetic_close(seed=1)
    rng = np.random.default_rng(2)
    # Copper proxy tracks Teck closely; the market proxy is independent.
    copper = teck * np.exp(rng.normal(0, 0.01, len(teck)))
    spy = pd.Series(
        400 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(teck)))),
        index=teck.index,
    )
    preds = pd.DataFrame({"Copper miners (COPX)": copper, "S&P 500 (SPY)": spy})

    res = predictor_analysis(teck, preds)
    assert res.top_predictor == "Copper miners (COPX)"
    assert res.correlations.iloc[0] > res.correlations.iloc[-1]
    assert 0.0 <= res.r_squared <= 1.0
    # Copper should carry the larger standardised coefficient.
    assert abs(res.betas["Copper miners (COPX)"]) > abs(res.betas["S&P 500 (SPY)"])


def test_predictor_analysis_needs_enough_data():
    teck = _synthetic_close(n=10)
    preds = pd.DataFrame({"X": _synthetic_close(n=10, seed=5)})
    with pytest.raises(ValueError):
        predictor_analysis(teck, preds)
