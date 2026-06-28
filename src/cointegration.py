"""Pairwise cointegration screening.

Ranks candidate pairs by the Engle-Granger cointegration test p-value: the
lower the p-value, the stronger the evidence that a stationary linear
combination of the two price series exists. We restrict candidate pairs to
within-sector combinations by default, which keeps the sweep fast and the hits
economically sensible.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from .kalman import KalmanResult, half_life, kalman_hedge_ratio
from .universe import NYSE_SECTORS, sector_of


@dataclass
class PairResult:
    """Screening result for a single (y, x) pair."""

    y: str
    x: str
    sector: str | None
    pvalue: float
    correlation: float
    beta_last: float                       # latest Kalman hedge ratio
    half_life_bars: float                  # mean-reversion half-life of spread
    zscore_last: float                     # latest spread z-score
    kalman: KalmanResult = field(repr=False)

    @property
    def label(self) -> str:
        return f"{self.y} / {self.x}"


def candidate_pairs(
    tickers: list[str],
    within_sector_only: bool = True,
) -> list[tuple[str, str]]:
    """Enumerate unordered candidate pairs from ``tickers``.

    With ``within_sector_only`` (the default) only pairs whose members share a
    universe sector are returned, which is both faster and more meaningful. The
    ``tickers`` argument bounds the result to symbols actually present in the
    loaded price data.
    """
    present = set(tickers)
    if not within_sector_only:
        return list(itertools.combinations(sorted(present), 2))

    pairs: list[tuple[str, str]] = []
    for members in NYSE_SECTORS.values():
        usable = [t for t in members if t in present]
        pairs.extend(itertools.combinations(sorted(usable), 2))
    return pairs


def screen_pairs(
    prices: pd.DataFrame,
    within_sector_only: bool = True,
    max_pvalue: float = 0.05,
    z_window: int = 60,
    kalman_delta: float = 1e-4,
    kalman_obs_cov: float = 1e-3,
) -> list[PairResult]:
    """Run cointegration screening over all candidate pairs in ``prices``.

    For every candidate pair we run the Engle-Granger test, and for pairs that
    pass ``max_pvalue`` we additionally fit the Kalman dynamic hedge ratio and
    compute the smoothed-spread half-life and latest z-score. Results are
    returned sorted by ascending p-value (best cointegration first).
    """
    tickers = list(prices.columns)
    results: list[PairResult] = []

    for a, b in candidate_pairs(tickers, within_sector_only=within_sector_only):
        s_a, s_b = prices[a], prices[b]
        joined = pd.concat([s_a, s_b], axis=1).dropna()
        if len(joined) < 60:
            continue
        ya, xb = joined[a], joined[b]

        try:
            # coint() returns (t-stat, pvalue, crit values).
            _, pvalue, _ = coint(ya, xb)
        except Exception:
            continue
        if not np.isfinite(pvalue) or pvalue > max_pvalue:
            continue

        corr = float(np.corrcoef(ya.to_numpy(), xb.to_numpy())[0, 1])

        try:
            kf = kalman_hedge_ratio(
                ya, xb,
                delta=kalman_delta,
                obs_cov=kalman_obs_cov,
                z_window=z_window,
            )
        except Exception:
            continue

        results.append(
            PairResult(
                y=a,
                x=b,
                sector=sector_of(a),
                pvalue=float(pvalue),
                correlation=corr,
                beta_last=float(kf.beta.iloc[-1]),
                half_life_bars=half_life(kf.spread),
                zscore_last=float(kf.zscore.iloc[-1]) if kf.zscore.notna().any() else float("nan"),
                kalman=kf,
            )
        )

    results.sort(key=lambda r: r.pvalue)
    return results


def top_pairs(results: list[PairResult], n: int = 10) -> list[PairResult]:
    """Return the ``n`` most strongly cointegrated pairs."""
    return results[:n]
