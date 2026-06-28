"""Pairwise cointegration screening over a price panel.

Ranks candidate pairs by the Engle-Granger cointegration test p-value: the
lower the p-value, the stronger the evidence that a stationary linear
combination of the two price series exists.

Running the test on every pair of an exchange-sized universe is O(N^2) tests,
which is far too slow. Instead we pre-filter cheaply: compute the correlation
of daily returns (a fast vectorised matrix operation) and only run the
cointegration test on the most correlated candidate pairs. Highly correlated
co-movement is a necessary precondition for a tradable cointegrated spread, so
this prunes the search dramatically while keeping the real hits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from .kalman import KalmanResult, half_life, kalman_hedge_ratio


@dataclass
class PairResult:
    """Screening result for a single (y, x) pair."""

    y: str
    x: str
    pvalue: float
    correlation: float
    beta_last: float                       # latest Kalman hedge ratio
    half_life_bars: float                  # mean-reversion half-life of spread
    zscore_last: float                     # latest spread z-score
    kalman: KalmanResult = field(repr=False)

    @property
    def label(self) -> str:
        return f"{self.y} / {self.x}"


def correlation_candidates(
    prices: pd.DataFrame,
    min_correlation: float = 0.8,
    max_candidates: int = 1000,
) -> list[tuple[str, str]]:
    """Pick the most correlated pairs to feed into the cointegration test.

    Correlation is computed on daily returns. Pairs with absolute correlation
    below ``min_correlation`` are discarded; the remainder are ranked by
    descending absolute correlation and truncated to ``max_candidates`` to bound
    the (slower) cointegration sweep that follows.
    """
    rets = prices.pct_change().dropna(how="all")
    if rets.shape[1] < 2:
        return []

    corr = rets.corr()
    cols = list(corr.columns)
    # Upper triangle only (unordered pairs), skip the diagonal.
    mat = corr.to_numpy()
    iu, ju = np.triu_indices(len(cols), k=1)
    vals = mat[iu, ju]

    candidates: list[tuple[float, str, str]] = []
    for k in range(len(vals)):
        c = vals[k]
        if np.isfinite(c) and abs(c) >= min_correlation:
            a, b = cols[iu[k]], cols[ju[k]]
            candidates.append((abs(c), a, b))

    candidates.sort(key=lambda t: t[0], reverse=True)
    return [(a, b) for _, a, b in candidates[:max_candidates]]


def screen_pairs(
    prices: pd.DataFrame,
    min_correlation: float = 0.8,
    max_candidates: int = 1000,
    max_pvalue: float = 0.05,
    z_window: int = 60,
    kalman_delta: float = 1e-4,
    kalman_obs_cov: float = 1e-3,
) -> list[PairResult]:
    """Screen ``prices`` for cointegrated pairs.

    A correlation pre-filter selects candidate pairs (see
    :func:`correlation_candidates`); each candidate is run through the
    Engle-Granger test, and pairs that pass ``max_pvalue`` get the Kalman
    dynamic hedge ratio, smoothed-spread half-life and latest z-score. Results
    are returned sorted by ascending p-value (best cointegration first).
    """
    if prices.shape[1] < 2:
        return []

    candidates = correlation_candidates(
        prices, min_correlation=min_correlation, max_candidates=max_candidates
    )

    results: list[PairResult] = []
    for a, b in candidates:
        joined = pd.concat([prices[a], prices[b]], axis=1).dropna()
        if len(joined) < 60:
            continue
        ya, xb = joined.iloc[:, 0], joined.iloc[:, 1]

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
