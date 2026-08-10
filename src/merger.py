"""Anglo American / Teck merger analysis and forward stock projection.

In September 2025 Anglo American and Teck Resources announced a merger of equals
to form **Anglo Teck**, a copper-focused major. This module layers that event
onto the Teck dashboard in two ways:

1. **Merger analysis** — pulls Anglo American's (US ADR) price alongside Teck's,
   measures how tightly the two now co-move (rolling correlation since the
   announcement), and computes the **merger-arbitrage spread**: the gap between
   Teck's market price and the deal-implied value (Anglo price × exchange
   ratio). That spread encodes the market's view on completion odds and timing.

2. **Forward projection** — a Monte Carlo (geometric Brownian motion) simulation
   of Teck's price over a chosen horizon, then a **scenario blend** that mixes a
   standalone path against a deal-completion path (Teck converging to the
   Anglo-linked deal value), weighted by an adjustable completion probability.

Deal terms are genuinely uncertain and forward-looking, so they are exposed as
**adjustable assumptions** (defaults reflect the announced deal) rather than
hard-coded facts — always verify against official filings.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .teck import _fetch_close_yf, fetch_stooq_ohlcv

# Anglo American US ADR (USD-denominated, so it compares cleanly with Teck's
# NYSE price without an FX leg). Stooq symbol for the ADR.
ANGLO_SYMBOL = "ngloy.us"
ANGLO_NAME = "Anglo American plc (US ADR)"

# Announced 9 September 2025. Used only as a default UI value.
MERGER_ANNOUNCED = date(2025, 9, 9)


@dataclass
class MergerTerms:
    """Adjustable, forward-looking assumptions about the Anglo-Teck deal.

    ``exchange_ratio`` is expressed in **Anglo ADRs received per Teck share** so
    that the deal-implied value (ratio × ADR price) is directly comparable with
    Teck's USD price. Tune it to match the ADR ratio in the official terms.
    """

    exchange_ratio: float = 1.3301          # Anglo ADR-equivalent per Teck share
    completion_prob: float = 0.80           # assumed probability the deal closes
    announced: date = MERGER_ANNOUNCED
    horizon_days: int = 252                 # projection horizon (~1 year)


def fetch_anglo_history(period_years: int = 5) -> pd.DataFrame:
    """Anglo American ADR OHLCV, scraped from Stooq with a yfinance fallback."""
    try:
        df = fetch_stooq_ohlcv(ANGLO_SYMBOL)
    except Exception:
        close = _fetch_close_yf(ANGLO_SYMBOL, f"{period_years}y")
        df = close.to_frame("Close")
    cutoff = df.index.max() - pd.DateOffset(years=period_years)
    return df.loc[df.index >= cutoff]


# --------------------------------------------------------------------------- #
# Merger-arbitrage spread
# --------------------------------------------------------------------------- #
def merger_spread(
    teck_close: pd.Series,
    anglo_close: pd.Series,
    exchange_ratio: float,
    since: date | None = None,
) -> pd.DataFrame:
    """Deal-implied Teck value and arbitrage spread over time.

    Returns a frame indexed by date with the Teck price, the deal-implied value
    (``exchange_ratio × anglo``) and the spread as a percentage of the implied
    value: ``(teck / implied) − 1``. A negative spread means Teck trades below
    the deal value — the usual merger-arb setup, where the discount compensates
    for deal risk and time to close.
    """
    df = pd.concat(
        [teck_close.rename("teck"), anglo_close.rename("anglo")], axis=1
    ).dropna()
    if since is not None:
        df = df.loc[df.index >= pd.Timestamp(since)]
    df["implied"] = exchange_ratio * df["anglo"]
    df["spread_pct"] = df["teck"] / df["implied"] - 1.0
    return df


def spread_summary(spread_df: pd.DataFrame) -> dict[str, float]:
    """Latest arbitrage metrics from a :func:`merger_spread` frame."""
    if spread_df.empty:
        return {}
    last = spread_df.iloc[-1]
    teck, implied = float(last["teck"]), float(last["implied"])
    # Upside if Teck converges to the implied deal value.
    upside = implied / teck - 1.0
    return {
        "teck": teck,
        "implied": implied,
        "spread_pct": float(last["spread_pct"]),
        "upside_to_deal": upside,
    }


def annualized_arb_return(upside_to_deal: float, days_to_close: int) -> float:
    """Annualise the convergence upside over the expected days to close."""
    if days_to_close <= 0:
        return float("nan")
    years = days_to_close / 365.25
    return (1.0 + upside_to_deal) ** (1.0 / years) - 1.0 if upside_to_deal > -1 else float("nan")


# --------------------------------------------------------------------------- #
# Monte Carlo projection
# --------------------------------------------------------------------------- #
@dataclass
class Projection:
    """Result of a Monte Carlo price projection."""

    dates: pd.DatetimeIndex           # future business days
    bands: pd.DataFrame               # percentile bands (p05..p95) per date
    s0: float                         # starting price
    terminal: np.ndarray              # terminal-price distribution across paths
    prob_gain: float                  # P(terminal > s0)
    exp_price: float                  # mean terminal price
    drift_daily: float
    vol_daily: float


def _daily_params(close: pd.Series, drift_mode: str) -> tuple[float, float]:
    logret = np.log(close / close.shift(1)).dropna()
    sigma = float(logret.std(ddof=0))
    if drift_mode == "historical":
        mu = float(logret.mean())
    else:  # "random_walk" — zero-drift martingale, the conservative default
        mu = 0.0
    return mu, sigma


def _future_bdays(last: pd.Timestamp, n: int) -> pd.DatetimeIndex:
    return pd.bdate_range(last + pd.tseries.offsets.BDay(1), periods=n)


def project_gbm(
    close: pd.Series,
    horizon_days: int = 252,
    n_paths: int = 5000,
    drift_mode: str = "random_walk",
    seed: int = 42,
) -> Projection:
    """Project a price series forward with geometric Brownian motion.

    Daily log-return volatility is estimated from history; drift is either the
    historical mean (``drift_mode="historical"``) or zero (``"random_walk"``,
    the default — a driftless random walk avoids over-extrapolating past
    returns). Returns percentile bands and the terminal-price distribution.
    """
    close = close.dropna()
    if len(close) < 30:
        raise ValueError("Need at least 30 observations to estimate volatility.")
    mu, sigma = _daily_params(close, drift_mode)
    s0 = float(close.iloc[-1])

    rng = np.random.default_rng(seed)
    shocks = rng.normal(mu, sigma, size=(n_paths, horizon_days))
    log_paths = np.cumsum(shocks, axis=1)
    price_paths = s0 * np.exp(log_paths)      # shape (n_paths, horizon_days)

    qs = [5, 25, 50, 75, 95]
    band_vals = np.percentile(price_paths, qs, axis=0)  # (5, horizon_days)
    dates = _future_bdays(close.index[-1], horizon_days)
    bands = pd.DataFrame(
        {f"p{q:02d}": band_vals[i] for i, q in enumerate(qs)}, index=dates
    )

    terminal = price_paths[:, -1]
    return Projection(
        dates=dates,
        bands=bands,
        s0=s0,
        terminal=terminal,
        prob_gain=float(np.mean(terminal > s0)),
        exp_price=float(np.mean(terminal)),
        drift_daily=mu,
        vol_daily=sigma,
    )


@dataclass
class ScenarioForecast:
    """Merger-aware projection blending standalone and deal-completion paths."""

    standalone: Projection            # Teck as an independent stock
    deal_median: pd.Series            # deal-implied value path (ratio × Anglo median)
    deal_value_now: float             # current deal-implied value
    completion_prob: float
    blended_exp_price: float          # probability-weighted expected terminal price
    blended_prob_gain: float          # probability-weighted P(gain)
    implied_upside: float             # convergence upside from today's price


def scenario_forecast(
    teck_close: pd.Series,
    anglo_close: pd.Series,
    terms: MergerTerms,
    n_paths: int = 5000,
    drift_mode: str = "random_walk",
    seed: int = 42,
) -> ScenarioForecast:
    """Project Teck forward under a blend of standalone and deal scenarios.

    * **Standalone**: Teck simulated as an independent stock (GBM).
    * **Deal completes**: Teck converges to the Anglo-linked deal value; we
      simulate Anglo forward and scale by the exchange ratio.

    Terminal outcomes are mixed with weight ``completion_prob`` on the deal
    branch, giving a probability-weighted expected price and probability of gain.
    """
    standalone = project_gbm(
        teck_close, terms.horizon_days, n_paths, drift_mode, seed
    )
    anglo_proj = project_gbm(
        anglo_close, terms.horizon_days, n_paths, drift_mode, seed + 1
    )

    ratio = terms.exchange_ratio
    deal_median = (anglo_proj.bands["p50"] * ratio).rename("Deal-implied (median)")
    deal_value_now = ratio * float(anglo_close.dropna().iloc[-1])

    deal_terminal = anglo_proj.terminal * ratio
    p = float(np.clip(terms.completion_prob, 0.0, 1.0))

    # Mixture of the two terminal distributions.
    n = len(standalone.terminal)
    rng = np.random.default_rng(seed + 2)
    take_deal = rng.random(n) < p
    blended_terminal = np.where(take_deal, deal_terminal, standalone.terminal)

    s0 = standalone.s0
    return ScenarioForecast(
        standalone=standalone,
        deal_median=deal_median,
        deal_value_now=deal_value_now,
        completion_prob=p,
        blended_exp_price=float(np.mean(blended_terminal)),
        blended_prob_gain=float(np.mean(blended_terminal > s0)),
        implied_upside=deal_value_now / s0 - 1.0,
    )
