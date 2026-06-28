"""Price data loading from Yahoo Finance.

Thin wrapper around :mod:`yfinance` that downloads adjusted close prices for a
basket of tickers, aligns them on a common calendar, and drops names with
insufficient history. The on-disk cache keeps the dashboard responsive across
reruns and avoids hammering Yahoo on every interaction.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd

try:  # yfinance is an optional import so the module can be imported in tests.
    import yfinance as yf
except Exception:  # pragma: no cover - exercised only when yfinance is absent.
    yf = None

CACHE_DIR = Path(os.environ.get("TEAMTRACY_CACHE_DIR", ".cache"))
# How long a cached download stays fresh. Daily bars don't change intraday, so
# a few hours is plenty and keeps reruns instant.
CACHE_TTL_SECONDS = int(os.environ.get("TEAMTRACY_CACHE_TTL", str(6 * 3600)))


def _cache_path(tickers: list[str], period: str, interval: str) -> Path:
    key = f"{'-'.join(sorted(tickers))}_{period}_{interval}"
    # Hash to keep filenames short and filesystem-safe for large baskets.
    import hashlib

    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    return CACHE_DIR / f"prices_{digest}.parquet"


def _is_fresh(path: Path) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < CACHE_TTL_SECONDS


def load_prices(
    tickers: list[str],
    period: str = "2y",
    interval: str = "1d",
    use_cache: bool = True,
    min_obs: int = 252,
) -> pd.DataFrame:
    """Download adjusted close prices for ``tickers``.

    Parameters
    ----------
    tickers:
        Symbols to download.
    period:
        yfinance period string (e.g. ``"1y"``, ``"2y"``, ``"5y"``).
    interval:
        Bar size; ``"1d"`` for daily.
    use_cache:
        When true, reuse a recent on-disk download if available.
    min_obs:
        Columns (tickers) with fewer than this many non-NaN observations are
        dropped — cointegration on a stub of history is meaningless.

    Returns
    -------
    DataFrame indexed by date, one column per surviving ticker, forward-filled
    over isolated gaps and trimmed to the common date range.
    """
    tickers = sorted(set(tickers))
    if not tickers:
        return pd.DataFrame()

    path = _cache_path(tickers, period, interval)
    if use_cache and _is_fresh(path):
        try:
            return pd.read_parquet(path)
        except Exception:
            pass  # Corrupt cache — fall through and re-download.

    if yf is None:
        raise RuntimeError(
            "yfinance is not installed. Run `pip install -r requirements.txt`."
        )

    raw = yf.download(
        tickers,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    prices = _extract_close(raw, tickers)
    prices = _clean(prices, min_obs=min_obs)

    if use_cache and not prices.empty:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            prices.to_parquet(path)
        except Exception:
            pass  # Caching is best-effort; never fail a load over it.

    return prices


def _extract_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Pull the close price frame out of yfinance's multi-shape output."""
    if raw is None or len(raw) == 0:
        return pd.DataFrame()

    # Multi-ticker downloads come back with a column MultiIndex
    # (field, ticker); single-ticker downloads are flat.
    if isinstance(raw.columns, pd.MultiIndex):
        field = "Close" if "Close" in raw.columns.get_level_values(0) else None
        if field is None:
            return pd.DataFrame()
        close = raw[field].copy()
    else:
        col = "Close" if "Close" in raw.columns else raw.columns[0]
        name = tickers[0] if len(tickers) == 1 else col
        close = raw[[col]].copy()
        close.columns = [name]

    return close


def _clean(prices: pd.DataFrame, min_obs: int) -> pd.DataFrame:
    if prices.empty:
        return prices
    prices = prices.sort_index()
    # Forward-fill isolated gaps (holidays/halts), then require enough history.
    prices = prices.ffill()
    enough = prices.notna().sum() >= min_obs
    prices = prices.loc[:, enough[enough].index]
    # Trim leading rows where some surviving ticker is still NaN.
    prices = prices.dropna(how="any")
    return prices
