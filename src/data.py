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
# Tickers per yfinance batch download. Smaller batches are friendlier to
# Yahoo's rate limiter on shared cloud IPs.
BATCH_SIZE = int(os.environ.get("TEAMTRACY_BATCH_SIZE", "100"))


def _chunks(seq: list[str], size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


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

    # Download in batches: a single yf.download call with thousands of symbols
    # is fragile (partial failures, huge memory spikes). Batching keeps each
    # request reasonable and lets us tolerate per-batch failures. Yahoo also
    # rate-limits shared cloud IPs (e.g. Streamlit Cloud), so each batch is
    # retried with backoff and a single-threaded fallback.
    frames: list[pd.DataFrame] = []
    for batch in _chunks(tickers, BATCH_SIZE):
        close = _download_batch(batch, period, interval)
        if not close.empty:
            frames.append(close)

    if not frames:
        return pd.DataFrame()

    prices = pd.concat(frames, axis=1)
    prices = _clean(prices, min_obs=min_obs)

    if use_cache and not prices.empty:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            prices.to_parquet(path)
        except Exception:
            pass  # Caching is best-effort; never fail a load over it.

    return prices


def _download_batch(
    batch: list[str], period: str, interval: str, retries: int = 3
) -> pd.DataFrame:
    """Download one batch of tickers, retrying on transient/rate-limit failures.

    Yahoo frequently returns HTTP 429 ("Too Many Requests") to data-center IPs.
    We retry with exponential backoff and, on the final attempt, drop threading
    (single-threaded requests are gentler and sometimes get through when the
    threaded path is throttled).
    """
    for attempt in range(retries):
        threaded = attempt < retries - 1
        try:
            raw = yf.download(
                batch,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=threaded,
            )
            close = _extract_close(raw, batch)
            # Keep only columns that actually came back with data.
            close = close.dropna(axis=1, how="all")
            if not close.empty:
                return close
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"[data] batch download failed (attempt {attempt + 1}): {exc}")
        time.sleep(1.5 * (attempt + 1))
    print(f"[data] giving up on batch of {len(batch)} tickers after {retries} tries")
    return pd.DataFrame()


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
    # Drop duplicate columns that can arise from concatenating batches.
    prices = prices.loc[:, ~prices.columns.duplicated()]
    # Forward-fill isolated gaps (holidays/halts), then require enough history.
    prices = prices.ffill()
    enough = prices.notna().sum() >= min_obs
    prices = prices.loc[:, enough[enough].index]
    # Trim leading rows where some surviving ticker is still NaN.
    prices = prices.dropna(how="any")
    return prices


_INFO_CACHE = CACHE_DIR / "company_info.json"


def get_company_info(tickers: list[str]) -> dict[str, dict[str, str]]:
    """Fetch company name and sector for ``tickers`` from Yahoo Finance.

    Returns ``{ticker: {"name": ..., "sector": ...}}``. Results are cached on
    disk (keyed by ticker) so we only hit Yahoo's metadata endpoint once per
    name — this is meant for the handful of tickers actually shown on screen,
    not the whole universe. Missing fields fall back to the ticker itself.
    """
    cache = _load_info_cache()
    missing = [t for t in tickers if t not in cache]

    if missing and yf is not None:
        for t in missing:
            name, sector = t, "—"
            try:
                info = yf.Ticker(t).get_info()
                name = info.get("shortName") or info.get("longName") or t
                sector = info.get("sector") or "—"
            except Exception:
                pass  # Network/parse failure -> fall back to ticker.
            cache[t] = {"name": str(name), "sector": str(sector)}
        _save_info_cache(cache)

    return {t: cache.get(t, {"name": t, "sector": "—"}) for t in tickers}


def _load_info_cache() -> dict[str, dict[str, str]]:
    if not _INFO_CACHE.exists():
        return {}
    try:
        import json

        return json.loads(_INFO_CACHE.read_text())
    except Exception:
        return {}


def _save_info_cache(cache: dict[str, dict[str, str]]) -> None:
    try:
        import json

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _INFO_CACHE.write_text(json.dumps(cache))
    except Exception:
        pass
