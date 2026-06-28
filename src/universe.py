"""Dynamic NYSE equity universe.

The full list of NYSE-listed symbols is fetched at runtime from the official
NASDAQ Trader symbol directory (``otherlisted.txt``), which is the canonical
free source for non-NASDAQ listings. Nothing here is hard-coded — the universe
reflects whatever is currently listed.

We keep only NYSE common stock: rows whose Exchange code is ``N`` (NYSE),
excluding ETFs, test issues, and non-common securities (warrants, units,
preferreds, rights), which we drop via simple symbol heuristics.
"""

from __future__ import annotations

import io
import os
import time
import urllib.request
from pathlib import Path

import pandas as pd

# Official NASDAQ Trader symbol directory. "otherlisted" covers NYSE, NYSE
# American, NYSE Arca, etc.; we filter to NYSE proper below.
OTHERLISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

CACHE_DIR = Path(os.environ.get("TEAMTRACY_CACHE_DIR", ".cache"))
_UNIVERSE_CACHE = CACHE_DIR / "nyse_universe.csv"
# The listing changes slowly; refresh at most once a day.
_UNIVERSE_TTL = int(os.environ.get("TEAMTRACY_UNIVERSE_TTL", str(24 * 3600)))


def _is_common_stock_symbol(symbol: str) -> bool:
    """Heuristic: keep plain common-stock tickers, drop derivative securities.

    Warrants, units, preferreds and rights carry punctuation in the ACT symbol
    (``$``, ``.``, ``+``, ``=``, ``#``...). Plain common stock is alphabetic and
    at most five characters. This errs toward dropping edge cases rather than
    polluting the screen with non-equity instruments.
    """
    return bool(symbol) and symbol.isalpha() and 1 <= len(symbol) <= 5


def _fetch_otherlisted() -> pd.DataFrame:
    """Download and parse the NASDAQ Trader otherlisted directory."""
    # urllib honours HTTPS_PROXY/HTTP_PROXY from the environment.
    with urllib.request.urlopen(OTHERLISTED_URL, timeout=30) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    # The file is pipe-delimited with a trailing "File Creation Time" footer row.
    df = pd.read_csv(io.StringIO(raw), sep="|")
    if "ACT Symbol" not in df.columns:
        raise RuntimeError("Unexpected otherlisted.txt format from NASDAQ Trader.")
    df = df[~df["ACT Symbol"].astype(str).str.startswith("File Creation Time")]
    return df


def nyse_tickers(limit: int | None = None, use_cache: bool = True) -> list[str]:
    """Return the list of NYSE common-stock tickers, fetched live.

    Parameters
    ----------
    limit:
        If given, return only the first ``limit`` tickers (alphabetical). Useful
        to bound an exploratory scan; ``None`` returns the entire NYSE.
    use_cache:
        Reuse a recent on-disk copy of the directory if available.

    Raises
    ------
    RuntimeError if the directory cannot be fetched and no cache exists — we do
    not fall back to a hard-coded list.
    """
    tickers = _load_cached_universe() if use_cache else None
    if tickers is None:
        df = _fetch_otherlisted()
        nyse = df[df["Exchange"].astype(str).str.upper() == "N"].copy()
        if "ETF" in nyse.columns:
            nyse = nyse[nyse["ETF"].astype(str).str.upper() != "Y"]
        if "Test Issue" in nyse.columns:
            nyse = nyse[nyse["Test Issue"].astype(str).str.upper() != "Y"]
        symbols = (
            nyse["ACT Symbol"].astype(str).str.strip().str.upper().tolist()
        )
        tickers = sorted({s for s in symbols if _is_common_stock_symbol(s)})
        _save_cached_universe(tickers)

    if limit is not None:
        return tickers[:limit]
    return tickers


def _load_cached_universe() -> list[str] | None:
    if not _UNIVERSE_CACHE.exists():
        return None
    if time.time() - _UNIVERSE_CACHE.stat().st_mtime >= _UNIVERSE_TTL:
        return None
    try:
        return pd.read_csv(_UNIVERSE_CACHE)["ticker"].astype(str).tolist()
    except Exception:
        return None


def _save_cached_universe(tickers: list[str]) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"ticker": tickers}).to_csv(_UNIVERSE_CACHE, index=False)
    except Exception:
        pass  # Caching is best-effort.
