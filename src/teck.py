"""Teck Resources data acquisition and predictor analysis.

Teck Resources (NYSE/TSX: ``TECK``) is a diversified miner — copper, zinc and,
historically, steelmaking coal — so its equity is driven far more by commodity
and macro factors than by company-idiosyncratic news. This module web-scrapes
the raw material for a Teck dashboard and quantifies those drivers:

* **Prices** are scraped from Stooq's CSV endpoint (a scrape-friendly source
  that, unlike Yahoo, rarely rate-limits), with a yfinance fallback.
* **Predictors** are a basket of liquid, US-listed proxy ETFs/indices for the
  factors that move Teck — copper miners, broad metals & mining, energy, gold,
  the US dollar, Canadian equities and the S&P 500 — scraped the same way.
* **News headlines** are scraped from the Google News RSS feed.
* **Key statistics** are computed from the price history and augmented with a
  best-effort scrape of Stooq's quote page.

The predictor analysis regresses Teck's daily returns on the predictor returns
(ordinary least squares on standardised factors) to show which drivers explain
its moves, alongside pairwise correlations and a rolling copper correlation.

Nothing is hard-coded beyond the choice of predictor proxies; all values are
fetched live.
"""

from __future__ import annotations

import io
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Teck's US listing (NYSE). Stooq uses the ``.us`` suffix for US-listed names.
TECK_SYMBOL = "teck.us"
TECK_NAME = "Teck Resources Ltd. (Class B)"

# Factor proxies for the drivers of a diversified base-metals miner. Using
# liquid US-listed ETFs keeps the data clean and scrape-friendly.
DEFAULT_PREDICTORS: dict[str, str] = {
    "Copper miners (COPX)": "copx.us",
    "Metals & mining (XME)": "xme.us",
    "Energy (XLE)": "xle.us",
    "Gold (GLD)": "gld.us",
    "US dollar (UUP)": "uup.us",
    "Canada equities (EWC)": "ewc.us",
    "S&P 500 (SPY)": "spy.us",
}

_STOOQ_CSV = "https://stooq.com/q/d/l/?s={symbol}&i=d"
_STOOQ_QUOTE = "https://stooq.com/q/?s={symbol}"
_GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
_UA = "Mozilla/5.0 (compatible; TeamTracy/1.0; +https://github.com/teamtracy2026)"


# --------------------------------------------------------------------------- #
# Low-level HTTP
# --------------------------------------------------------------------------- #
def _http_get(url: str, timeout: int = 25, retries: int = 3) -> str:
    """GET a URL as text, retrying with backoff. Honours HTTP(S)_PROXY env."""
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as exc:  # pragma: no cover - network dependent
            last_exc = exc
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"GET failed for {url}: {last_exc}")


# --------------------------------------------------------------------------- #
# Prices (Stooq scrape, yfinance fallback)
# --------------------------------------------------------------------------- #
def _parse_stooq_csv(text: str) -> pd.DataFrame:
    """Parse a Stooq daily CSV into an OHLCV frame indexed by date."""
    df = pd.read_csv(io.StringIO(text))
    if "Date" not in df.columns or "Close" not in df.columns:
        raise ValueError("Unexpected Stooq CSV format.")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    # Standardise column names to Title case (Open/High/Low/Close/Volume).
    df.columns = [c.capitalize() for c in df.columns]
    return df


def fetch_stooq_ohlcv(symbol: str) -> pd.DataFrame:
    """Scrape daily OHLCV history for a Stooq symbol (e.g. ``teck.us``)."""
    text = _http_get(_STOOQ_CSV.format(symbol=symbol))
    if text.strip().lower().startswith("<") or "N/D" in text[:64]:
        raise ValueError(f"Stooq returned no data for {symbol}.")
    return _parse_stooq_csv(text)


def _fetch_close_yf(symbol: str, period: str) -> pd.Series:
    """Fallback: fetch a close series via yfinance (Stooq -> Yahoo ticker map)."""
    import yfinance as yf  # local import; optional dependency

    yticker = symbol.replace(".us", "").upper()
    raw = yf.download(yticker, period=period, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise ValueError(f"yfinance returned no data for {yticker}.")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.rename(symbol)


def fetch_teck_history(period_years: int = 5) -> pd.DataFrame:
    """Teck OHLCV history, scraped from Stooq with a yfinance fallback."""
    try:
        df = fetch_stooq_ohlcv(TECK_SYMBOL)
    except Exception:
        close = _fetch_close_yf(TECK_SYMBOL, f"{period_years}y")
        df = close.to_frame("Close")
    cutoff = df.index.max() - pd.DateOffset(years=period_years)
    return df.loc[df.index >= cutoff]


def fetch_predictor_closes(
    predictors: dict[str, str] | None = None,
    period_years: int = 5,
) -> pd.DataFrame:
    """Scrape close-price series for each predictor proxy into one frame.

    Columns are the human-readable predictor labels. Series that fail to fetch
    are skipped rather than aborting the whole dashboard.
    """
    predictors = predictors or DEFAULT_PREDICTORS
    series: dict[str, pd.Series] = {}
    for label, symbol in predictors.items():
        try:
            close = fetch_stooq_ohlcv(symbol)["Close"]
        except Exception:
            try:
                close = _fetch_close_yf(symbol, f"{period_years}y")
            except Exception:
                continue
        series[label] = close
    if not series:
        return pd.DataFrame()
    df = pd.DataFrame(series).sort_index()
    cutoff = df.index.max() - pd.DateOffset(years=period_years)
    return df.loc[df.index >= cutoff].ffill()


# --------------------------------------------------------------------------- #
# Key statistics (computed + scraped)
# --------------------------------------------------------------------------- #
def compute_performance_stats(ohlcv: pd.DataFrame) -> dict[str, str]:
    """Performance metrics computed directly from the price history."""
    close = ohlcv["Close"].dropna()
    if close.empty:
        return {}
    last = float(close.iloc[-1])
    rets = close.pct_change().dropna()

    def _ret_over(days: int) -> float | None:
        if len(close) <= days:
            return None
        return last / float(close.iloc[-days - 1]) - 1.0

    def _since(offset: pd.Timestamp) -> float | None:
        window = close.loc[close.index >= offset]
        if window.empty:
            return None
        return last / float(window.iloc[0]) - 1.0

    ytd = _since(pd.Timestamp(close.index[-1].year, 1, 1))
    hi_52 = float(close.loc[close.index >= close.index[-1] - pd.DateOffset(weeks=52)].max())
    lo_52 = float(close.loc[close.index >= close.index[-1] - pd.DateOffset(weeks=52)].min())
    vol_ann = float(rets.std(ddof=0) * np.sqrt(252))

    def pct(v: float | None) -> str:
        return "—" if v is None else f"{v * 100:+.1f}%"

    return {
        "Last close": f"${last:,.2f}",
        "1-month": pct(_ret_over(21)),
        "3-month": pct(_ret_over(63)),
        "YTD": pct(ytd),
        "1-year": pct(_ret_over(252)),
        "52-week range": f"${lo_52:,.2f} – ${hi_52:,.2f}",
        "Annualised volatility": f"{vol_ann * 100:.1f}%",
    }


def scrape_quote_snapshot(symbol: str = TECK_SYMBOL) -> dict[str, str]:
    """Best-effort scrape of Stooq's quote page for a few live fields.

    Stooq's markup is terse and changes occasionally, so this is wrapped to
    degrade gracefully — the dashboard's core stats come from the price history.
    """
    from bs4 import BeautifulSoup

    out: dict[str, str] = {}
    try:
        html = _http_get(_STOOQ_QUOTE.format(symbol=symbol))
        soup = BeautifulSoup(html, "lxml")
        # Stooq exposes the last price in an element id like "aq_<sym>_c2".
        base = symbol.split(".")[0]
        node = soup.find(id=f"aq_{base}_c2")
        if node and node.text.strip():
            out["Quote (scraped)"] = node.text.strip()
    except Exception:
        pass
    return out


# --------------------------------------------------------------------------- #
# News (Google News RSS scrape)
# --------------------------------------------------------------------------- #
@dataclass
class NewsItem:
    title: str
    link: str
    published: str
    source: str


def scrape_news(query: str = "Teck Resources stock", limit: int = 10) -> list[NewsItem]:
    """Scrape recent headlines for ``query`` from the Google News RSS feed."""
    import xml.etree.ElementTree as ET

    url = _GOOGLE_NEWS_RSS.format(query=urllib.parse.quote(query))
    try:
        xml = _http_get(url)
        root = ET.fromstring(xml)
    except Exception:
        return []

    items: list[NewsItem] = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        source_el = item.find("source")
        source = (source_el.text or "").strip() if source_el is not None else ""
        if title:
            items.append(NewsItem(title=title, link=link, published=pub, source=source))
        if len(items) >= limit:
            break
    return items


# --------------------------------------------------------------------------- #
# Predictor analysis
# --------------------------------------------------------------------------- #
@dataclass
class PredictorAnalysis:
    correlations: pd.Series          # corr of Teck daily returns vs each predictor
    betas: pd.Series                 # standardised OLS coefficients
    r_squared: float                 # fraction of Teck return variance explained
    rolling_corr: pd.Series = field(repr=False)  # rolling corr with top predictor
    top_predictor: str = ""


def predictor_analysis(
    teck_close: pd.Series,
    predictor_closes: pd.DataFrame,
    roll_window: int = 60,
) -> PredictorAnalysis:
    """Quantify how predictor factors explain Teck's returns.

    Computes pairwise return correlations, a multivariate OLS of Teck returns on
    standardised predictor returns (so coefficients are comparable in size), the
    regression R², and a rolling correlation with the single most correlated
    predictor.
    """
    teck_ret = teck_close.pct_change().rename("TECK")
    pred_ret = predictor_closes.pct_change()
    data = pd.concat([teck_ret, pred_ret], axis=1).dropna()
    if len(data) < 30 or data.shape[1] < 2:
        raise ValueError("Not enough overlapping data for predictor analysis.")

    y = data["TECK"]
    X = data.drop(columns=["TECK"])

    correlations = X.apply(lambda col: col.corr(y)).sort_values(ascending=False)

    # Standardise predictors so OLS betas are comparable across factors.
    Xz = (X - X.mean()) / X.std(ddof=0)
    Xz = Xz.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    A = np.column_stack([np.ones(len(Xz)), Xz.to_numpy()])
    coef, *_ = np.linalg.lstsq(A, y.to_numpy(), rcond=None)
    betas = pd.Series(coef[1:], index=X.columns).sort_values(key=np.abs, ascending=False)

    resid = y.to_numpy() - A @ coef
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y.to_numpy() - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    top = correlations.index[0]
    rolling = X[top].rolling(roll_window).corr(y).rename(f"Rolling corr vs {top}")

    return PredictorAnalysis(
        correlations=correlations,
        betas=betas,
        r_squared=float(r2),
        rolling_corr=rolling,
        top_predictor=top,
    )
