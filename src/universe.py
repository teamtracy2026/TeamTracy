"""NYSE equity universe used for pair screening.

The full NYSE listing runs to a couple thousand names, which makes an
all-pairs cointegration sweep both slow and statistically noisy. Instead we
work from a curated set of liquid, NYSE-listed large caps grouped by sector.
Screening within a sector keeps the combinatorics manageable and tends to
surface economically meaningful relationships (two refiners, two railroads,
two money-center banks) rather than spurious ones.

Every ticker below is listed on the NYSE (not NASDAQ). If you extend the list,
keep that invariant — mixing in NASDAQ names is fine mechanically but the
"NYSE" label on the dashboard would no longer be accurate.
"""

from __future__ import annotations

# Sector -> list of NYSE tickers. Kept deliberately liquid so Yahoo Finance
# returns clean, gap-free history for the lookback windows we use.
NYSE_SECTORS: dict[str, list[str]] = {
    "Money-center & regional banks": [
        "JPM", "BAC", "WFC", "C", "USB", "PNC", "TFC", "GS", "MS", "BK",
    ],
    "Payments & consumer finance": [
        "V", "MA", "AXP", "COF", "DFS", "SYF",
    ],
    "Integrated oil & gas": [
        "XOM", "CVX", "COP", "OXY", "EOG", "PXD",
    ],
    "Oil services & refiners": [
        "SLB", "HAL", "BKR", "MPC", "VLO", "PSX",
    ],
    "Healthcare & pharma": [
        "JNJ", "PFE", "MRK", "ABBV", "LLY", "BMY", "UNH", "CVS",
    ],
    "Consumer staples": [
        "PG", "KO", "CL", "KMB", "GIS", "K", "MO",
    ],
    "Retail": [
        "WMT", "TGT", "HD", "LOW", "TJX", "DG", "DLTR",
    ],
    "Industrials": [
        "BA", "CAT", "DE", "GE", "MMM", "UPS", "FDX", "EMR", "ETN",
    ],
    "Telecom & media": [
        "T", "VZ", "DIS", "CMCSA",
    ],
    "Autos & transport": [
        "F", "GM", "DAL", "LUV", "UNP", "CSX",
    ],
    "Utilities": [
        "NEE", "DUK", "SO", "D", "AEP", "EXC",
    ],
}


def all_tickers() -> list[str]:
    """Flat, de-duplicated list of every ticker in the universe."""
    seen: set[str] = set()
    out: list[str] = []
    for tickers in NYSE_SECTORS.values():
        for t in tickers:
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out


def sector_of(ticker: str) -> str | None:
    """Return the sector label for a ticker, or ``None`` if not in the universe."""
    for sector, tickers in NYSE_SECTORS.items():
        if ticker in tickers:
            return sector
    return None
