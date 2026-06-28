"""TeamTracy — NYSE cointegrated pairs dashboard.

Streamlit app that screens a curated NYSE universe for the most strongly
cointegrated stock pairs (Engle-Granger test), then uses a Kalman filter to
estimate a time-varying hedge ratio and smooth each pair's spread. The top 10
pairs are ranked in a table; selecting any pair drills into its prices, dynamic
hedge ratio, smoothed spread and trading-signal z-score.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.cointegration import PairResult, screen_pairs, top_pairs
from src.data import load_prices
from src.universe import all_tickers

st.set_page_config(
    page_title="TeamTracy · NYSE Cointegrated Pairs",
    page_icon="📈",
    layout="wide",
)


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def _cached_prices(tickers: tuple[str, ...], period: str) -> pd.DataFrame:
    return load_prices(list(tickers), period=period)


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def _cached_screen(
    period: str,
    within_sector_only: bool,
    max_pvalue: float,
    z_window: int,
    kalman_delta: float,
    kalman_obs_cov: float,
) -> list[PairResult]:
    prices = _cached_prices(tuple(all_tickers()), period)
    if prices.empty:
        return []
    return screen_pairs(
        prices,
        within_sector_only=within_sector_only,
        max_pvalue=max_pvalue,
        z_window=z_window,
        kalman_delta=kalman_delta,
        kalman_obs_cov=kalman_obs_cov,
    )


def _ranking_frame(results: list[PairResult]) -> pd.DataFrame:
    rows = []
    for i, r in enumerate(results, start=1):
        hl = r.half_life_bars
        rows.append(
            {
                "Rank": i,
                "Pair": r.label,
                "Sector": r.sector or "—",
                "p-value": round(r.pvalue, 5),
                "Correlation": round(r.correlation, 3),
                "Hedge ratio β": round(r.beta_last, 3),
                "Half-life (days)": round(hl, 1) if hl != float("inf") else "∞",
                "z-score (latest)": round(r.zscore_last, 2),
            }
        )
    return pd.DataFrame(rows)


def _pair_figure(r: PairResult) -> go.Figure:
    kf = r.kalman
    prices_idx = kf.spread.index

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.34, 0.33, 0.33],
        subplot_titles=(
            f"Prices — {r.y} vs {r.x}",
            "Kalman hedge ratio β (units of x per unit of y)",
            "Smoothed spread z-score (mean-reversion signal)",
        ),
    )

    # Row 1: dual-axis prices.
    fig.add_trace(
        go.Scatter(x=prices_idx, y=_align(r, r.y), name=r.y, line=dict(color="#1f77b4")),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=prices_idx, y=_align(r, r.x), name=r.x,
            line=dict(color="#ff7f0e"), yaxis="y2",
        ),
        row=1, col=1,
    )

    # Row 2: dynamic hedge ratio.
    fig.add_trace(
        go.Scatter(x=kf.beta.index, y=kf.beta.values, name="β", line=dict(color="#2ca02c")),
        row=2, col=1,
    )

    # Row 3: z-score with trading bands.
    fig.add_trace(
        go.Scatter(x=kf.zscore.index, y=kf.zscore.values, name="z-score", line=dict(color="#9467bd")),
        row=3, col=1,
    )
    for level, dash in [(0, "solid"), (2, "dash"), (-2, "dash")]:
        fig.add_hline(y=level, line=dict(color="gray", dash=dash, width=1), row=3, col=1)

    fig.update_layout(
        height=720,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        # Secondary y-axis for the second price series in row 1.
        yaxis2=dict(overlaying="y", side="right", showgrid=False),
    )
    return fig


def _align(r: PairResult, ticker: str) -> pd.Series:
    """Reattach the original price series for a leg, aligned to the spread index."""
    # Prices are re-derived from the dashboard's price cache to keep PairResult
    # lightweight; fall back gracefully if unavailable.
    prices = st.session_state.get("_prices")
    if prices is not None and ticker in prices.columns:
        return prices[ticker].reindex(r.kalman.spread.index)
    return pd.Series(index=r.kalman.spread.index, dtype=float)


def main() -> None:
    st.title("📈 TeamTracy — NYSE Cointegrated Pairs")
    st.caption(
        "Top cointegrated NYSE stock pairs (Engle-Granger) with Kalman-filtered "
        "dynamic hedge ratios and smoothed spreads. Data: Yahoo Finance."
    )

    with st.sidebar:
        st.header("Settings")
        period = st.selectbox(
            "History window", ["1y", "2y", "3y", "5y"], index=1,
            help="Lookback used for both the cointegration test and the Kalman filter.",
        )
        within_sector_only = st.checkbox(
            "Screen within sector only", value=True,
            help="Faster and more economically meaningful. Uncheck for an all-pairs sweep.",
        )
        max_pvalue = st.slider(
            "Max cointegration p-value", 0.01, 0.10, 0.05, 0.01,
            help="Pairs above this Engle-Granger p-value are discarded.",
        )
        z_window = st.slider("Z-score window (days)", 20, 120, 60, 5)
        with st.expander("Kalman filter tuning"):
            kalman_delta = st.select_slider(
                "δ (hedge-ratio adaptivity)",
                options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
                value=1e-4,
                format_func=lambda v: f"{v:.0e}",
            )
            kalman_obs_cov = st.select_slider(
                "R (observation noise)",
                options=[1e-4, 1e-3, 1e-2, 1e-1],
                value=1e-3,
                format_func=lambda v: f"{v:.0e}",
            )
        run = st.button("Run screen", type="primary", use_container_width=True)

    if run or "results" not in st.session_state:
        with st.spinner("Downloading prices and screening pairs…"):
            prices = _cached_prices(tuple(all_tickers()), period)
            st.session_state["_prices"] = prices
            if prices.empty:
                st.error(
                    "No price data returned from Yahoo Finance. Check connectivity "
                    "and try again."
                )
                st.session_state["results"] = []
            else:
                st.session_state["results"] = _cached_screen(
                    period, within_sector_only, max_pvalue,
                    z_window, kalman_delta, kalman_obs_cov,
                )

    results: list[PairResult] = st.session_state.get("results", [])
    if not results:
        st.info("Adjust the settings and press **Run screen** to find cointegrated pairs.")
        return

    best = top_pairs(results, n=10)

    st.subheader("Top 10 cointegrated pairs")
    st.dataframe(
        _ranking_frame(best),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        f"Screened {len(results)} cointegrated pairs out of the NYSE universe; "
        "ranked by ascending Engle-Granger p-value. Lower is stronger."
    )

    st.subheader("Pair detail")
    labels = [r.label for r in best]
    choice = st.selectbox("Select a pair to inspect", labels, index=0)
    selected = next(r for r in best if r.label == choice)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cointegration p-value", f"{selected.pvalue:.4f}")
    c2.metric("Latest hedge ratio β", f"{selected.beta_last:.3f}")
    hl = selected.half_life_bars
    c3.metric("Spread half-life", "∞" if hl == float("inf") else f"{hl:.1f} d")
    c4.metric("Latest z-score", f"{selected.zscore_last:.2f}")

    st.plotly_chart(_pair_figure(selected), use_container_width=True)

    with st.expander("How to read this"):
        st.markdown(
            """
            - **Cointegration p-value** — Engle-Granger test. Low values mean a
              stationary (mean-reverting) linear combination of the two prices exists.
            - **Kalman hedge ratio β** — the time-varying number of shares of the
              second leg per share of the first. Letting it drift keeps the spread
              stationary even as the relationship evolves.
            - **Smoothed spread z-score** — the Kalman residual, standardised. A
              z-score beyond ±2 is a classic mean-reversion entry signal: short the
              spread above +2, long it below −2, exit near 0.
            - **Half-life** — expected time for the spread to revert halfway to its
              mean. Shorter half-lives suit faster mean-reversion strategies.

            *Research/education only — not investment advice.*
            """
        )


if __name__ == "__main__":
    main()
