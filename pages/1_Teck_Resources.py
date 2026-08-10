"""Teck Resources — stock performance & predictor dashboard.

A second page in the TeamTracy app. It web-scrapes Teck Resources price history,
a basket of commodity/macro predictor proxies, and recent news, then visualises
Teck's performance and quantifies which factors drive its returns.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.teck import (
    DEFAULT_PREDICTORS,
    TECK_NAME,
    compute_performance_stats,
    fetch_predictor_closes,
    fetch_teck_history,
    predictor_analysis,
    scrape_news,
    scrape_quote_snapshot,
)

st.set_page_config(page_title="Teck Resources · TeamTracy", page_icon="⛏️", layout="wide")

# Consistent, colour-blind-safe palette shared across the page's charts.
_ACCENT = "#1f77b4"
_UP = "#2ca02c"
_DOWN = "#d62728"
_MUTED = "#8c8c8c"


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def _load_history(period_years: int) -> pd.DataFrame:
    return fetch_teck_history(period_years=period_years)


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def _load_predictors(period_years: int) -> pd.DataFrame:
    return fetch_predictor_closes(DEFAULT_PREDICTORS, period_years=period_years)


@st.cache_data(show_spinner=False, ttl=3 * 3600)
def _load_news() -> list:
    return scrape_news()


@st.cache_data(show_spinner=False, ttl=3 * 3600)
def _load_snapshot() -> dict:
    return scrape_quote_snapshot()


def _price_figure(ohlcv: pd.DataFrame) -> go.Figure:
    close = ohlcv["Close"]
    has_vol = "Volume" in ohlcv.columns and ohlcv["Volume"].notna().any()

    fig = make_subplots(
        rows=2 if has_vol else 1, cols=1, shared_xaxes=True,
        vertical_spacing=0.05, row_heights=[0.75, 0.25] if has_vol else [1.0],
        subplot_titles=("Price & moving averages", "Volume") if has_vol else ("Price & moving averages",),
    )
    fig.add_trace(
        go.Scatter(x=close.index, y=close.values, name="Close", line=dict(color=_ACCENT)),
        row=1, col=1,
    )
    for win, color in [(50, "#ff7f0e"), (200, _MUTED)]:
        if len(close) >= win:
            ma = close.rolling(win).mean()
            fig.add_trace(
                go.Scatter(x=ma.index, y=ma.values, name=f"{win}-day MA",
                           line=dict(color=color, width=1.3)),
                row=1, col=1,
            )
    if has_vol:
        fig.add_trace(
            go.Bar(x=ohlcv.index, y=ohlcv["Volume"], name="Volume",
                   marker_color=_MUTED, opacity=0.6),
            row=2, col=1,
        )
    fig.update_layout(
        height=520, margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _relative_figure(teck_close: pd.Series, predictors: pd.DataFrame) -> go.Figure:
    """Growth of $100 in Teck vs each predictor over the common window."""
    combined = pd.concat([teck_close.rename("TECK"), predictors], axis=1).dropna()
    norm = combined / combined.iloc[0] * 100.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=norm.index, y=norm["TECK"], name="TECK",
        line=dict(color=_ACCENT, width=2.5),
    ))
    for col in predictors.columns:
        if col in norm:
            fig.add_trace(go.Scatter(x=norm.index, y=norm[col], name=col,
                                     line=dict(width=1)))
    fig.update_layout(
        height=440, margin=dict(l=40, r=40, t=30, b=40),
        yaxis_title="Growth of $100",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def _corr_bar(corr: pd.Series) -> go.Figure:
    colors = [_UP if v >= 0 else _DOWN for v in corr.values]
    fig = go.Figure(go.Bar(
        x=corr.values, y=corr.index, orientation="h", marker_color=colors,
        text=[f"{v:.2f}" for v in corr.values], textposition="auto",
    ))
    fig.update_layout(
        height=360, margin=dict(l=40, r=20, t=10, b=30),
        xaxis_title="Correlation of daily returns with TECK",
        yaxis=dict(autorange="reversed"),
    )
    return fig


def _beta_bar(betas: pd.Series) -> go.Figure:
    colors = [_UP if v >= 0 else _DOWN for v in betas.values]
    fig = go.Figure(go.Bar(
        x=betas.values, y=betas.index, orientation="h", marker_color=colors,
        text=[f"{v:+.3f}" for v in betas.values], textposition="auto",
    ))
    fig.update_layout(
        height=360, margin=dict(l=40, r=20, t=10, b=30),
        xaxis_title="Standardised OLS coefficient (return sensitivity)",
        yaxis=dict(autorange="reversed"),
    )
    return fig


def _rolling_figure(rolling: pd.Series) -> go.Figure:
    fig = go.Figure(go.Scatter(x=rolling.index, y=rolling.values,
                               line=dict(color=_ACCENT), name=rolling.name))
    fig.add_hline(y=0, line=dict(color=_MUTED, dash="dash", width=1))
    fig.update_layout(
        height=320, margin=dict(l=40, r=40, t=30, b=40),
        yaxis_title="Rolling correlation", yaxis_range=[-1, 1],
    )
    return fig


def main() -> None:
    st.title("⛏️ Teck Resources — Performance & Predictors")
    st.caption(
        f"{TECK_NAME} (NYSE: TECK). Prices and predictors web-scraped from Stooq; "
        "headlines from Google News. Teck is a diversified base-metals miner, so "
        "its stock tracks copper, metals & energy far more than company news."
    )

    with st.sidebar:
        st.header("Teck settings")
        years = st.select_slider("History (years)", options=[1, 2, 3, 5], value=3)
        refresh = st.button("Refresh data", use_container_width=True)
        if refresh:
            st.cache_data.clear()

    try:
        with st.spinner("Scraping Teck price history…"):
            ohlcv = _load_history(years)
    except Exception as exc:
        st.error(
            "Could not scrape Teck price data. The source may be temporarily "
            f"unreachable — try **Refresh data** in a moment.\n\n`{exc}`"
        )
        return

    if ohlcv.empty or "Close" not in ohlcv:
        st.error("No Teck price data was returned. Try **Refresh data** shortly.")
        return

    # ---- Key statistics --------------------------------------------------- #
    stats = compute_performance_stats(ohlcv)
    stats.update(_load_snapshot())  # best-effort scraped live quote
    st.subheader("Key statistics")
    keys = list(stats.keys())
    for row_keys in (keys[:4], keys[4:]):
        cols = st.columns(max(len(row_keys), 1))
        for col, k in zip(cols, row_keys):
            col.metric(k, stats[k])

    # ---- Price performance ------------------------------------------------ #
    st.subheader("Stock performance")
    st.plotly_chart(_price_figure(ohlcv), use_container_width=True)

    # ---- Predictors ------------------------------------------------------- #
    with st.spinner("Scraping predictor factors…"):
        predictors = _load_predictors(years)

    if predictors.empty:
        st.warning(
            "Predictor factor data could not be scraped right now, so the driver "
            "analysis is unavailable. Try **Refresh data** shortly."
        )
    else:
        st.subheader("Performance vs predictor factors")
        st.plotly_chart(
            _relative_figure(ohlcv["Close"], predictors), use_container_width=True
        )

        try:
            analysis = predictor_analysis(ohlcv["Close"], predictors)
        except ValueError as exc:
            st.info(f"Not enough overlapping data for the driver model: {exc}")
            analysis = None

        if analysis is not None:
            st.subheader("What drives Teck's returns?")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Correlation of daily returns**")
                st.plotly_chart(_corr_bar(analysis.correlations), use_container_width=True)
            with c2:
                st.markdown(
                    f"**Multi-factor model** · R² = {analysis.r_squared:.2f} "
                    f"({analysis.r_squared * 100:.0f}% of daily variance explained)"
                )
                st.plotly_chart(_beta_bar(analysis.betas), use_container_width=True)

            st.markdown(
                f"**Rolling 60-day correlation with the top driver "
                f"({analysis.top_predictor})**"
            )
            st.plotly_chart(_rolling_figure(analysis.rolling_corr.dropna()),
                            use_container_width=True)
            st.caption(
                "Standardised OLS coefficients show each factor's return "
                "sensitivity on a comparable scale; a high R² means Teck moves "
                "largely with these commodity/macro factors rather than on its own."
            )

    # ---- News ------------------------------------------------------------- #
    st.subheader("Latest Teck headlines")
    news = _load_news()
    if not news:
        st.caption("No headlines could be scraped right now.")
    else:
        for item in news:
            meta = " · ".join(x for x in (item.source, item.published) if x)
            st.markdown(f"- [{item.title}]({item.link})  \n  <small>{meta}</small>",
                        unsafe_allow_html=True)

    st.divider()
    st.caption(
        "Sources: Stooq (prices, scraped CSV), Google News RSS (headlines). "
        "Predictor factors are proxy ETFs/indices, not Teck's exact inputs. "
        "Research/educational use only — not investment advice."
    )


main()
