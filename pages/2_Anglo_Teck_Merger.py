"""Anglo-Teck merger analysis & forward stock projection.

Third page of the TeamTracy app. Layers the September-2025 Anglo American / Teck
merger onto the Teck dashboard: co-movement and merger-arbitrage spread versus
Anglo American, plus a Monte Carlo projection that blends a standalone path with
a deal-completion scenario.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.merger import (
    ANGLO_NAME,
    MERGER_ANNOUNCED,
    MergerTerms,
    annualized_arb_return,
    fetch_anglo_history,
    merger_spread,
    scenario_forecast,
    spread_summary,
)
from src.teck import TECK_NAME, fetch_teck_history

st.set_page_config(page_title="Anglo-Teck Merger · TeamTracy", page_icon="🤝", layout="wide")

_TECK = "#1f77b4"
_ANGLO = "#d62728"
_DEAL = "#2ca02c"
_MUTED = "#8c8c8c"


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def _teck(years: int) -> pd.DataFrame:
    return fetch_teck_history(period_years=years)


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def _anglo(years: int) -> pd.DataFrame:
    return fetch_anglo_history(period_years=years)


def _relative_figure(teck: pd.Series, anglo: pd.Series, since: pd.Timestamp) -> go.Figure:
    df = pd.concat([teck.rename("TECK"), anglo.rename("Anglo")], axis=1).dropna()
    df = df.loc[df.index >= since]
    if df.empty:
        return go.Figure()
    norm = df / df.iloc[0] * 100.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=norm.index, y=norm["TECK"], name="TECK",
                             line=dict(color=_TECK, width=2)))
    fig.add_trace(go.Scatter(x=norm.index, y=norm["Anglo"], name="Anglo American",
                             line=dict(color=_ANGLO, width=2)))
    fig.update_layout(height=380, margin=dict(l=40, r=40, t=30, b=40),
                      yaxis_title="Growth of $100 since announcement",
                      legend=dict(orientation="h", y=1.02, x=0))
    return fig


def _rolling_corr_figure(teck: pd.Series, anglo: pd.Series, window: int = 60) -> go.Figure:
    rets = pd.concat([teck.pct_change().rename("t"), anglo.pct_change().rename("a")],
                     axis=1).dropna()
    roll = rets["t"].rolling(window).corr(rets["a"])
    fig = go.Figure(go.Scatter(x=roll.index, y=roll.values, line=dict(color=_TECK)))
    fig.add_vline(x=pd.Timestamp(MERGER_ANNOUNCED), line=dict(color=_DEAL, dash="dash"),
                  annotation_text="Announcement", annotation_position="top")
    fig.add_hline(y=0, line=dict(color=_MUTED, dash="dot", width=1))
    fig.update_layout(height=320, margin=dict(l=40, r=40, t=30, b=40),
                      yaxis_title=f"{window}-day corr (TECK vs Anglo)", yaxis_range=[-1, 1])
    return fig


def _spread_figure(spread_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spread_df.index, y=spread_df["teck"], name="TECK price",
                             line=dict(color=_TECK)))
    fig.add_trace(go.Scatter(x=spread_df.index, y=spread_df["implied"],
                             name="Deal-implied value", line=dict(color=_DEAL, dash="dash")))
    fig.update_layout(height=360, margin=dict(l=40, r=40, t=30, b=40),
                      yaxis_title="Price ($)", legend=dict(orientation="h", y=1.02, x=0))
    return fig


def _projection_figure(forecast, teck_close: pd.Series) -> go.Figure:
    sa = forecast.standalone
    bands = sa.bands
    hist = teck_close.dropna().iloc[-120:]

    fig = go.Figure()
    # History.
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="History",
                             line=dict(color=_MUTED)))
    # Standalone MC fan (p05-p95 shaded, p25-p75 shaded, p50 line).
    fig.add_trace(go.Scatter(x=bands.index, y=bands["p95"], line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=bands.index, y=bands["p05"], fill="tonexty",
                             fillcolor="rgba(31,119,180,0.12)", line=dict(width=0),
                             name="5–95%"))
    fig.add_trace(go.Scatter(x=bands.index, y=bands["p75"], line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=bands.index, y=bands["p25"], fill="tonexty",
                             fillcolor="rgba(31,119,180,0.25)", line=dict(width=0),
                             name="25–75%"))
    fig.add_trace(go.Scatter(x=bands.index, y=bands["p50"], name="Standalone median",
                             line=dict(color=_TECK, width=2)))
    # Deal-completion median (Anglo-linked).
    fig.add_trace(go.Scatter(x=forecast.deal_median.index, y=forecast.deal_median.values,
                             name="Deal-completion median", line=dict(color=_DEAL, dash="dash", width=2)))
    fig.update_layout(height=460, margin=dict(l=40, r=40, t=30, b=40),
                      yaxis_title="Projected TECK price ($)",
                      legend=dict(orientation="h", y=1.02, x=0))
    return fig


def _terminal_hist(forecast) -> go.Figure:
    term = forecast.standalone.terminal
    fig = go.Figure(go.Histogram(x=term, nbinsx=50, marker_color=_TECK, opacity=0.75))
    fig.add_vline(x=forecast.standalone.s0, line=dict(color=_MUTED, dash="dash"),
                  annotation_text="Today")
    fig.add_vline(x=forecast.deal_value_now, line=dict(color=_DEAL, dash="dash"),
                  annotation_text="Deal value")
    fig.update_layout(height=320, margin=dict(l=40, r=40, t=30, b=40),
                      xaxis_title="Projected terminal price ($)", yaxis_title="Paths")
    return fig


def main() -> None:
    st.title("🤝 Anglo-Teck Merger & Stock Projection")
    st.caption(
        f"In September 2025 {ANGLO_NAME.split(' (')[0]} and {TECK_NAME.split(' (')[0]} "
        "announced a merger of equals to form **Anglo Teck**, a copper-focused "
        "major. This page tracks the two stocks' convergence, the merger-arbitrage "
        "spread, and a forward projection blending standalone and deal scenarios."
    )

    with st.sidebar:
        st.header("Merger assumptions")
        st.caption("Forward-looking — adjust to match official filings.")
        years = st.select_slider("History (years)", options=[2, 3, 5], value=3)
        ratio = st.number_input(
            "Anglo ADRs per Teck share (exchange ratio)",
            min_value=0.1, max_value=5.0, value=1.3301, step=0.01, format="%.4f",
        )
        completion = st.slider("Deal completion probability", 0.0, 1.0, 0.80, 0.05)
        months = st.slider("Horizon / expected months to close", 3, 24, 12, 1)
        drift_mode = st.radio(
            "Projection drift", ["random_walk", "historical"],
            format_func=lambda m: "Random walk (zero drift)" if m == "random_walk"
            else "Historical drift",
            help="Random walk is the conservative default; historical extrapolates "
                 "past average returns.",
        )
        n_paths = st.select_slider("Monte Carlo paths", options=[1000, 2000, 5000, 10000], value=5000)
        if st.button("Refresh data", use_container_width=True):
            st.cache_data.clear()

    try:
        with st.spinner("Scraping Teck and Anglo American prices…"):
            teck_ohlcv = _teck(years)
            anglo_ohlcv = _anglo(years)
    except Exception as exc:
        st.error(f"Could not scrape price data — try **Refresh data** shortly.\n\n`{exc}`")
        return

    if teck_ohlcv.empty or "Close" not in teck_ohlcv:
        st.error("No Teck price data returned. Try **Refresh data** shortly.")
        return
    if anglo_ohlcv.empty or "Close" not in anglo_ohlcv:
        st.warning(
            "Anglo American price data could not be scraped, so the merger and "
            "projection analysis is unavailable. Try **Refresh data** shortly."
        )
        return

    teck_close = teck_ohlcv["Close"]
    anglo_close = anglo_ohlcv["Close"]
    terms = MergerTerms(
        exchange_ratio=ratio,
        completion_prob=completion,
        horizon_days=int(months / 12 * 252),
    )

    # ---- Convergence ------------------------------------------------------ #
    st.subheader("Teck vs Anglo American since the announcement")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            _relative_figure(teck_close, anglo_close, pd.Timestamp(MERGER_ANNOUNCED)),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(_rolling_corr_figure(teck_close, anglo_close),
                        use_container_width=True)

    # ---- Merger arbitrage ------------------------------------------------- #
    st.subheader("Merger-arbitrage spread")
    spread_df = merger_spread(teck_close, anglo_close, ratio,
                              since=MERGER_ANNOUNCED)
    if spread_df.empty:
        st.info("No overlapping post-announcement data yet for the arbitrage spread.")
    else:
        summ = spread_summary(spread_df)
        days_to_close = int(months / 12 * 365.25)
        arb = annualized_arb_return(summ["upside_to_deal"], days_to_close)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Teck price", f"${summ['teck']:,.2f}")
        m2.metric("Deal-implied value", f"${summ['implied']:,.2f}",
                  help="Exchange ratio × Anglo ADR price.")
        m3.metric("Spread to deal", f"{summ['spread_pct'] * 100:+.1f}%",
                  help="Teck vs implied value. Negative = trading below deal value.")
        m4.metric("Annualised arb return", "—" if np.isnan(arb) else f"{arb * 100:+.1f}%",
                  help=f"Convergence upside annualised over ~{months} months to close.")
        st.plotly_chart(_spread_figure(spread_df), use_container_width=True)
        st.caption(
            "When Teck trades below the deal-implied value, the discount is the "
            "market pricing in deal risk and time to close; convergence on "
            "completion is the arbitrage return."
        )

    # ---- Projection ------------------------------------------------------- #
    st.subheader(f"Projected stock performance — next ~{months} months")
    try:
        forecast = scenario_forecast(
            teck_close, anglo_close, terms,
            n_paths=n_paths, drift_mode=drift_mode,
        )
    except ValueError as exc:
        st.info(f"Not enough data to project: {exc}")
        return

    sa = forecast.standalone
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Blended expected price", f"${forecast.blended_exp_price:,.2f}",
              f"{(forecast.blended_exp_price / sa.s0 - 1) * 100:+.1f}%")
    k2.metric("P(gain), blended", f"{forecast.blended_prob_gain * 100:.0f}%")
    k3.metric("Standalone median", f"${sa.bands['p50'].iloc[-1]:,.2f}",
              f"{(sa.bands['p50'].iloc[-1] / sa.s0 - 1) * 100:+.1f}%")
    k4.metric("Deal-implied upside now", f"{forecast.implied_upside * 100:+.1f}%",
              help="Upside if Teck converged to today's deal-implied value.")

    st.plotly_chart(_projection_figure(forecast, teck_close), use_container_width=True)
    cc1, cc2 = st.columns([2, 1])
    with cc1:
        st.markdown("**Terminal price distribution (standalone Monte Carlo)**")
        st.plotly_chart(_terminal_hist(forecast), use_container_width=True)
    with cc2:
        st.markdown("**Model inputs**")
        st.write(
            {
                "Start price": f"${sa.s0:,.2f}",
                "Ann. volatility": f"{sa.vol_daily * np.sqrt(252) * 100:.1f}%",
                "Drift": "zero (random walk)" if drift_mode == "random_walk" else "historical",
                "Completion prob.": f"{completion * 100:.0f}%",
                "Exchange ratio": f"{ratio:.4f}",
                "Paths": f"{n_paths:,}",
            }
        )
    st.caption(
        "The blend weights a deal-completion outcome (Teck converging to the "
        "Anglo-linked value) against a standalone path by the completion "
        "probability. Monte Carlo assumes log-normal returns — a simplification."
    )

    st.divider()
    st.caption(
        "Merger terms and completion probability are user-set assumptions, not "
        "official figures — verify against Anglo American / Teck filings. Sources: "
        "Stooq (scraped prices). Projections are illustrative and probabilistic. "
        "Research/educational use only — not investment advice."
    )


main()
