"""TeamTracy — NYSE cointegrated pairs dashboard.

Streamlit app that scans the entire NYSE for the most strongly cointegrated
stock pairs (Engle-Granger test), then uses a Kalman filter to estimate a
time-varying hedge ratio and smooth each pair's spread. The top 10 pairs are
ranked in a table; selecting any pair drills into its prices, dynamic hedge
ratio, smoothed spread, trading-signal z-score, and a $100k backtest.

The ticker universe is fetched live from the NASDAQ Trader symbol directory and
company names come from Yahoo Finance — nothing is hard-coded.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.backtest import BacktestResult, backtest_pair
from src.cointegration import PairResult, screen_pairs, top_pairs
from src.data import get_company_info, load_prices
from src.universe import nyse_tickers

st.set_page_config(
    page_title="TeamTracy · NYSE Cointegrated Pairs",
    page_icon="📈",
    layout="wide",
)


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def _cached_universe(limit: int | None) -> list[str]:
    return nyse_tickers(limit=limit)


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def _cached_prices(tickers: tuple[str, ...], period: str) -> pd.DataFrame:
    return load_prices(list(tickers), period=period)


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def _cached_company_info(tickers: tuple[str, ...]) -> dict:
    return get_company_info(list(tickers))


def _ranking_frame(results: list[PairResult], info: dict) -> pd.DataFrame:
    rows = []
    for i, r in enumerate(results, start=1):
        hl = r.half_life_bars
        a_info = info.get(r.y, {})
        b_info = info.get(r.x, {})
        rows.append(
            {
                "Rank": i,
                "Pair": r.label,
                "Stock A": a_info.get("name", r.y),
                "Stock B": b_info.get("name", r.x),
                "Sector": a_info.get("sector", "—"),
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
        scan_all = st.checkbox(
            "Scan the entire NYSE", value=True,
            help="Fetches every NYSE common stock live. Thorough but slower on the "
                 "first run; results are cached afterwards. Uncheck to cap the size.",
        )
        universe_limit = None
        if not scan_all:
            universe_limit = st.slider(
                "Max tickers to scan", 50, 1500, 300, 50,
                help="Alphabetical cap on the NYSE universe for a faster scan.",
            )
        min_correlation = st.slider(
            "Min return correlation (pre-filter)", 0.5, 0.99, 0.80, 0.01,
            help="Only the most correlated pairs are tested for cointegration. "
                 "Higher = fewer candidates, faster scan.",
        )
        max_candidates = st.slider(
            "Max candidate pairs", 100, 5000, 1000, 100,
            help="Upper bound on how many correlated pairs run the cointegration test.",
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
        with st.expander("Backtest settings"):
            initial_capital = st.number_input(
                "Starting capital ($)", min_value=1_000, max_value=10_000_000,
                value=100_000, step=10_000,
            )
            entry_z = st.slider("Entry z-score", 1.0, 3.0, 2.0, 0.1)
            exit_z = st.slider("Exit z-score", 0.0, 1.5, 0.5, 0.1)
            cost_bps = st.slider(
                "Transaction cost (bps per turn)", 0.0, 10.0, 1.0, 0.5,
            )
        run = st.button("Run screen", type="primary", use_container_width=True)

    if run:
        st.session_state["ran"] = True
        diag: dict = {}
        try:
            with st.spinner("Fetching the NYSE universe…"):
                universe = _cached_universe(universe_limit)
        except Exception as exc:
            universe = []
            diag["error"] = f"Could not fetch the NYSE listing: {exc}"

        diag["requested"] = len(universe)
        if universe:
            with st.spinner(
                f"Downloading prices for {len(universe)} NYSE tickers… "
                "(first run can take a few minutes; Yahoo may throttle)"
            ):
                prices = _cached_prices(tuple(universe), period)
            st.session_state["_prices"] = prices
            diag["downloaded"] = int(prices.shape[1])

            if not prices.empty:
                with st.spinner(
                    f"Screening {prices.shape[1]} tickers for cointegrated pairs…"
                ):
                    st.session_state["results"] = screen_pairs(
                        prices,
                        min_correlation=min_correlation,
                        max_candidates=max_candidates,
                        max_pvalue=max_pvalue,
                        z_window=z_window,
                        kalman_delta=kalman_delta,
                        kalman_obs_cov=kalman_obs_cov,
                    )
            else:
                st.session_state["results"] = []
        else:
            st.session_state["results"] = []
        st.session_state["diag"] = diag

    if not st.session_state.get("ran"):
        st.info(
            "Adjust the settings in the sidebar and press **Run screen** to scan "
            "the NYSE for cointegrated pairs."
        )
        return

    # --- Report on the most recent run, surfacing any data problems clearly. ---
    diag = st.session_state.get("diag", {})
    results: list[PairResult] = st.session_state.get("results", [])

    if diag.get("error"):
        st.error(diag["error"])
        return

    requested = diag.get("requested", 0)
    downloaded = diag.get("downloaded", 0)

    if downloaded == 0:
        st.error(
            "**Yahoo Finance returned no price data.** Its servers frequently "
            "rate-limit shared cloud IPs (HTTP 429), which is the most likely "
            "cause on Streamlit Cloud. Wait a minute and press **Run screen** "
            "again, or lower **Max tickers to scan** in the sidebar for a "
            "lighter request."
        )
        return

    if requested and downloaded < 0.5 * requested:
        st.warning(
            f"Only **{downloaded} of {requested}** tickers returned data — Yahoo "
            "likely throttled the rest. Coverage (and results) may be thin; retry "
            "in a minute for a fuller scan."
        )

    if not results:
        st.warning(
            f"Downloaded **{downloaded}** tickers but found **no cointegrated "
            "pairs** at these settings. Try lowering **Min return correlation** "
            "(e.g. 0.70–0.80) or raising **Max cointegration p-value**."
        )
        return

    best = top_pairs(results, n=10)

    # Company names/sectors for just the displayed tickers, from Yahoo Finance.
    shown = sorted({t for r in best for t in (r.y, r.x)})
    info = _cached_company_info(tuple(shown))

    st.subheader("Top 10 cointegrated pairs")
    st.dataframe(
        _ranking_frame(best, info),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        f"Downloaded {downloaded} of {requested} NYSE tickers; found "
        f"{len(results)} cointegrated pairs, ranked by ascending Engle-Granger "
        "p-value (lower is stronger)."
    )

    st.subheader("Pair detail")
    labels = [r.label for r in best]
    choice = st.selectbox("Select a pair to inspect", labels, index=0)
    selected = next(r for r in best if r.label == choice)
    st.markdown(
        f"**{info.get(selected.y, {}).get('name', selected.y)}** ({selected.y}) "
        f"vs **{info.get(selected.x, {}).get('name', selected.x)}** ({selected.x})"
    )

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

    _render_backtest(selected, initial_capital, entry_z, exit_z, cost_bps)


def _render_backtest(
    selected: PairResult,
    initial_capital: float,
    entry_z: float,
    exit_z: float,
    cost_bps: float,
) -> None:
    st.subheader("Backtest")
    st.caption(
        f"Z-score mean-reversion on the Kalman spread, dollar-neutral, starting "
        f"from ${initial_capital:,.0f}. Enter at |z| ≥ {entry_z:g}, exit at "
        f"|z| ≤ {exit_z:g}. Tune in the sidebar under **Backtest settings**."
    )

    prices = st.session_state.get("_prices")
    if prices is None or selected.y not in prices or selected.x not in prices:
        st.info("Run a screen to backtest the selected pair.")
        return

    y = prices[selected.y].reindex(selected.kalman.spread.index)
    x = prices[selected.x].reindex(selected.kalman.spread.index)
    try:
        bt = backtest_pair(
            y, x, selected.kalman.zscore,
            entry=entry_z, exit=exit_z,
            initial_capital=float(initial_capital), cost_bps=cost_bps,
        )
    except ValueError as exc:
        st.warning(f"Could not backtest this pair: {exc}")
        return

    metrics = bt.metrics_dict()
    row1 = st.columns(5)
    row2 = st.columns(4)
    keys = list(metrics.keys())
    for col, key in zip(row1, keys[:5]):
        col.metric(key, metrics[key])
    for col, key in zip(row2, keys[5:]):
        col.metric(key, metrics[key])

    st.plotly_chart(_equity_figure(bt), use_container_width=True)

    if bt.num_trades == 0:
        st.info(
            "No trades were triggered over this window with the current bands. "
            "Try a lower entry z-score."
        )


def _equity_figure(bt: BacktestResult) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=("Equity curve ($)", "Drawdown"),
    )
    fig.add_trace(
        go.Scatter(
            x=bt.equity.index, y=bt.equity.values, name="Equity",
            line=dict(color="#1f77b4"),
        ),
        row=1, col=1,
    )
    fig.add_hline(
        y=bt.initial_capital, line=dict(color="gray", dash="dash", width=1),
        row=1, col=1,
    )
    drawdown = bt.equity / bt.equity.cummax() - 1.0
    fig.add_trace(
        go.Scatter(
            x=drawdown.index, y=drawdown.values, name="Drawdown",
            fill="tozeroy", line=dict(color="#d62728"),
        ),
        row=2, col=1,
    )
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    fig.update_layout(
        height=460, margin=dict(l=40, r=40, t=50, b=40), showlegend=False,
    )
    return fig


if __name__ == "__main__":
    main()
