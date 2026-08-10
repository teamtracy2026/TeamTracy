# 📈 TeamTracy — NYSE Cointegrated Pairs Dashboard

A Streamlit app with two dashboards: (1) the **top 10 best cointegrated stock
pairs on the NYSE**, using **Yahoo Finance** price data and a **Kalman filter**
to estimate a time-varying hedge ratio and smooth each pair's spread over time;
and (2) a web-scraped **Teck Resources** performance & predictor dashboard.

Cointegrated pairs are the raw material of statistical-arbitrage / pairs-trading
strategies: two stocks whose prices wander individually but whose *spread* is
mean-reverting. The dashboard scans the **entire NYSE**, ranks pairs by the
strength of their cointegration, and lets you drill into each one's prices,
dynamic hedge ratio, smoothed spread, trading-signal z-score, and a backtest.

**Nothing is hard-coded:** the ticker universe is fetched live from the NYSE
listing directory and company names come from Yahoo Finance.

---

## How it works

1. **Universe** (`src/universe.py`) — the full list of NYSE common stocks is
   fetched at runtime from the official **NASDAQ Trader symbol directory**
   (`otherlisted.txt`), filtered to Exchange = NYSE and excluding ETFs, test
   issues and non-common securities (warrants, units, preferreds). No tickers
   are hard-coded; the universe reflects whatever is currently listed.
2. **Data** (`src/data.py`) — adjusted close prices are downloaded from Yahoo
   Finance via `yfinance` in batches, aligned on a common calendar, and cached
   on disk. Company **names and sectors** for the displayed pairs are pulled
   from Yahoo Finance too (`get_company_info`).
3. **Cointegration screen** (`src/cointegration.py`) — scanning every pair of an
   exchange-sized universe is O(N²) and far too slow, so a cheap **correlation
   pre-filter** (on daily returns) selects the most correlated candidate pairs;
   only those run the **Engle-Granger** cointegration test (`statsmodels`).
   Pairs are ranked by ascending p-value — lower means stronger evidence of a
   stationary linear combination.
4. **Kalman filter** (`src/kalman.py`) — for each surviving pair we model the
   hedge ratio as a slowly varying hidden state and recover it with a two-state
   Kalman filter (a time-varying linear regression). The filtered residual is
   the **smoothed spread**; its rolling z-score is the trading signal, and we
   also report the mean-reversion **half-life**.
5. **Backtest** (`src/backtest.py`) — the selected pair is backtested with a
   dollar-neutral z-score mean-reversion strategy starting from **$100,000**:
   enter when the spread's z-score is stretched, exit as it reverts. Reports the
   equity curve, total return, CAGR, Sharpe, max drawdown, trades and win rate.
6. **Dashboard** (`app.py`) — ranks the top 10 pairs in a table (with stock
   names) and renders an interactive Plotly drill-down (prices, dynamic β,
   z-score with ±2 bands) plus the backtest equity curve and drawdown.

## Teck Resources dashboard (second page)

A second Streamlit page, **⛏️ Teck Resources** (`pages/1_Teck_Resources.py`),
web-scrapes and visualises the performance and *predictors* of Teck Resources
(NYSE: TECK), a diversified base-metals miner:

- **Web-scraped data** (`src/teck.py`) — Teck's price history and a basket of
  commodity/macro **predictor** proxies (copper miners, metals & mining, energy,
  gold, the US dollar, Canadian equities, the S&P 500) are scraped from Stooq's
  CSV endpoint (scrape-friendly, rarely rate-limited), with a yfinance fallback.
  Recent **headlines** are scraped from the Google News RSS feed.
- **Performance view** — price with 50/200-day moving averages and volume, key
  statistics (returns over 1M/3M/YTD/1Y, 52-week range, volatility), and growth
  of $100 in Teck versus each predictor.
- **Predictor analysis** — pairwise return correlations, a standardised
  multi-factor **OLS model** (with R², showing how much of Teck's daily variance
  the commodity/macro factors explain), and a rolling correlation with the
  single strongest driver. For a miner these factors dominate company news, so
  the model typically shows copper/metals as the leading predictors.

Because it's a multipage app, both dashboards run from the same `streamlit run
app.py` — switch pages from the sidebar.

### Why a Kalman filter?

The textbook spread, `y − β·x`, assumes a single static hedge ratio estimated
once by OLS. Real relationships drift, so a stale β makes a genuinely
cointegrated spread look non-stationary. The Kalman filter lets β adapt bar by
bar:

```
Observation:  y_t = β_t · x_t + α_t + e_t,           e_t ~ N(0, R)
State:        [α_t, β_t] = [α_{t-1}, β_{t-1}] + w_t,  w_t ~ N(0, Q)
```

The filtered residual `e_t` is the smoothed, mean-reverting spread.

---

## Deploy from your browser (no terminal needed)

This app is built to host on **[Streamlit Community Cloud](https://streamlit.io/cloud)**
straight from this GitHub repo — everything below is done in a browser, so it
works fine from an iPad with no local setup:

1. Go to **<https://share.streamlit.io>** and click **Sign in with GitHub**,
   authorising access to this repository.
2. Click **Create app → Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `teamtracy2026/TeamTracy`
   - **Branch:** `claude/nyse-cointegrated-pairs-dashboard-ejohk9` (or `main`
     once merged)
   - **Main file path:** `app.py`
4. (Optional) Under **Advanced settings**, set **Python version** to `3.11`.
5. Click **Deploy**. Streamlit installs `requirements.txt` and launches the app,
   giving you a public `https://…streamlit.app` URL.

After that, **every push to the selected branch auto-redeploys** the app — no
terminal, ever. GitHub Actions (`.github/workflows/ci.yml`) runs the test suite
on each push so you can see green/red checks in the GitHub UI before it deploys.

## Run locally (optional)

```bash
# 1. Install dependencies (a virtualenv is recommended)
pip install -r requirements.txt

# 2. Launch the dashboard
streamlit run app.py
```

Then open the local URL Streamlit prints (default <http://localhost:8501>).
Use the sidebar to choose the history window, scan the whole NYSE or cap its
size, tune the correlation pre-filter and p-value, adjust the Kalman filter and
backtest settings, then press **Run screen**.

### Run the tests

```bash
pip install -r requirements.txt pytest
pytest -q
```

The tests run entirely on synthetic data (no network) and verify that the
Kalman filter recovers a known hedge ratio and that the screener ranks a
genuinely cointegrated pair above independent random walks.

---

## Project layout

```
TeamTracy/
├── app.py                     # Streamlit app — NYSE pairs dashboard (home page)
├── pages/
│   └── 1_Teck_Resources.py    # Teck performance & predictor dashboard
├── requirements.txt
├── src/
│   ├── universe.py            # live NYSE listing (NASDAQ Trader directory)
│   ├── data.py               # Yahoo Finance prices + names, batched + cached
│   ├── kalman.py             # Kalman dynamic hedge ratio + half-life
│   ├── cointegration.py      # correlation pre-filter + Engle-Granger ranking
│   ├── backtest.py           # $100k z-score mean-reversion backtest
│   └── teck.py               # Teck scraping (Stooq/Google News) + predictor model
└── tests/
    ├── test_pipeline.py      # cointegration/Kalman/backtest unit tests
    └── test_teck.py          # Teck parsing + predictor-analysis tests
```

## Configuration

| Environment variable      | Default  | Purpose                                   |
| ------------------------- | -------- | ----------------------------------------- |
| `TEAMTRACY_CACHE_DIR`     | `.cache` | Where downloaded prices/listing are cached. |
| `TEAMTRACY_CACHE_TTL`     | `21600`  | Price cache freshness in seconds (6h).    |
| `TEAMTRACY_UNIVERSE_TTL`  | `86400`  | NYSE listing cache freshness (24h).       |
| `TEAMTRACY_BATCH_SIZE`    | `100`    | Tickers per Yahoo Finance batch download. |

---

## Notes & caveats

- Scanning the **entire NYSE** is thorough but heavy: the first run downloads
  history for every listed common stock, which can take a few minutes. Results
  are cached, so subsequent runs are fast. The sidebar lets you cap the universe
  size and tune the correlation pre-filter for a quicker scan. The dashboard
  needs outbound internet to Yahoo Finance and the NASDAQ Trader directory.
- The backtest is a deliberately simple, transparent model (dollar-neutral
  sizing, one-bar execution lag, a basis-point cost) — not a production engine.
  Returns are in-sample and ignore slippage, borrow costs and capacity.
- Cointegration is estimated in-sample; relationships break down. Treat the
  output as a research starting point.
- **Research / educational use only. Not investment advice.**
