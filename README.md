# 📈 TeamTracy — NYSE Cointegrated Pairs Dashboard

A Streamlit dashboard that finds the **top 10 best cointegrated stock pairs on
the NYSE**, using **Yahoo Finance** price data and a **Kalman filter** to
estimate a time-varying hedge ratio and smooth each pair's spread over time.

Cointegrated pairs are the raw material of statistical-arbitrage / pairs-trading
strategies: two stocks whose prices wander individually but whose *spread* is
mean-reverting. The dashboard screens a curated universe of liquid NYSE names,
ranks pairs by the strength of their cointegration, and lets you drill into each
one's prices, dynamic hedge ratio, smoothed spread and trading-signal z-score.

---

## How it works

1. **Universe** (`src/universe.py`) — a curated set of liquid, NYSE-listed large
   caps grouped by sector (banks, oil, healthcare, retail, …). Screening within
   sectors keeps the combinatorics manageable and surfaces economically
   meaningful relationships.
2. **Data** (`src/data.py`) — adjusted close prices are downloaded from Yahoo
   Finance via `yfinance`, aligned on a common calendar, and cached on disk so
   the dashboard stays responsive.
3. **Cointegration screen** (`src/cointegration.py`) — every candidate pair is
   run through the **Engle-Granger** cointegration test (`statsmodels`). Pairs
   are ranked by ascending p-value; the lower the p-value, the stronger the
   evidence of a stationary linear combination.
4. **Kalman filter** (`src/kalman.py`) — for each surviving pair we model the
   hedge ratio as a slowly varying hidden state and recover it with a two-state
   Kalman filter (a time-varying linear regression). The filtered residual is
   the **smoothed spread**; its rolling z-score is the trading signal, and we
   also report the mean-reversion **half-life**.
5. **Dashboard** (`app.py`) — ranks the top 10 pairs in a table and renders an
   interactive Plotly drill-down (prices, dynamic β, z-score with ±2 bands).

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
Use the sidebar to choose the history window, restrict to within-sector pairs,
set the maximum p-value, and tune the Kalman filter, then press **Run screen**.

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
├── app.py                  # Streamlit dashboard
├── requirements.txt
├── src/
│   ├── universe.py         # curated NYSE universe by sector
│   ├── data.py             # Yahoo Finance loading + caching
│   ├── kalman.py           # Kalman dynamic hedge ratio + half-life
│   └── cointegration.py    # Engle-Granger screen + ranking
└── tests/
    └── test_pipeline.py    # synthetic-data unit tests
```

## Configuration

| Environment variable     | Default  | Purpose                              |
| ------------------------ | -------- | ------------------------------------ |
| `TEAMTRACY_CACHE_DIR`    | `.cache` | Where downloaded prices are cached.  |
| `TEAMTRACY_CACHE_TTL`    | `21600`  | Cache freshness in seconds (6h).     |

---

## Notes & caveats

- The NYSE universe is intentionally a curated large-cap subset, not the full
  exchange listing — this keeps the all-pairs sweep fast and the statistics
  meaningful. Extend `src/universe.py` to widen it (keep names NYSE-listed).
- Cointegration is estimated in-sample; relationships break down. Treat the
  output as a research starting point.
- **Research / educational use only. Not investment advice.**
