# Market Predictor (PredictAStock)

This repository contains a small research/prototype application that trains per-horizon models to predict stock actions and exposes the predictions to a static UI via JSON dumps.

The goal: produce per-ticker predictions (buy/sell/hold) across several time horizons, combine those predictions with recent price data, and provide a simple single-page UI that displays watchlist tables and market summary cards. The UI is intentionally static-first: it loads pre-generated JSON from the `json/` folder so the frontend can be hosted without a running server.

---

Table of contents

- What this project does
- Repository layout
- How data flows and how to (re)generate JSON
- Front-end (index.html) behavior and UI features
- Optional API server (Flask)
- Dependencies and environment notes
- Development tips, tests, and next steps

---

What this project does
----------------------

- Trains models and produces `results.json` (historical predictions produced by `decision_tree.py`).
- Collects latest price data (via `yfinance` when available, or from historical CSV files), merges prices with model predictions, and writes static JSON files to `json/` for the UI to consume. (`stocks.py` provides a generator that writes `json/stocks.json` and there's another dump `json/latest_with_preds.json`.)
- The UI (`index.html`) loads static JSON (prefers `json/latest_with_preds.json` then `json/stocks.json`) and displays a watchlist table with a timeframe toggle, per-row open/close/change, prediction (BUY/SELL/HOLD), and three summary cards (Market Overview, Prediction Snapshot, Economic Indicators).

Repository layout
-----------------

Top-level files you should know:

- `index.html` — Single-file Vue 3 + Vuetify UI. Loads JSON from the `json/` folder and renders the watchlist and summary cards. Includes a Night (dark) mode, timeframe toggle, loading indicator, and small UX niceties.
- `stocks.py` — Script that fetches the latest S&P500 price (via `yfinance` when available, falls back to `data/SP500.csv`), loads `results.json` (if present), and writes `json/stocks.json` as a merged artifact. Run this to generate static JSON for the UI.
- `decision_tree.py` — (Model pipeline) Feature engineering, model training/backtest, and saving model artifacts and `results.json` for predictions. This is the core modeling pipeline; run it to produce or re-train models and recreate `results.json`.
- `api_routes.py` — Optional Flask API used during development to serve predictions and to generate the `json/` folder programmatically. The UI no longer requires the server (the UI prefers static JSON), but the API is still present if you want HTTP endpoints.
- `data/` — CSV data files used by the pipeline (example: `SP500.csv`, `SP500_macro.csv`, `OIL.csv`, `INFLATION.csv`, `GDP.csv`). Keep these under version control if you rely on them.
- `json/` — Generated runtime artifacts for the UI. Two notable files used by the UI:
  - `latest_with_preds.json` — array of rows with `Ticker`, `Open`, `Close`, `Date`, `Pred_1d`, `Pred_1m`, `Pred_1y`, `Pred_5y`, etc. This file provides direct Open/Close values and is preferred by the UI.
  - `stocks.json` — object of the form `{ sp500: {...}, predictions: [...] }`. The UI will read `predictions` from this file if `latest_with_preds.json` is not present.

How data flows and how to (re)generate JSON
------------------------------------------

High level:

1. Run the modeling pipeline (`decision_tree.py`) to (re)train models and produce `results.json`.
2. Run `stocks.py` to fetch latest prices (uses `yfinance` where available) and merge those prices with `results.json`; this writes `json/stocks.json`.
3. Optionally you may run other scripts which produce `json/latest_with_preds.json` (this file is preferred by the UI since it contains Open/Close fields and prediction flags ready-to-show).

Commands

From the project root:

```bash
# (optional) create virtualenv and activate
python -m venv .venv
source .venv/bin/activate

# install recommended packages (see dependencies below)
pip install -r requirements.txt

# regenerate model predictions (slow; does training/backtest)
python decision_tree.py

# generate a UI-ready JSON (fetches latest prices via yfinance if available)
python stocks.py
```

Notes:
- `stocks.py` will look for `results.json` in the project root. If you don't have `results.json`, the `predictions` field in `json/stocks.json` will be empty.
- `stocks.py` prefers `yfinance` to fetch the latest S&P500 price. If `yfinance` is unavailable or fails, it falls back to `data/SP500.csv`.

Front-end (index.html) behavior and UI features
----------------------------------------------

- Static-first: `index.html` tries to fetch `./json/latest_with_preds.json` first, then `./json/stocks.json`. If neither exists the UI shows a friendly message and a Retry button.
- Watchlist table: symbol, open, close, change (close - open), and a single prediction column. Use the timeframe toggle (1d / 1m / 1y / 5y) to switch which Pred_* or Action_* value is shown.
- Prediction normalization: the UI normalizes several shapes found in JSON (boolean Pred_*, numeric Pred_* with 0/1, string Action_* like BUY/SELL/HOLD, or Combined_Action) and renders BUY / SELL / HOLD.
- Coloring: BUY is green, SELL is red, HOLD is amber. Change is colored green/red based on sign.
- Summary cards (computed client-side from the loaded JSON):
  - Market Overview: top gainers/losers and a simple trend computed from averaged percent change.
  - Prediction Snapshot: buy share (fraction of rows labeled BUY for 1d), expected next day % (avg close/open %), volatility index (stddev).
  - Economic Indicators: derived heuristics; the UI will try to use `sp500.last_close` from `stocks.json` where present or fall back to average close.
- Night mode: there is a Night toggle in the app bar. We use a compact CSS override approach (lightweight) to style dark mode; preferences persist in sessionStorage.

Optional API server (Flask)
--------------------------

There is an optional Flask-based server (`api_routes.py`) which was used during development. It exposes endpoints such as `/api/predict` and `/api/generate_json` and can serve files under `/json/`.

You do not need this server to run the UI — the UI is designed to be static-first — but if you want the server features:

```bash
# install Flask
pip install flask flask-cors
python api_routes.py
# by default the server listens on 127.0.0.1:5001 in development code
```

When running, `/api/generate_json` can produce `json/latest_with_preds.json` by fetching prices and merging with `results.json`.

Dependencies and environment notes
---------------------------------

- Python 3.10+ recommended (project used 3.13 during development). Use a virtualenv.
- Key Python packages used in the repo:
  - pandas, numpy — data handling
  - scikit-learn, xgboost — modeling (xgboost optional; code has fallbacks)
  - joblib — model serialization
  - yfinance — fetch live prices (optional; used by `stocks.py`)
  - flask, flask-cors — optional API server

If you intend to run model training with `xgboost`, macOS users may need to install `libomp` (OpenMP) before installing `xgboost`:

```bash
# macOS (Homebrew)
brew install libomp
pip install xgboost
```

Minimal `requirements.txt` suggestion (create/adjust as needed):

```
pandas
numpy
scikit-learn
xgboost
joblib
requests
yfinance
flask
flask-cors
```

Development tips and next steps
------------------------------

- If you want the UI to always show up-to-date JSON, add a scheduled job (cron, launchd, or APScheduler inside `api_routes.py`) to regenerate the JSON periodically (`stocks.py` or `/api/generate_json`).
- Consider moving UI assets into a simple static server (Nginx, GitHub Pages) and pre-generate `json/latest_with_preds.json` on a separate job — this makes the UI low-cost to host and reduces rate-limit pressure on external APIs.
- Add unit tests for `stocks.py` and a small integration test that runs the generator and confirms `json/stocks.json` or `json/latest_with_preds.json` exists and contains expected keys.
- If you prefer a more integrated theming approach, convert the lightweight CSS-based dark mode to Vuetify's theme system (update `Vuetify.createVuetify({ theme: {...} })`). That will give more consistent component styling.

Contact / Maintainers
---------------------

If you are continuing this project, keep the README updated with any changes to the data shapes or JSON files. The UI reads two canonical JSON shapes — keep backwards compatibility if possible (Pred_*/Action_* fields and Combined_Action/Final_Action fallback).

License
-------
This repository is provided as-is (add your chosen license here).
