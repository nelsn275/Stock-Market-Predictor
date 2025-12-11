# api_routes.py

import io
import requests
import pandas as pd
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from datetime import datetime
import joblib
import numpy as np
import os
import logging

app = Flask(__name__)
CORS(app)

# logging to file for easier debugging of server-side errors
logging.basicConfig(
    filename='api.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s'
)
logger = logging.getLogger(__name__)

API_KEY = "pbuLIX4fITAQYW0redFM9cLuRvKnLpyo"
FRED_API_KEY = "cfd0fcba89040180b67e70fc1f026139"

# Horizons and weights
HORIZONS = {"1d": 1, "1m": 21, "1y": 252, "5y": 1260}
HORIZON_WEIGHTS = {"1d": 0.5, "1m": 0.5, "1y": 1.5, "5y": 2.0}

# ----------------------------------
# Load Macro History + Models
# ----------------------------------
hist_path = "data/SP500_macro.csv"
if os.path.exists(hist_path):
    try:
        historical_macro = pd.read_csv(hist_path, parse_dates=["Date"])
    except Exception:
        logger.exception(f"Failed to read {hist_path}; starting with empty historical_macro DataFrame")
        historical_macro = pd.DataFrame(columns=["Date", "Ticker", "Close", "Open", "Oil", "Inflation", "GDP"])
        historical_macro["Date"] = pd.to_datetime(historical_macro["Date"])
else:
    logger.warning(f"{hist_path} not found; starting with empty historical_macro DataFrame")
    historical_macro = pd.DataFrame(columns=["Date", "Ticker", "Close", "Open", "Oil", "Inflation", "GDP"])
    historical_macro["Date"] = pd.to_datetime(historical_macro["Date"])

# Load each model from /models/
models = {}
features = {}

# Cache results.json in memory to avoid re-reading on every request and speed up responses
cached_results_df = None
if os.path.exists("results.json"):
    try:
        import json as _json
        _saved = _json.load(open("results.json"))
        cached_results_df = pd.DataFrame(_saved)
        # normalize to Pred_* and Final_Action similar to request-time logic
        for _h in HORIZONS.keys():
            _act_col = f"Action_{_h}"
            _pred_col = f"Pred_{_h}"
            if _act_col in cached_results_df.columns:
                cached_results_df[_pred_col] = cached_results_df[_act_col].map({"BUY":1, "SELL":0, "HOLD":0}).fillna(0).astype(int)
            elif _pred_col in cached_results_df.columns:
                cached_results_df[_pred_col] = cached_results_df[_pred_col].fillna(0).astype(int)
            else:
                cached_results_df[_pred_col] = 0
        if "Combined_Action" in cached_results_df.columns:
            cached_results_df["Final_Action"] = cached_results_df["Combined_Action"]
        elif "Final_Action" not in cached_results_df.columns:
            cached_results_df["Final_Action"] = None
    except Exception:
        logger.exception("Failed to preload results.json into cache")
        cached_results_df = None

# Try to load a single aggregated models file first
all_models_path = "models/all_models.pkl"
if os.path.exists(all_models_path):
    try:
        all_models = joblib.load(all_models_path)
        # all_models may be a dict mapping horizon -> (model, feature_cols)
        if isinstance(all_models, dict):
            for h in HORIZONS.keys():
                val = all_models.get(h)
                if val is None:
                    continue
                # support either tuple (model, features) or dict with keys
                if isinstance(val, (list, tuple)) and len(val) >= 2:
                    models[h], features[h] = val[0], val[1]
                elif isinstance(val, dict) and "model" in val and "features" in val:
                    models[h], features[h] = val["model"], val["features"]
    except Exception:
        # fall back to per-horizon files
        pass

# Load per-horizon model files if any horizons missing
for h in HORIZONS.keys():
    if h in models and h in features:
        continue
    # Try several filename patterns that might exist in the repo
    tried = []
    candidates = [f"models/{h}.pkl", f"models/model_{h}.pkl", f"models/{h}_model.pkl"]
    for p in candidates:
        tried.append(p)
        if os.path.exists(p):
            try:
                loaded = joblib.load(p)
                # file might contain a tuple (model, features) or a dict
                if isinstance(loaded, dict) and "model" in loaded and "features" in loaded:
                    models[h] = loaded["model"]
                    features[h] = loaded["features"]
                elif isinstance(loaded, (list, tuple)) and len(loaded) >= 2:
                    models[h], features[h] = loaded[0], loaded[1]
                else:
                    # assume the pickle is the model object and look for features file
                    models[h] = loaded
                    fpath = f"models/{h}_features.pkl"
                    if os.path.exists(fpath):
                        features[h] = joblib.load(fpath)
                break
            except Exception:
                continue
    # if still missing features, try alternate features filename
    if h in models and h not in features:
        for fpat in [f"models/{h}_features.pkl", f"models/features_{h}.pkl", f"models/{h}_feat.pkl"]:
            if os.path.exists(fpat):
                try:
                    features[h] = joblib.load(fpat)
                    break
                except Exception:
                    continue

# At this point models/features may be partially populated. The API will raise if a horizon is requested without a model.


# ----------------------------------
# Polygon Most Recent Trading Day
# ----------------------------------
def get_most_recent_trading_day():
    url = f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/last?apiKey={API_KEY}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        # Return None to the caller so it can handle the failure gracefully
        logger.exception("Polygon request failed")
        return None, {"error": f"Polygon request failed: {str(e)}"}

    # Polygon responses can vary — be defensive when accessing keys
    results = data.get("results") if isinstance(data, dict) else None
    # queryCount may not always be present; fall back to None
    query_count = data.get("queryCount") if isinstance(data, dict) else None
    return query_count, results


# ----------------------------------
# Pull S&P 500 Components
# ----------------------------------
def get_stocks():
    """Use yfinance as the primary live price source. Return (stocks_df, data_source).

    If yfinance is not available or fails, fall back to the latest rows from historical_macro.
    """
    # Determine tickers to fetch: prefer cached results.json tickers, otherwise fallback to historical_macro tickers
    tickers = []
    try:
        if cached_results_df is not None and not cached_results_df.empty and "Ticker" in cached_results_df.columns:
            tickers = list(cached_results_df["Ticker"].unique())
    except Exception:
        tickers = []

    if not tickers and not historical_macro.empty:
        try:
            tickers = historical_macro.sort_values(["Ticker", "Date"]).groupby("Ticker").tail(1)["Ticker"].tolist()
        except Exception:
            tickers = []

    if not tickers:
        # No tickers available; return empty DataFrame and indicate cached/historical as source
        last = historical_macro.sort_values(["Ticker", "Date"]).groupby("Ticker").tail(1)
        stocks = last[["Ticker", "Close", "Open"]].copy()
        stocks["Date"] = pd.Timestamp.today().normalize()
        logger.warning("No tickers found; returning historical latest per-ticker prices")
        return stocks, "historical"

    # Limit to a reasonable number to avoid long downloads
    fetch_list = tickers[:200]

    try:
        import yfinance as yf
        df = yf.download(tickers=fetch_list, period="2d", interval="1d", group_by="ticker", threads=True, progress=False, auto_adjust=True)
        rows = []
        today_dt = pd.Timestamp.today().normalize()

        if isinstance(df.columns, pd.MultiIndex):
            for tk in fetch_list:
                try:
                    sub = df[tk].dropna(how='all')
                    if sub.shape[0] == 0:
                        continue
                    last_row = sub.iloc[-1]
                    rows.append({"Ticker": tk, "Open": float(last_row.get("Open", 0)), "Close": float(last_row.get("Close", 0)), "Date": today_dt})
                except Exception:
                    continue
        else:
            # single ticker scenario
            try:
                if not df.empty:
                    last_row = df.iloc[-1]
                    rows.append({"Ticker": fetch_list[0], "Open": float(last_row.get("Open", 0)), "Close": float(last_row.get("Close", 0)), "Date": today_dt})
            except Exception:
                rows = []

        if rows:
            stocks = pd.DataFrame(rows)
            logger.info(f"Fetched {len(stocks)} tickers via yfinance")
            return stocks, "yfinance"
        else:
            logger.warning("yfinance returned no usable rows; falling back to historical_macro")

    except ImportError:
        logger.warning("yfinance not installed; please install it to enable live prices: pip install yfinance")
    except Exception:
        logger.exception("yfinance fetch failed; falling back to historical_macro")

    # Final fallback: historical_macro
    last = historical_macro.sort_values(["Ticker", "Date"]).groupby("Ticker").tail(1)
    stocks = last[["Ticker", "Close", "Open"]].copy()
    stocks["Date"] = pd.Timestamp.today().normalize()
    return stocks, "historical"


# ----------------------------------
# FRED Fetchers
# ----------------------------------
def get_oil_price_fred():
    url = "https://api.stlouisfed.org/fred/series/observations"
    r = requests.get(url, params={
        "series_id": "DCOILWTICO",
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 2
    }).json()
    try:
        obs_list = r.get("observations") if isinstance(r, dict) else None
        if not obs_list:
            return None
        obs = [x for x in obs_list if x.get("value") not in (None, ".")]
        return float(obs[0]["value"]) if obs else None
    except Exception:
        logger.exception("Failed to fetch/parse oil price from FRED")
        return None


def get_inflation_fred():
    url = "https://api.stlouisfed.org/fred/series/observations"
    r = requests.get(url, params={
        "series_id": "CPIAUCSL",
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 2
    }).json()
    try:
        obs_list = r.get("observations") if isinstance(r, dict) else None
        if not obs_list or len(obs_list) < 2:
            return None
        obs = [x for x in obs_list if x.get("value") not in (None, ".")]
        if len(obs) < 2:
            return None
        return (float(obs[0]["value"]) - float(obs[1]["value"])) / float(obs[1]["value"])
    except Exception:
        logger.exception("Failed to fetch/parse inflation from FRED")
        return None


def get_gdp_worldbank():
    url = "https://api.worldbank.org/v2/country/us/indicator/NY.GDP.MKTP.CD?format=json"
    r = requests.get(url).json()
    try:
        return r[1][0]["value"]
    except:
        return None


# ----------------------------------
# PREDICT ENDPOINT
# ----------------------------------
@app.route("/api/predict")
def predict_stocks():
    try:
        # Latest prices (stocks DataFrame and data source)
        latest, data_source = get_stocks()

        # Prefer live prices and then enrich them from previously computed results
        # Pull macro values for display (oil, inflation, GDP)
        today = latest.copy() if latest is not None else pd.DataFrame()
        today["Oil"] = get_oil_price_fred()
        today["Inflation"] = get_inflation_fred()
        today["GDP"] = get_gdp_worldbank()

        # Try to attach precomputed predictions from results.json (preferred)
        # Use cached results_df if preloaded at startup, otherwise attempt to read file once
        preds_df = None
        if cached_results_df is not None:
            saved_df = cached_results_df
        else:
            import json
            if os.path.exists("results.json"):
                try:
                    with open("results.json", "r") as fh:
                        saved = json.load(fh)
                    saved_df = pd.DataFrame(saved)
                except Exception:
                    logger.exception("Failed to parse results.json; will attempt to use live features or historical fallback")
                    saved_df = None
            else:
                saved_df = None
        # Normalize columns to Pred_* and Final_Action if saved_df exists
        if saved_df is not None and not saved_df.empty:
            for h in HORIZONS.keys():
                act_col = f"Action_{h}"
                pred_col = f"Pred_{h}"
                if act_col in saved_df.columns:
                    saved_df[pred_col] = saved_df[act_col].map({"BUY": 1, "SELL": 0, "HOLD": 0}).fillna(0).astype(int)
                elif pred_col in saved_df.columns:
                    saved_df[pred_col] = saved_df[pred_col].fillna(0).astype(int)
                else:
                    saved_df[pred_col] = 0

            if "Combined_Action" in saved_df.columns:
                saved_df["Final_Action"] = saved_df["Combined_Action"]
            elif "Final_Action" not in saved_df.columns:
                saved_df["Final_Action"] = None

            keep_cols = ["Ticker"] + [f"Pred_{h}" for h in HORIZONS.keys()] + ["Final_Action"]
            preds_df = saved_df[[c for c in keep_cols if c in saved_df.columns]].copy()

        # Merge precomputed preds into live latest rows when available
        if preds_df is not None and not preds_df.empty:
            # If we couldn't pull live prices, return predictions directly from results.json
            if latest is None or latest.empty:
                preds_payload = []
                for _, row in preds_df.iterrows():
                    obj = {"Ticker": row.get("Ticker")}
                    for h in HORIZONS.keys():
                        obj[f"Pred_{h}"] = int(row.get(f"Pred_{h}", 0)) if pd.notna(row.get(f"Pred_{h}")) else 0
                    obj["Final_Action"] = row.get("Final_Action") if pd.notna(row.get("Final_Action")) else None
                    preds_payload.append(obj)

                # Attempt to compute a lightweight UI summary using historical_macro if available
                if not historical_macro.empty:
                    last_prices = historical_macro.sort_values(["Ticker", "Date"]).groupby("Ticker").tail(1)
                    if not last_prices.empty and "Open" in last_prices.columns and "Close" in last_prices.columns:
                        last_prices["change_pct"] = (last_prices["Close"] - last_prices["Open"]) / last_prices["Open"]
                        top_gainers = last_prices.nlargest(3, "change_pct")["Ticker"].tolist()
                        top_losers = last_prices.nsmallest(3, "change_pct")["Ticker"].tolist()
                        trend = "Bullish" if last_prices["change_pct"].mean() > 0 else "Bearish"
                        vol_index = float(last_prices["change_pct"].std() * 100)
                    else:
                        top_gainers, top_losers, trend, vol_index = [], [], "Neutral", 0.0
                else:
                    top_gainers, top_losers, trend, vol_index = [], [], "Neutral", 0.0

                ui_summary = {
                    "market_overview": {"top_gainers": top_gainers, "top_losers": top_losers, "trend": trend},
                    "prediction_snapshot": {"buy_share_1d": float(preds_df["Pred_1d"].mean()) if "Pred_1d" in preds_df.columns else 0.0,
                                              "expected_next_day_pct": 0.0,
                                              "volatility_index": vol_index},
                    "economic_indicators": {"Oil": None, "Inflation": None, "GDP": None}
                }

                return jsonify({"predictions": preds_payload, "summary": ui_summary, "data_source": "cached"})

            # Otherwise merge preds into live rows
            latest = latest.merge(preds_df, on="Ticker", how="left")
            for h in HORIZONS.keys():
                if f"Pred_{h}" in latest.columns:
                    latest[f"Pred_{h}"] = latest[f"Pred_{h}"].fillna(0).astype(int)
                else:
                    latest[f"Pred_{h}"] = 0
            if "Final_Action" not in latest.columns:
                latest["Final_Action"] = None

            # Build UI summary from live latest + preds
            latest_local = latest.copy()
            if not latest_local.empty and "Open" in latest_local.columns and "Close" in latest_local.columns:
                latest_local["change_pct"] = (latest_local["Close"] - latest_local["Open"]) / latest_local["Open"]
                top_gainers = latest_local.nlargest(3, "change_pct")["Ticker"].tolist()
                top_losers = latest_local.nsmallest(3, "change_pct")["Ticker"].tolist()
                trend = "Bullish" if latest_local["change_pct"].mean() > 0 else "Bearish"
                vol_index = float(latest_local["change_pct"].std() * 100)
            else:
                top_gainers, top_losers, trend, vol_index = [], [], "Neutral", 0.0

            preds_payload = []
            for _, row in latest.iterrows():
                obj = {"Ticker": row.get("Ticker")}
                for h in HORIZONS.keys():
                    obj[f"Pred_{h}"] = int(row.get(f"Pred_{h}", 0)) if pd.notna(row.get(f"Pred_{h}")) else 0
                obj["Final_Action"] = row.get("Final_Action") if pd.notna(row.get("Final_Action")) else None
                preds_payload.append(obj)

            ui_summary = {
                "market_overview": {"top_gainers": top_gainers, "top_losers": top_losers, "trend": trend},
                "prediction_snapshot": {"buy_share_1d": float(latest[f"Pred_1d"].mean()) if f"Pred_1d" in latest.columns else 0.0,
                                          "expected_next_day_pct": 0.0,
                                          "volatility_index": vol_index},
                "economic_indicators": {"Oil": float(today["Oil"].iloc[0]) if ("Oil" in today.columns and len(today)>0 and pd.notna(today["Oil"].iloc[0])) else None,
                                         "Inflation": float(today["Inflation"].iloc[0]) if ("Inflation" in today.columns and len(today)>0 and pd.notna(today["Inflation"].iloc[0])) else None,
                                         "GDP": float(today["GDP"].iloc[0]) if ("GDP" in today.columns and len(today)>0 and pd.notna(today["GDP"].iloc[0])) else None}
            }

            return jsonify({"predictions": preds_payload, "summary": ui_summary, "data_source": data_source})

        # If we reach here, results.json wasn't available — fall back to rebuilding features from historical_macro + live today
        # Build updated macro DF
        macro = pd.concat([historical_macro, today], ignore_index=True)
        macro = macro.sort_values(["Ticker", "Date"])

        # ---- REBUILD SAME FEATURES AS PIPELINE ----
        macro["Return_1d"] = macro.groupby("Ticker")["Close"].pct_change(1)
        for lb in [1, 3, 5]:
            macro[f"Return_{lb}d"] = macro.groupby("Ticker")["Close"].pct_change(lb)
            macro[f"MA_{lb}"] = macro.groupby("Ticker")["Close"].transform(lambda x: x.rolling(lb).mean())
            macro[f"Vol_{lb}d"] = macro.groupby("Ticker")["Return_1d"].transform(
                lambda x: x.rolling(lb).std()
            )

        macro["Oil_Return_1d"] = macro["Oil"].pct_change()
        macro["Inflation_Change"] = macro["Inflation"].diff()
        macro["GDP_Growth"] = macro["GDP"].diff()

        macro["MA_21"] = macro.groupby("Ticker")["Close"].transform(lambda x: x.rolling(21).mean())
        macro["Vol_21d"] = macro.groupby("Ticker")["Return_1d"].transform(lambda x: x.rolling(21).std())
        macro["MA_252"] = macro.groupby("Ticker")["Close"].transform(lambda x: x.rolling(252).mean())
        macro["Vol_252d"] = macro.groupby("Ticker")["Return_1d"].transform(lambda x: x.rolling(252).std())
        macro["Trend_90d"] = macro.groupby("Ticker")["Close"].pct_change(90)

        # Today's rows
        today_date = pd.Timestamp.today().normalize()
        today_features = macro[macro["Date"] == today_date].copy()

        # ---- PREDICT PER HORIZON ----
        for h in HORIZONS.keys():
            # Defensive: if model/features not available, default to 0 (SELL)
            if h not in models or h not in features:
                today_features[f"Pred_{h}"] = 0
                continue

            fcols = features[h]
            # Ensure feature columns exist on the DataFrame; if missing, create with zeros
            for c in fcols:
                if c not in today_features.columns:
                    today_features[c] = 0

            # Coerce feature columns to numeric (XGBoost requires numeric dtypes)
            try:
                today_features[fcols] = today_features[fcols].apply(pd.to_numeric, errors='coerce').fillna(0)
            except Exception:
                # worst-case: ensure no non-numeric objects remain
                for c in fcols:
                    today_features[c] = pd.to_numeric(today_features[c], errors='coerce').fillna(0)

            # Run prediction and guard against model errors
            try:
                today_features[f"Pred_{h}"] = models[h].predict(today_features[fcols])
            except Exception as e:
                # fallback: log full traceback and set default predictions
                logger.exception(f"Prediction error for horizon {h}")
                today_features[f"Pred_{h}"] = 0

        # If there were no live rows today (Polygon down), fall back to the latest historical rows
        if today_features.empty:
            logger.info("No live today rows; using historical_macro latest per-ticker as fallback for predictions")
            fallback = historical_macro.sort_values(["Ticker", "Date"]).groupby("Ticker").tail(1).copy()
            # ensure necessary feature columns exist on fallback
            for h in HORIZONS.keys():
                fcols = features.get(h, [])
                if not fcols:
                    fallback[f"Pred_{h}"] = 0
                    continue
                for c in fcols:
                    if c not in fallback.columns:
                        fallback[c] = 0
                # coerce to numeric
                try:
                    fallback[fcols] = fallback[fcols].apply(pd.to_numeric, errors='coerce').fillna(0)
                except Exception:
                    for c in fcols:
                        fallback[c] = pd.to_numeric(fallback[c], errors='coerce').fillna(0)
                # predict
                try:
                    fallback[f"Pred_{h}"] = models[h].predict(fallback[fcols]) if h in models else 0
                except Exception:
                    logger.exception(f"Fallback prediction error for horizon {h}")
                    fallback[f"Pred_{h}"] = 0

            # compute Final_Action on fallback
            score = pd.Series(0, index=fallback.index, dtype=float)
            for h in HORIZONS.keys():
                score += fallback[f"Pred_{h}"].fillna(0).astype(float) * HORIZON_WEIGHTS[h]
            t1 = sum(HORIZON_WEIGHTS.values()) * 0.33
            t2 = sum(HORIZON_WEIGHTS.values()) * 0.66
            fallback["Final_Action"] = pd.Series(np.where(score >= t2, "BUY", np.where(score >= t1, "HOLD", "SELL")), index=fallback.index)

            # Build predictions payload from fallback
            preds = []
            if fallback.empty:
                # As a last resort, load previously saved results.json produced by the pipeline
                try:
                    import json
                    with open('results.json', 'r') as fh:
                        saved = json.load(fh)
                    for r in saved:
                        obj = {"Ticker": r.get("Ticker")}
                        # map historical actions to Pred_* booleans when possible
                        for h in HORIZONS.keys():
                            key = f"Action_{h}"
                            if key in r:
                                obj[f"Pred_{h}"] = 1 if r.get(key) == "BUY" else 0
                            else:
                                obj[f"Pred_{h}"] = 0
                        obj["Final_Action"] = r.get("Combined_Action") or r.get("Final_Action")
                        preds.append(obj)
                except Exception:
                    preds = []
            else:
                for _, row in fallback.iterrows():
                    obj = {"Ticker": row.get("Ticker")}
                    for h in HORIZONS.keys():
                        obj[f"Pred_{h}"] = int(row.get(f"Pred_{h}", 0)) if not pd.isna(row.get(f"Pred_{h}")) else 0
                    obj["Final_Action"] = row.get("Final_Action")
                    preds.append(obj)

            # summary from fallback
            latest_local = fallback.copy()
            if not latest_local.empty and "Open" in latest_local.columns and "Close" in latest_local.columns:
                latest_local["change_pct"] = (latest_local["Close"] - latest_local["Open"]) / latest_local["Open"]
                top_gainers = latest_local.nlargest(3, "change_pct")["Ticker"].tolist()
                top_losers = latest_local.nsmallest(3, "change_pct")["Ticker"].tolist()
                trend = "Bullish" if latest_local["change_pct"].mean() > 0 else "Bearish"
                vol_index = float(latest_local["change_pct"].std() * 100)
            else:
                top_gainers, top_losers, trend, vol_index = [], [], "Neutral", 0.0

            economic = {
                "Oil": None,
                "Inflation": None,
                "GDP": None,
                "Unemployment": None
            }

            ui_summary = {
                "market_overview": {"top_gainers": top_gainers, "top_losers": top_losers, "trend": trend},
                "prediction_snapshot": {"buy_share_1d": float(fallback[f"Pred_1d"].mean()) if "Pred_1d" in fallback.columns else 0.0,
                                          "expected_next_day_pct": ((float(fallback[f"Pred_1d"].mean())-0.5)*2.0*100.0) if "Pred_1d" in fallback.columns else 0.0,
                                          "volatility_index": vol_index},
                "economic_indicators": economic
            }

            return jsonify({"predictions": preds, "summary": ui_summary, "data_source": "historical"})

        # ---- COMBINE SIGNALS ----
        score = np.zeros(len(today_features))
        for h in HORIZONS.keys():
            score += today_features[f"Pred_{h}"] * HORIZON_WEIGHTS[h]

        # thresholds (static for API use)
        t1 = sum(HORIZON_WEIGHTS.values()) * 0.33
        t2 = sum(HORIZON_WEIGHTS.values()) * 0.66

        today_features["Final_Action"] = np.where(
            score >= t2, "BUY",
            np.where(score >= t1, "HOLD", "SELL")
        )

        # Build a lightweight UI summary for the front-end cards
        # Use the raw `latest` rows (Polygon response) to compute intraday changes
        latest_local = latest.copy()
        if not latest_local.empty and "Open" in latest_local.columns and "Close" in latest_local.columns:
            latest_local["change_pct"] = (latest_local["Close"] - latest_local["Open"]) / latest_local["Open"]
            top_gainers = latest_local.nlargest(3, "change_pct")["Ticker"].tolist()
            top_losers = latest_local.nsmallest(3, "change_pct")["Ticker"].tolist()
            trend = "Bullish" if latest_local["change_pct"].mean() > 0 else "Bearish"
            vol_index = float(latest_local["change_pct"].std() * 100)
        else:
            top_gainers, top_losers, trend, vol_index = [], [], "Neutral", 0.0

        # Prediction snapshot (simple aggregates)
        pred_1d_col = f"Pred_1d"
        if pred_1d_col in today_features.columns and len(today_features) > 0:
            buy_share_1d = float(today_features[pred_1d_col].mean())
        else:
            buy_share_1d = 0.0

        # crude expected next-day % estimate (not a rigorous forecast)
        expected_next_day_pct = (buy_share_1d - 0.5) * 2.0 * 100.0

        # Economic indicators (take from the `today` frame we attached earlier)
        economic = {
            "Oil": float(today["Oil"].iloc[0]) if ("Oil" in today.columns and len(today) > 0 and pd.notna(today["Oil"].iloc[0])) else None,
            "Inflation": float(today["Inflation"].iloc[0]) if ("Inflation" in today.columns and len(today) > 0 and pd.notna(today["Inflation"].iloc[0])) else None,
            "GDP": float(today["GDP"].iloc[0]) if ("GDP" in today.columns and len(today) > 0 and pd.notna(today["GDP"].iloc[0])) else None,
            "Unemployment": None
        }

        ui_summary = {
            "market_overview": {
                "top_gainers": top_gainers,
                "top_losers": top_losers,
                "trend": trend
            },
            "prediction_snapshot": {
                "buy_share_1d": buy_share_1d,
                "expected_next_day_pct": expected_next_day_pct,
                "volatility_index": vol_index
            },
            "economic_indicators": economic
        }

        # API Output: include both the per-ticker predictions and a small UI summary
        return jsonify({
            "predictions": today_features[["Ticker", "Final_Action"] + [f"Pred_{h}" for h in HORIZONS.keys()]].to_dict(orient="records"),
            "summary": ui_summary,
            "data_source": data_source
        })

    except Exception as e:
        logger.exception("Unhandled error in predict_stocks")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Use a non-conflicting port if 5000 is already in use on macOS (Control Center / AirPlay)
    # Run without the automatic reloader to avoid the server restarting on file changes
    # which can cause the browser or editor live-server to refresh the page repeatedly.
    app.run(debug=False, use_reloader=False, host="127.0.0.1", port=5001)


@app.route('/')
def root_index():
    """Serve the project's index.html for convenience so visiting the API host shows the UI."""
    return send_from_directory('.', 'index.html')


@app.route('/favicon.ico')
def favicon():
    """Serve favicon if present; return 204 if not to avoid noisy 404s in logs."""
    if os.path.exists('favicon.ico'):
        return send_from_directory('.', 'favicon.ico')
    return ('', 204)
