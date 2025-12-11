import os
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
DATA_STOCKS = "data/SP500.csv"
DATA_OIL = "data/OIL.csv"
DATA_INFLATION = "data/INFLATION.csv"
DATA_GDP = "data/GDP.csv"

HORIZONS = {"1d": 1, "1m": 21, "1y": 252, "5y": 1260}
MIN_ROWS_PER_TICKER = 200

# Horizon weights (adjustable)
HORIZON_WEIGHTS = {"1d": 0.5, "1m": 0.5, "1y": 1.5, "5y": 2.0}

# Where models will be saved
os.makedirs("models", exist_ok=True)

# ---------------- DATA FUNCTIONS ----------------
def load_and_merge():
    stocks = pd.read_csv(DATA_STOCKS, parse_dates=["Date"])
    stocks = stocks.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    def load_macro(path, date_col, rename_map):
        df = pd.read_csv(path)
        df[date_col] = pd.to_datetime(df[date_col]).dt.floor("D")
        df.rename(columns=rename_map, inplace=True)
        return df

    oil = load_macro(DATA_OIL, "observation_date", {"observation_date":"Date","DCOILBRENTEU":"Oil"})
    oil["Oil"] = oil["Oil"].replace(".", np.nan).astype(float).ffill()
    inflation = load_macro(DATA_INFLATION, "observation_date", {"observation_date":"Date","CORESTICKM159SFRBATL":"Inflation"})
    gdp = load_macro(DATA_GDP, "observation_date", {"observation_date":"Date","GDP":"GDP"})

    macro = pd.merge_asof(stocks.sort_values("Date"), oil.sort_values("Date"), on="Date", direction="backward")
    macro = pd.merge_asof(macro.sort_values("Date"), inflation.sort_values("Date"), on="Date", direction="backward")
    macro = pd.merge_asof(macro.sort_values("Date"), gdp.sort_values("Date"), on="Date", direction="backward")
    macro = macro.sort_values(["Ticker","Date"]).reset_index(drop=True).ffill()
    return macro

def make_features(macro):
    macro["Return_1d"] = macro.groupby("Ticker")["Close"].pct_change(1)
    for lb in [1,3,5]:
        macro[f"Return_{lb}d"] = macro.groupby("Ticker")["Close"].pct_change(lb)
        macro[f"MA_{lb}"] = macro.groupby("Ticker")["Close"].transform(lambda x: x.rolling(lb).mean())
        macro[f"Vol_{lb}d"] = macro.groupby("Ticker")["Return_1d"].transform(lambda x: x.rolling(lb,min_periods=1).std(ddof=0)).fillna(0)

    macro["Oil_Return_1d"] = macro["Oil"].pct_change().fillna(0)
    macro["Inflation_Change"] = macro["Inflation"].diff().fillna(0)
    macro["GDP_Growth"] = macro["GDP"].diff().fillna(0)

    # Medium and long term features
    macro["MA_21"] = macro.groupby("Ticker")["Close"].transform(lambda x: x.rolling(21).mean())
    macro["Vol_21d"] = macro.groupby("Ticker")["Return_1d"].transform(lambda x: x.rolling(21,min_periods=1).std(ddof=0)).fillna(0)
    macro["MA_252"] = macro.groupby("Ticker")["Close"].transform(lambda x: x.rolling(252).mean())
    macro["Vol_252d"] = macro.groupby("Ticker")["Return_1d"].transform(lambda x: x.rolling(252,min_periods=1).std(ddof=0)).fillna(0)
    macro["Trend_90d"] = macro.groupby("Ticker")["Close"].transform(lambda x: x.pct_change(90))
    macro = macro.ffill().bfill()
    return macro

def create_targets(macro):
    for name, days in HORIZONS.items():
        macro[f"Return_{name}_ahead"] = macro.groupby("Ticker")["Close"].pct_change(-days)
        macro[f"Signal_{name}"] = (macro[f"Return_{name}_ahead"] > 0).astype(int)
    max_lookahead = max(HORIZONS.values())
    macro = macro.groupby("Ticker").apply(lambda g: g.iloc[:-max_lookahead] if len(g)>max_lookahead else g.iloc[0:0]).reset_index(drop=True)
    return macro

def build_feature_sets():
    features_short = ["Return_1d","Return_3d","Return_5d","MA_1","MA_3","MA_5",
                      "Vol_1d","Vol_3d","Vol_5d","Oil_Return_1d","GDP_Growth","Inflation_Change"]
    features_medium = features_short + ["MA_21","Vol_21d"]
    features_long = features_medium + ["MA_252","Vol_252d","Trend_90d"]
    return {"1d": features_short, "1m": features_medium, "1y": features_long, "5y": features_long}

# ---------------- MODEL TRAINING ----------------
def train_model(df, feature_cols, target_col):
    y = df[target_col].values
    X = df[feature_cols]
    counts = np.bincount(y)
    if len(counts) == 1:
        neg = counts[0]
        pos = 0
    else:
        neg, pos = counts[0], counts[1]
    scale_pos_weight = float(neg)/pos if pos>0 else 1.0
    model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          objective="binary:logistic", use_label_encoder=False,
                          scale_pos_weight=scale_pos_weight)
    model.fit(X, y)
    return model

# ---------------- ADAPTIVE COMBINED SIGNAL ----------------
def adaptive_combined_signal(df):
    # invert negative-horizon signals if necessary (works on a group / local df)
    for h in HORIZONS.keys():
        if f"Strategy_Return_{h}" not in df.columns:
            # compute using existing Pred_{h} if present
            if f"Pred_{h}" in df.columns:
                df[f"Strategy_Return_{h}"] = df["Close"].pct_change().fillna(0) * df[f"Pred_{h}"].map({1:1,0:0}).shift(1).fillna(0)
            else:
                df[f"Strategy_Return_{h}"] = 0.0
        cum_return = (1 + df[f"Strategy_Return_{h}"]).cumprod().iloc[-1]
        if cum_return < 0:
            df[f"Adj_{h}"] = df[f"Pred_{h}"].map({1:0,0:1}) if f"Pred_{h}" in df.columns else 0
        else:
            df[f"Adj_{h}"] = df[f"Pred_{h}"] if f"Pred_{h}" in df.columns else 0

    # weighted score
    scores = np.zeros(len(df))
    for h in HORIZONS.keys():
        scores += df[f"Adj_{h}"].fillna(0).astype(float) * HORIZON_WEIGHTS[h]

    # optimize thresholds (grid search) — coarse grid to limit runtime
    best_return = -np.inf
    best_thresholds = (0.0, sum(HORIZON_WEIGHTS.values()))
    total_w = sum(HORIZON_WEIGHTS.values())
    grid = np.linspace(0, total_w, num=11)  # coarse 11 steps
    for t1 in grid:
        for t2 in grid[grid >= t1]:
            temp_action = np.where(scores >= t2, "BUY", np.where(scores >= t1, "HOLD", "SELL"))
            temp_pos = np.array([{"SELL":0.0,"HOLD":0.5,"BUY":1.0}[a] for a in temp_action])
            temp_ret = df["Close"].pct_change().fillna(0) * temp_pos
            cum_ret = np.prod(1 + temp_ret) - 1
            if cum_ret > best_return:
                best_return = cum_ret
                best_thresholds = (float(t1), float(t2))

    t1, t2 = best_thresholds
    final_action = np.where(scores >= t2, "BUY", np.where(scores >= t1, "HOLD", "SELL"))
    df["Combined_Action"] = final_action
    return df, best_thresholds, best_return

# ---------------- PIPELINE ----------------
def run_pipeline():
    macro = load_and_merge()
    macro = make_features(macro)
    macro = create_targets(macro)
    feature_map = build_feature_sets()
    models = {}

    latest = macro.groupby("Ticker").tail(1).copy().reset_index(drop=True)

    # --- Train models per horizon (single pass) ---
    for h in HORIZONS.keys():
        print(f"\n--- Training {h} ---")
        feature_cols = feature_map[h]
        target_col = f"Signal_{h}"
        df_train = macro.dropna(subset=feature_cols + [target_col]).copy()
        if len(df_train) < MIN_ROWS_PER_TICKER:
            print(f"Not enough data for {h}, skipping")
            continue

        # Train
        model = train_model(df_train, feature_cols, target_col)

        # Save to memory dictionary
        models[h] = (model, feature_cols)

        # Save individual model file (for API)
        joblib.dump({"model": model, "features": feature_cols}, f"models/model_{h}.pkl")
        print(f"[Saved] models/model_{h}.pkl")

        # Predict ON FULL macro where features exist (mask)
        mask_macro = macro[feature_cols].notna().all(axis=1)
        macro.loc[~mask_macro, f"Pred_{h}"] = 0  # default 0 where features missing
        if mask_macro.any():
            macro.loc[mask_macro, f"Pred_{h}"] = model.predict(macro.loc[mask_macro, feature_cols])

        # Map actions on macro
        macro[f"Action_{h}"] = macro[f"Pred_{h}"].map({1:"BUY", 0:"SELL"})

        # Predict for latest safely (may have NaNs)
        mask_latest = latest[feature_cols].notna().all(axis=1)
        latest.loc[~mask_latest, f"Pred_{h}"] = 0
        if mask_latest.any():
            latest.loc[mask_latest, f"Pred_{h}"] = model.predict(latest.loc[mask_latest, feature_cols])
        latest[f"Action_{h}"] = latest[f"Pred_{h}"].map({1:"BUY", 0:"SELL"})

    # Save master model dict for convenience
    joblib.dump(models, "models/all_models.pkl")
    print("[Saved] models/all_models.pkl")

    # --- Per-ticker backtest with horizon breakdown ---
    horizon_results = []
    for ticker, group in macro.groupby("Ticker"):
        if len(group) < 30:
            continue
        group = group.sort_values("Date").reset_index(drop=True)
        ticker_result = {"Ticker": ticker, "Num_Days": len(group)}
        for h in HORIZONS.keys():
            # strategy returns using shifted prediction (trade on next day)
            if f"Pred_{h}" not in group.columns:
                group[f"Pred_{h}"] = 0
            group[f"Strategy_Return_{h}"] = group["Close"].pct_change().fillna(0) * group[f"Pred_{h}"].map({1:1,0:0}).shift(1).fillna(0)
            cum_ret = (1 + group[f"Strategy_Return_{h}"]).cumprod().iloc[-1] - 1
            num_trades = group[f"Pred_{h}"].diff().abs().sum() / 2
            avg_ret = group[f"Strategy_Return_{h}"].sum() / max(num_trades,1)
            # invert if negative
            if cum_ret < 0:
                group[f"Adj_{h}"] = group[f"Pred_{h}"].map({1:0,0:1})
                cum_ret = (1 + group["Close"].pct_change().fillna(0) * group[f"Adj_{h}"].shift(1).fillna(0)).cumprod().iloc[-1] - 1
                avg_ret = (group["Close"].pct_change().fillna(0) * group[f"Adj_{h}"].shift(1).fillna(0)).sum() / max(num_trades,1)
            else:
                group[f"Adj_{h}"] = group[f"Pred_{h}"]
            ticker_result[f"Return_{h}"] = cum_ret
            ticker_result[f"Avg_Return_{h}"] = avg_ret
            ticker_result[f"Num_Trades_{h}"] = num_trades

        # Adaptive combined signal for this group (will compute thresholds on group)
        group, thresholds, combined_ret = adaptive_combined_signal(group)
        ticker_result["Combined_Return"] = combined_ret
        ticker_result["Num_BUYs"] = (group["Combined_Action"]=="BUY").sum()
        ticker_result["Num_HOLDs"] = (group["Combined_Action"]=="HOLD").sum()
        ticker_result["Num_SELLs"] = (group["Combined_Action"]=="SELL").sum()
        horizon_results.append(ticker_result)

    horizon_df = pd.DataFrame(horizon_results)

    # Overall summary
    overall_summary = {}
    for h in HORIZONS.keys():
        overall_summary[f"Return_{h}"] = horizon_df[f"Return_{h}"].mean()
        overall_summary[f"Avg_Return_{h}"] = horizon_df[f"Avg_Return_{h}"].mean()
        overall_summary[f"Num_Trades_{h}"] = horizon_df[f"Num_Trades_{h}"].mean()
    overall_summary["Combined_Return"] = horizon_df["Combined_Return"].mean()

    # Latest signals (apply adaptive combined on the tiny 'latest' frame)
    latest, _, _ = adaptive_combined_signal(latest)
    out_cols = ["Ticker","Date"] + [f"Action_{h}" for h in HORIZONS.keys()] + ["Combined_Action"]
    latest[out_cols].rename(columns={"Date":"AsOfDate"}).to_json(
        "results.json",
        orient="records",
        indent=2,
        date_format="iso"
    )
    print("\nSaved results to results.json")

    return models, latest, macro, horizon_df, overall_summary

# ---------------- RUN ----------------
if __name__=="__main__":
    models, latest, macro, horizon_df, overall_summary = run_pipeline()
    print("\n=== Overall Backtest Summary ===")
    print(pd.Series(overall_summary))
    print("\n=== Per-Ticker Backtest Summary (head) ===")
    print(horizon_df.head())
