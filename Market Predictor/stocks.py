#!/usr/bin/env python3
"""
Fetch latest S&P500 price (yfinance preferred, CSV fallback), merge with results.json
and write json/stocks.json.

Usage:
    python stocks.py

Output:
    json/stocks.json

The output structure:
{
  "sp500": {"symbol": "^GSPC", "last_date": "YYYY-MM-DD", "last_close": 1234.56, "source": "yfinance"},
  "predictions": [ ... contents of results.json as loaded (list/array or object) ]
}
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def fetch_sp500_yfinance() -> Dict[str, Any] | None:
    try:
        import yfinance as yf
    except Exception as e:
        print("yfinance not available:", e)
        return None

    try:
        # ^GSPC is the Yahoo symbol for S&P 500
        ticker = yf.Ticker("^GSPC")
        # get last 7 days to be safe around weekends/holidays
        hist = ticker.history(period="7d", interval="1d")
        if hist is None or hist.empty:
            print("yfinance returned no historical data for ^GSPC")
            return None
        # pick the last valid close (dropna)
        hist = hist.dropna(subset=["Close"], how="all")
        if hist.empty:
            return None
        last_row = hist.iloc[-1]
        last_close = float(last_row["Close"]) if "Close" in last_row else None
        last_date = pd.to_datetime(last_row.name).strftime("%Y-%m-%d")
        return {"symbol": "^GSPC", "last_date": last_date, "last_close": last_close, "source": "yfinance"}
    except Exception as e:
        print("Error fetching yfinance data for ^GSPC:", e)
        return None


def fetch_sp500_csv(csv_path: Path) -> Dict[str, Any] | None:
    if not csv_path.exists():
        print(f"CSV fallback not found at {csv_path}")
        return None
    try:
        df = pd.read_csv(csv_path, parse_dates=True, infer_datetime_format=True)
        # heuristics for date and price columns
        date_col = None
        for c in ["Date", "date", "DATE"]:
            if c in df.columns:
                date_col = c
                break
        price_col = None
        for c in ["Adj Close", "Adj_Close", "Close", "close", "CLOSE"]:
            if c in df.columns:
                price_col = c
                break
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col])
            df = df.sort_values(by=date_col)
            last = df.iloc[-1]
            last_date = pd.to_datetime(last[date_col]).strftime("%Y-%m-%d")
            last_close = float(last[price_col]) if price_col and price_col in last else None
            return {"symbol": "^GSPC", "last_date": last_date, "last_close": last_close, "source": "historical_csv"}
        else:
            # fallback: try using index as dates
            if df.index.size:
                last = df.iloc[-1]
                # try common price columns
                if price_col:
                    last_close = float(last[price_col])
                else:
                    # try first numeric column
                    nums = df.select_dtypes(include=["number"]).columns
                    last_close = float(last[nums[0]]) if len(nums) else None
                return {"symbol": "^GSPC", "last_date": None, "last_close": last_close, "source": "historical_csv"}
        return None
    except Exception as e:
        print("Error reading CSV fallback for SP500:", e)
        return None


def load_results_json(results_path: Path) -> Any:
    if not results_path.exists():
        print(f"results.json not found at {results_path}")
        return None
    try:
        with results_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as e:
        print("Error loading results.json:", e)
        return None


def write_output_json(out_path: Path, payload: Dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        tmp.replace(out_path)
        print(f"Wrote {out_path}")
    except Exception as e:
        print(f"Failed to write {out_path}:", e)


def generate_stocks_json(base_dir: Path | None = None) -> Path:
    base_dir = Path(base_dir or Path(__file__).parent)
    results_path = base_dir / "results.json"
    csv_fallback = base_dir / "data" / "SP500.csv"
    out_dir = base_dir / "json"
    out_path = out_dir / "stocks.json"

    results = load_results_json(results_path)

    sp500 = fetch_sp500_yfinance()
    if sp500 is None:
        sp500 = fetch_sp500_csv(csv_fallback)
    if sp500 is None:
        # last-resort: try to read SP500.csv in cwd
        sp500 = {"symbol": "^GSPC", "last_date": None, "last_close": None, "source": "none"}

    payload = {"sp500": sp500, "predictions": results}

    write_output_json(out_path, payload)
    return out_path


if __name__ == "__main__":
    out = generate_stocks_json()
    print("Output:", out)
