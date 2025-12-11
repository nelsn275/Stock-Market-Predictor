import requests
import yfinance as yf
import pandas as pd
import time

# -----------------------------
# FETCH S&P 500 TICKERS
# -----------------------------
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
html = requests.get(url, headers=headers).text
sp500 = pd.read_html(html)[0]
tickers = sp500['Symbol'].tolist()
print(f"Found {len(tickers)} tickers.")

# -----------------------------
# DOWNLOAD STOCK DATA
# -----------------------------
all_data = []

for i, symbol in enumerate(tickers, 1):
    try:
        df = yf.download(symbol, start="2016-01-01", end="2025-10-01", progress=False)
        if df.empty:
            print(f"{i}/{len(tickers)}: {symbol} has no data, skipping.")
            continue
        
        # Flatten MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Add ticker column
        df['Ticker'] = symbol
        df.reset_index(inplace=True)
        
        all_data.append(df)
        print(f"{i}/{len(tickers)}: {symbol} downloaded.")

        # Optional: sleep to avoid hitting API limits
        time.sleep(0.1)

    except Exception as e:
        print(f"{i}/{len(tickers)}: Failed download for {symbol}: {e}")

# -----------------------------
# CONCATENATE AND SAVE
# -----------------------------
if all_data:
    stocks = pd.concat(all_data, ignore_index=True)
    stocks.to_csv("sp500_5yr_yf.csv", index=False)
    print("All data saved to sp500_5yr_yf.csv")
else:
    print("No data downloaded.")
