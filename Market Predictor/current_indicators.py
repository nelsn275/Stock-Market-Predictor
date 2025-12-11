import requests
import pandas as pd

# =======================
# CONFIGURATION
# =======================
FRED_API_KEY = "cfd0fcba89040180b67e70fc1f026139"



# =======================
# GET CURRENT OIL PRICE (WTI)
# =======================
def get_oil_price():
    """
    Fetch the latest WTI Crude Oil price (daily) from FRED.
    """
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": "DCOILWTICO",
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "sort_order": "desc",  # latest first
        "limit": 2             # last 2 observations in case latest is NaN
    }
    response = requests.get(url, params=params).json()

    try:
        observations = [obs for obs in response["observations"] if obs["value"] != "."]

        if not observations:
            return {"error": "No valid oil price data"}

        latest = float(observations[0]["value"])
        return {"date": observations[0]["date"], "price": latest}
    except Exception as e:
        return {"error": str(e), "response": response}



# =======================
# GET CURRENT INFLATION (US CPI)
# =======================

def get_inflation():
    """
    Fetch the latest US CPI (Consumer Price Index) data from FRED and compute
    the inflation rate (month-over-month % change).
    """
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": "CPIAUCSL",   # US Consumer Price Index for All Urban Consumers
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "sort_order": "desc",      # latest first
        "limit": 2                  # get last two months
    }
    response = requests.get(url, params=params).json()

    try:
        observations = response["observations"]
        if len(observations) < 2:
            return {"error": "Not enough data to calculate inflation"}

        latest = float(observations[0]["value"])
        previous = float(observations[1]["value"])
        inflation_rate = (latest - previous) / previous * 100

        return {
            "date": observations[0]["date"],
            "cpi": latest,
            "inflation_rate_pct": round(inflation_rate, 2)
        }
    except Exception as e:
        return {"error": str(e), "response": response}


# =======================
# GET CURRENT GDP (World Bank)
# =======================
def get_gdp():
    # Indicator: NY.GDP.MKTP.CD (GDP in USD current prices)
    url = "https://api.worldbank.org/v2/country/us/indicator/NY.GDP.MKTP.CD?format=json"
    response = requests.get(url).json()

    try:
        latest = response[1][0]   # most recent entry
        return {
            "year": latest["date"],
            "gdp": latest["value"]
        }
    except:
        return {"error": response}


# =======================
# MAIN
# =======================
if __name__ == "__main__":
    print("\n=== OIL PRICE ===")
    print(get_oil_price())

    print("\n=== INFLATION ===")
    print(get_inflation())

    print("\n=== GDP (World Bank) ===")
    print(get_gdp())