import pandas as pd

oil = pd.read_csv("data/OIL.csv")
inflation = pd.read_csv("data/INFLATION.csv")
gdp = pd.read_csv("data/GDP.csv")

print("OIL columns:", oil.columns.tolist())
print("INFLATION columns:", inflation.columns.tolist())
print("GDP columns:", gdp.columns.tolist())
