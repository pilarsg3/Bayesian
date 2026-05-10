import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# tickers = ['GC=F', 'BZ=F'] # Gold and Brent Crude
# data = yf.download(tickers, start='2025-01-01', end='2026-01-01')
# data.to_excel("gold_and_brent_crude_data.xlsx")
# data_brent = yf.download('BZ=F', start='2025-01-01', end='2026-01-01')
# data_brent.to_excel("brent_crude_data.xlsx")

# print(data_brent)



tickers = ['BZ=F', 'GC=F', 'DX-Y.NYB','^GSPC'] # Brent crude, Gold, USD, S&P500

tickers = {'brent': 'BZ=F', 'gold': 'GC=F', 'usd': 'DX-Y.NYB', 'sp500': '^GSPC'}


raw = {}
for name, ticker in tickers.items():
    data = yf.download(ticker, start='2025-01-01', end='2025-12-31', auto_adjust=True)
    print(data.columns)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    raw[name]  = data['Close'].rename(name)
    print(raw[name])
    raw[name].to_excel(f"01_raw_{name}.xlsx")


prices = pd.concat(raw.values(), axis=1).dropna()   # Concatenate the dataframes vertically (so column next to each other) in raw and drop the rows where at least one asset has a missing value
print(f"Downloaded {len(prices)} trading days from {prices.index[0].date()} to {prices.index[-1].date()}")  # .date() strips the time part from a datetime object
print(prices.tail())


prices.to_excel('01_raw_close_prices_all.xlsx')


# Now we extract some features from the data

# Compute daily returns in % change
returns = prices.pct_change().add_suffix("_ret") # creates a new table where we compute the percentage change from the prices of the raw table and change the name of each column by adding "_ret"

# Lagged brent returns
for lag in [1, 5, 10]:
    returns[f"brent_ret_lag{lag}"] = returns["brent_ret"].shift(lag) # creates 3 new columns where the value that day is the value of the asset {lag} days ago


# Rolling volatility (10-day and 20-day)
returns["brent_vol_10"] = returns["brent_ret"].rolling(10).std() # creates a new column that contains the standard deviation (= volatility) of the returns over the last 10 days
returns["brent_vol_20"] = returns["brent_ret"].rolling(20).std() 


# Moving average ratios
returns["brent_ma5_ratio"] = prices["brent"] / prices["brent"].rolling(5).mean()
returns["brent_ma20_ratio"] = prices["brent"] / prices["brent"].rolling(20).mean()

# Day of week (Monday=0 ... Friday=4)
returns["weekday"] = prices.index.dayofweek



# ── Target: next-day brent return ────────────────────────────────
returns["target"] = returns["brent_ret"].shift(-1)       # target on day t contains the return of day t+1, i.e. what we try to predict


# Drop rows with NaN (from lags / rolling windows)
data = returns.dropna().copy()
# .dropna() removes rows with any NaN — these appear at the start (rolling windows and lags need warmup rows) and at the end (the last row has no tomorrow, so target is NaN)
# .copy() makes a fresh independent copy so modifying data doesn't affect returns

print(f"\nFinal dataset: {len(data)} samples, {data.shape[1]} columns") # data.shape[1] returns the number of columns
print(data.describe().round(4))     #.describe() gives summary statistics (mean, std, min, max, percentiles) for each column, rounded to 4 decimal places




# ── 4. Quick exploratory plots ───────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Price history
axes[0, 0].plot(prices["brent"])
axes[0, 0].set_title("Brent Crude Price")
axes[0, 0].set_ylabel("USD")


# Return distribution
axes[0, 1].hist(data["brent_ret"], bins=50, edgecolor="black", alpha=0.7)
axes[0, 1].set_title("Daily brent Return Distribution")





# Correlation of features with target
feature_cols = [c for c in data.columns if c != "target"]           # builds a list of all column names except "target". The [c for c in ... if ...] is a list comprehension — it loops through all columns and keeps only those that aren't "target"
corrs = data[feature_cols].corrwith(data["target"]).sort_values()   # takes the DataFrame with only those feature columns, computes the correlation of each one with data["target"], then sorts the results from most negative to most positive
axes[1, 0].barh(corrs.index, corrs.values)                          # plots a horizontal bar chart where the y-axis is the feature names (corrs.index) and the x-axis is their correlation values (corrs.values)
axes[1, 0].set_title("Feature Correlation with Next-Day Return")
axes[1, 0].tick_params(axis="y", labelsize=7)




# Rolling volatility over time
axes[1, 1].plot(data.index, data["brent_vol_20"], label="20-day vol")
axes[1, 1].set_title("Rolling Volatility")
axes[1, 1].legend()


for ax in axes.flat:                        # iterates over all 4 subplots as a flat list
    ax.tick_params(axis="x", rotation=45)   # rotates the x-axis date labels 45 degrees on every subplot so they don't overlap
plt.tight_layout()                          # automatically adjusts spacing between subplots so nothing overlaps
plt.savefig("01_exploration.png", dpi=150)  # saves the figure as a PNG at 150 DPI
plt.show()
print("\nSaved plot → 01_exploration.png")




# ── 5. Save clean data for modelling ──────────────────────────────
data.to_csv("clean_data.csv")
print("Saved data → clean_data.csv")        # it contains all the engineered features (returns, lags, volatility, moving average ratios, weekday, target), with NaN rows removed
