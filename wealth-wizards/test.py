import yfinance as yf

ticker = "ZOMATO.NS"
df = yf.download(ticker, period="1y", interval="1d")

print("\n==== DEBUG: Downloaded Data ====")
print("Columns:", df.columns.tolist())  # Print column names
print("First Few Rows:\n", df.head())   # Print first rows