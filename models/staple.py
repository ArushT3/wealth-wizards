import yfinance as yf

# Path to your file containing stock symbols
file_path = "filtered_lines.txt"

# Read stock tickers from file
def read_tickers_from_file(file_path):
    with open(file_path, 'r') as file:
        tickers = [line.strip().split()[0] for line in file if line.strip()]
    return tickers

# Fetch stock prices
def get_stock_prices(tickers):
    prices = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker + ".NS")
            prices[ticker] = stock.history(period="1d")['Close'].iloc[-1]
        except Exception as e:
            prices[ticker] = f"Error: {e}"
    return prices

# Main workflow
tickers = read_tickers_from_file(file_path)
stock_prices = get_stock_prices(tickers)

sum = 0

# Print stock prices
for ticker, price in stock_prices.items():
    print(ticker + " " + str(price))
    sum += price

print(sum)
