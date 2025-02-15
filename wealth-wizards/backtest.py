import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import adfuller


# Mean Reversion Strategy
def mrs_pnl(lookback, std_dev, df):
    if "prices" not in df.columns:
        print("ERROR: 'prices' column is missing in DataFrame")
        print("Available columns:", df.columns)
        raise ValueError("DataFrame missing 'prices' column")

    df['moving_average'] = df['prices'].rolling(lookback).mean()
    df['moving_std_dev'] = df["prices"].rolling(lookback).std()
    df['upper_band'] = df['moving_average'] + std_dev * df['moving_std_dev']
    df['lower_band'] = df['moving_average'] - std_dev * df['moving_std_dev']
    df['long_entry'] = df["prices"] < df['lower_band']
    df['long_exit'] = df["prices"] >= df['moving_average']
    df['short_entry'] = df["prices"] > df['upper_band']
    df['short_exit'] = df["prices"] <= df['moving_average']
    df['positions_long'] = np.nan
    df.loc[df.long_entry, 'positions_long'] = 1
    df.loc[df.long_exit, 'positions_long'] = 0
    df['positions_short'] = np.nan
    df.loc[df.short_entry, 'positions_short'] = -1
    df.loc[df.short_exit, 'positions_short'] = 0
    df.fillna(method='ffill', inplace=True)
    df['positions'] = df['positions_long'] + df['positions_short']
    df['prices_difference'] = df["prices"] - df["prices"].shift(1)
    df['daily_returns'] = df['prices_difference'] / df["prices"].shift(1)
    df['pnl'] = df['positions'].shift(1) * df['daily_returns']
    df['cumpnl'] = df['pnl'].cumsum()
    return df


# Function to run backtest
def run_backtest(stock_symbol, lookback, std_dev):
    try:
        # Download stock data
        df = yf.download(stock_symbol, period="1y", interval="1d")

        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        df.rename(columns={'Close': 'prices'}, inplace=True)

        # Reset multi-index if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(0)  # Drop first level (Price)
            df = df.rename_axis(None, axis=1)  # Remove index name

        # Print actual column names for debugging
        print("\n==== DEBUG: Downloaded Data After Reset ====")
        print("Columns:", df.columns.tolist())
        print("First Few Rows:\n", df.head())

        if df.empty:
            return None, "No data found for stock."

        # Use 'Close' if available, otherwise 'Adj Close'
        close_col = "Close" if "Close" in df.columns else "Adj Close" if "Adj Close" in df.columns else None
        if close_col:
            df = df.rename(columns={close_col: 'prices'})
        else:
            return None, f"Stock data is missing price information. Columns found: {df.columns.tolist()}"

        # Run the strategy
        df = mrs_pnl(lookback=lookback, std_dev=std_dev, df=df)

        # Save the plot
        plt.figure(figsize=(10, 5))
        df.cumpnl.plot(title=f"Backtest for {stock_symbol}")
        plt.xlabel("Date")
        plt.ylabel("Cumulative PnL")
        image_path = f"static/{stock_symbol}_backtest.png"
        plt.savefig(image_path)
        plt.close()

        return image_path, None

    except Exception as e:
        return None, str(e)
