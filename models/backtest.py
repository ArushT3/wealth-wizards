import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Mean reversion strategy
def mrs_pnl(lookback,std_dev,df):
    df['moving_average'] = df.prices.rolling(lookback).mean()
    df['moving_std_dev'] = df.prices.rolling(lookback).std()
    df['upper_band'] = df.moving_average + std_dev*df.moving_std_dev
    df['lower_band'] = df.moving_average - std_dev*df.moving_std_dev
    df['long_entry'] = df.prices < df.lower_band
    df['long_exit'] = df.prices >= df.moving_average
    df['short_entry'] = df.prices > df.upper_band
    df['short_exit'] = df.prices <= df.moving_average
    df['positions_long'] = np.nan
    df.loc[df.long_entry,'positions_long']= 1
    df.loc[df.long_exit,'positions_long']= 0
    df['positions_short'] = np.nan
    df.loc[df.short_entry,'positions_short']= -1
    df.loc[df.short_exit,'positions_short']= 0
    df = df.fillna(method='ffill')
    df['positions'] = df.positions_long + df.positions_short
    df['prices_difference']= df.prices - df.prices.shift(1)
    df['daily_returns'] = df.prices_difference /df.prices.shift(1)
    df['pnl'] = df.positions.shift(1) * df.daily_returns
    df['cumpnl'] = df.pnl.cumsum()
    return df

with open("./ideal.txt", "r") as file:
    # Loop through each line in the file
    for line in file:
        lin = line.strip().split()
        print(lin)
        dff = pd.read_csv("data/" + lin[0] + ".csv")
        dfff = dff.rename(columns={'Close': 'prices'})
        dffff = mrs_pnl(lookback=int(lin[1]), std_dev=float(lin[2]), df=dfff)
        dffff.cumpnl.plot()
        plt.show()