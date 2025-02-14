import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

with open("./ideal.txt", "w") as file:
    pass  # This will effectively empty the file

# Mean reversion strategy
def mrs_pnl(lookback, std_dev, df):
    # Compute Bollinger Bands
    df['moving_average'] = df.Close.rolling(lookback).mean()
    df['moving_std_dev'] = df.Close.rolling(lookback).std()
    df['upper_band'] = df.moving_average + std_dev * df.moving_std_dev
    df['lower_band'] = df.moving_average - std_dev * df.moving_std_dev

    # Check for long and short positions
    df['long_entry'] = df.Close < df.lower_band
    df['long_exit'] = df.Close >= df.moving_average
    df['short_entry'] = df.Close > df.upper_band
    df['short_exit'] = df.Close <= df.moving_average
    df['positions_long'] = np.nan
    df.loc[df.long_entry, 'positions_long'] = 1
    df.loc[df.long_exit, 'positions_long'] = 0
    df['positions_short'] = np.nan
    df.loc[df.short_entry, 'positions_short'] = -1
    df.loc[df.short_exit, 'positions_short'] = 0
    df = df.fillna(method='ffill')
    df['positions'] = df.positions_long + df.positions_short

    # Calculate the PnL
    df['Close_difference'] = df.Close - df.Close.shift(1)
    df['pnl'] = df.positions.shift(1) * df.Close_difference
    df['cumpnl'] = df.pnl.cumsum()
    return df.cumpnl.iloc[-1]

with open("./stocks.txt", "r") as file:
    # Loop through each line in the file
    for line in file:
        # Possible values of lookback period to try out
        print(line)
        lookback = [int(x) for x in np.linspace(start=2, stop=15, num=5)]

        # Possible values of standard deviation period to try out
        std_dev = [round(x, 2) for x in np.linspace(start=0.5, stop=2.5, num=5)]

        path = "temp/" + line.strip() + ".csv"
        # Read data
        df = pd.read_csv(path, index_col=0, header=0)

        # Split the data to optimize and validate the parameter
        train_length = int(len(df) * 0.7)
        train_set = pd.DataFrame(data=df[:train_length])
        test_set = pd.DataFrame(data=df[train_length:])

        # Analyze the performance with different parameter setting on the train dataset
        matrix = np.zeros([len(lookback), len(std_dev)])
        for i in range(len(lookback)):
            for j in range(len(std_dev)):
                matrix[i][j] = mrs_pnl(lookback[i], std_dev[j], train_set) * 100

        # print(matrix)
        # import seaborn
        #
        # seaborn.heatmap(matrix, cmap='RdYlGn',
        #                 xticklabels=std_dev, yticklabels=lookback)
        # plt.show()

        opt = np.where(matrix == np.max(matrix))
        opt_lookback = lookback[opt[0][0]]
        opt_std_dev = std_dev[opt[1][0]]
        print('Lookback Optimal', opt_lookback)
        print('Standard Deviation Optimal', opt_std_dev)

        with open("./ideal.txt", "a") as file:
            file.write(line.strip() + " " + str(opt_lookback) + " " + str(opt_std_dev) + "\n")