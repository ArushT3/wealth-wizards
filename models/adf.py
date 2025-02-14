import yfinance as yf
from statsmodels.tsa.stattools import adfuller
import pandas as pd

def load_time_series(file_path, column_name):
    try:
        data = pd.read_csv(file_path)
        time_series = data[column_name].dropna()  # Ensure no NaN values
        return time_series
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except KeyError:
        print(f"Error: The column {column_name} does not exist in the file.")
        return None

with open("./symbols.txt", "r") as file:
    for line in file:
        print(line.strip() + ".NS")
        file_path = "data/" + line.strip() + ".csv"
        column_name = "Close"
        time_series = load_time_series(file_path, column_name)

        if time_series is not None:

            adf_result = adfuller(time_series)

            if adf_result[1] <= 0.05:
                print("The time series is stationary (Reject H0).")
                with open("./stocks.txt", "a") as file:
                    file.write(line.strip() + "\n")
            else:
                pass