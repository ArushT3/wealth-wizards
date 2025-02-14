import yfinance as yf
from statsmodels.tsa.stattools import adfuller
import pandas as pd

# Load the data from a CSV file
def load_time_series(file_path, column_name):
    """Load the time series data from a specified column in a CSV file."""
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


# Open the file in read mode
with open("./symbols.txt", "r") as file:
    # Loop through each line in the file
    for line in file:
        # Process each line (e.g., print it)
        print(line.strip() + ".NS")  # .strip() removes leading/trailing whitespace

        ticker = line.strip() + ".NS"
        data = yf.download(tickers=ticker, interval="1m")
        data.reset_index(inplace=True)
        data.insert(0, "Datetime", data.pop("Datetime"))
        output_file_path = "data/" + line.strip() + ".csv"
        data.to_csv(output_file_path, index=False)

        file_path = "data/" + line.strip() + ".csv"
        with open(file_path, 'r') as file:
            lines = file.readlines()
        lines = lines[:1] + lines[2:]
        modified_file_path = "data/" + line.strip() + ".csv"
        with open(modified_file_path, 'w') as file:
            file.writelines(lines)

        print(f"Modified file saved as: {modified_file_path}")