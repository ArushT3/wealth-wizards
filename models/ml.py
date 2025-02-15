# Import libraries
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def run_ml_model(file_path):
    # Load stock data
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)

    # Create features (Moving Average & Standard Deviation)
    df["std_10"] = df["Close"].rolling(window=10).std()
    df["ma_20"] = df["Close"].rolling(window=20).mean()

    # Drop NaN values
    df.dropna(inplace=True)

    # Convert to JSON for Flask response
    return df[["Close", "std_10", "ma_20"]].tail(10).to_json()
# Apply the default seaborn theme, scaling, and color palette
sns.set()
# One can use different colour palettes
# palettes = ["deep", "muted", "pastel", "bright", "dark", "colorblind"]
# sns.set(palette="deep")

# import warnings
# warnings.filterwarnings('ignore')

# Loading the data from the local file
df = pd.read_csv('nifty_data.csv', index_col=0, parse_dates=True)
# Copying the original dataframe. Will work on the new dataframe.
data = df.copy()
# Checking the shape
print('Number of observations:', data.shape[0])
print('Number of variables:', data.shape[1])
data.head()
# Creating features
features_list = []

# SD based features
for i in range(5, 20, 5):
    col_name = 'std_' + str(i)
    data[col_name] = data['Close'].rolling(window=i).std()
    features_list.append(col_name)

# MA based features
for i in range(10, 30, 5):
    col_name = 'ma_' + str(i)
    data[col_name] = data['Close'].rolling(window=i).mean()
    features_list.append(col_name)

# Daily pct change based features
for i in range(3, 12, 3):
    col_name = 'pct_' + str(i)
    data[col_name] = data['Close'].pct_change().rolling(i).sum()
    features_list.append(col_name)

# Intraday movement
col_name = 'co'
data[col_name] = data['Close'] - data['Open']
features_list.append(col_name)
features_list
# Use the following command on the terminal window on Anaconda to install ta-lib if it is not installed
# conda install -c conda-forge ta-lib
import talib as ta
data['upper_band'], data['middle_band'], data['lower_band'] = ta.BBANDS(data['Close'].values)
data['macd'], data['macdsignal'], data['macdhist'] = ta.MACD(data['Close'].values)
data['sar'] = ta.SAR(data['High'].values, data['Low'].values)
features_list +=['upper_band','middle_band','lower_band','macd','sar']
features_list
data[features_list].head()
data.dropna(inplace=True)
data[features_list].head()
import numpy as np
X = data[features_list]
data['target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, -1)
y = data['target']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
#sns.pairplot(X_train_scaled_df[features_list]);
X_train_scaled_df.describe().round(2)
from sklearn.neural_network import MLPClassifier
# Define model
# model = MLPClassifier(hidden_layer_sizes=(5), verbose=True, random_state=10)
model = MLPClassifier(hidden_layer_sizes=(5), max_iter=300, activation = 'tanh', solver='adam', random_state=1, shuffle=False)

# Train model
model.fit(X_train_scaled, y_train)
# Check number of layers in the model
model.n_layers_
model.get_params()
# Check weights
print('Weights between input layer and the hidden layer:')
print(model.coefs_[0])
print('Biases between input layer and the hidden layer:')
print(model.intercepts_[0])
print('Weights between hidden layer and the output layer:')
print(model.coefs_[1])
print('Biases between hidden layer and the output layer:')
print(model.intercepts_[1])
# Check model accuracy on training data
print('Model accuracy on training data:', model.score(X_train_scaled, y_train))
# Check model accuracy on testing data
print('Model accuracy on testing data:', model.score(X_test_scaled, y_test))
# Predict data
y_pred = model.predict(X_test_scaled)
y_pred
# Calculate Precision and Recall
from sklearn.metrics import precision_score, recall_score

precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
print("Precision:", precision)
print("Recall:", recall)


def backtest(df, model):
    # Copy data
    data = df.copy()

    # Create returns
    data['returns'] = np.log(data['Close'] / data['Close'].shift(1))
    # Creating features
    features_list = []

    # SD based features
    for i in range(5, 20, 5):
        col_name = 'std_' + str(i)
        data[col_name] = data['Close'].rolling(window=i).std()
        features_list.append(col_name)

    # MA based features
    for i in range(10, 30, 5):
        col_name = 'ma_' + str(i)
        data[col_name] = data['Close'].rolling(window=i).mean()
        features_list.append(col_name)

    # Daily pct change based features
    for i in range(3, 12, 3):
        col_name = 'pct_' + str(i)
        data[col_name] = data['Close'].pct_change().rolling(i).sum()
        features_list.append(col_name)

    # Intraday movement
    col_name = 'co'
    data[col_name] = data['Close'] - data['Open']
    features_list.append(col_name)
    # Create features
    data['upper_band'], data['middle_band'], data['lower_band'] = ta.BBANDS(data['Close'].values)
    data['macd'], data['macdsignal'], data['macdhist'] = ta.MACD(data['Close'].values)
    data['sar'] = ta.SAR(data['High'].values, data['Low'].values)
    features_list += ['upper_band', 'middle_band', 'lower_band', 'macd', 'sar']
    # Create target
    data['target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, -1)

    # Drop null values
    data.dropna(inplace=True)

    # Create feature matrix and target vector
    X = data[features_list]
    y = data['target']

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Predict
    y_pred = model.predict(X_scaled)

    data['predicted'] = y_pred

    # Create strategy returns
    data['strategy_returns'] = data['returns'].shift(-1) * data['predicted']

    # Return the last cumulative return
    bnh_returns = data['returns'].cumsum()[-1]

    # Return the last cumulative strategy return
    # we need to drop the last nan value
    data.dropna(inplace=True)
    strategy_returns = data['strategy_returns'].cumsum()[-1]

    plt.figure(figsize=(10, 6))
    plt.plot(data['returns'].cumsum())
    plt.plot(data['strategy_returns'].cumsum())
    plt.xlabel('Time')
    plt.ylabel('Cumulative Returns')
    plt.title('Returns Comparison')
    plt.legend(["Buy and Hold Returns", "Strategy Returns"])
    plt.show()

    return bnh_returns, strategy_returns, data
# Read backtest data
backtest_data = pd.read_csv('nifty_data2.csv', index_col=0, parse_dates=True)

# Backtest the strategy
bnh_returns, s_returns, data = backtest(backtest_data, model)

data
print('Buy and Hold Returns:', bnh_returns)
print('Strategy Returns:', s_returns)