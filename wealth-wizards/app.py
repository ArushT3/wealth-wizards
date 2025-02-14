
from cs50 import SQL
from flask import Flask, redirect, session
from werkzeug.security import check_password_hash, generate_password_hash
from flask_session import Session
from helpers import apology

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


# Mean reversion strategy
def mrs_pnl_pre(lookback, std_dev, df):
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




#Loads the database and initialises the session
app = Flask(__name__)
db = SQL("sqlite:///wealth_wizards.db")


app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)



#Login Function
@app.route("/login", methods=["GET", "POST"])


def login():
    session.clear()
    if request.method == "GET":
        return render_template("login.html")
    else:
        if not request.form.get("username"):
            return apology("must provide username", "/login", "Go back to login")

        elif not request.form.get("password"):
            return apology("must provide password", "/login", "Go back to login")

        rows = db.execute("SELECT * FROM users WHERE username = ?", request.form.get("username"))

        if len(rows) != 1 or not check_password_hash(rows[0]["hash"], request.form.get("password")):
            return apology("invalid username and/or password", "/login", redirect_text="Go back to login")

        session["user_id"] = rows[0]["id"]

        return redirect("/")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


@app.route("/register", methods=["GET", "POST"])
def register():
    """Register user"""
    session.clear()
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        confirmation = request.form.get("confirmation")

        rows_u = db.execute("SELECT * FROM users WHERE username = ?", username)
        if not username:
            return apology("Blank Username", "/register", "Go back to register")
        elif not password:
            return apology("Blank Password", "/register", "Go back to register")
        elif len(rows_u) != 0:
            return apology("Username Exists", "/register", "Go back to register")
        elif not confirmation:
            return apology("Blank Confirmation", "/register", "Go back to register")
        elif password != confirmation:
            return apology("Password not matching confirmation", "/register", "Go back to register")
        else:
            db.execute("INSERT INTO users (username, hash) VALUES(?, ?)", username,
                       generate_password_hash(password))
        return redirect("/login")
    else:
        return render_template("register.html")


@app.route("/")
def index():
    if not session.get("user_id"):
        return redirect("/login")

    name = db.execute("SELECT username FROM users WHERE id=?", session["user_id"])
    p_list =  []
    return render_template("index.html", name=name[0]["username"], list=p_list)


@app.route("/addstock", methods=["GET", "POST"])
def addstock():
    if request.method == "POST":
        stock = request.form.get("stock")
        type = request.form.get("type")
        if not stock:
            return apology("Please enter Stock", "/addstock", "Go back to Add a Stock")
        if not type:
            return apology("Please enter Type", "/addstock", "Go back to Add a Stock")
        db.execute(
            "INSERT INTO stocks (stock, type, user_id) VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
            stock, type, session["user_id"])
        return redirect("/")
    else:
        return render_template("addstock.html")


@app.route("/stocks", methods=["GET", "POST"])
def stocks():
    return render_template("stocks.html", options=[])


# @app.route("/backtest", methods=["GET", "POST"])
# def backtest():
#     if request.method == "POST":
#         stock = request.form.get("stock")
#         print(stock)
#         type = request.form.get("type")
#         if not stock:
#             return apology("Please enter Stock", "/backtest", "Go back to backtest")
#         if not type:
#             return apology("Please enter Type", "/backtest", "Go back to backtest")
#
#
#         print(stock + ".NS")
#         ticker = stock + ".NS"
#         data = yf.download(tickers=stock+".NS", interval="1m", period="1d")
#         data.head()
#         data = data.iloc[3:].reset_index(drop=True)
#         data.reset_index(inplace=True)
#         print(data)
#         output_file_path = "temp/" + stock + ".csv"
#         data.to_csv(output_file_path, index=False)
#         print(stock + ".NS")
#         file_path = "temp/" + stock + ".csv"
#         column_name = "Close"
#         time_series = load_time_series(file_path, column_name)
#
#         if time_series is None:
#             print("Error: Time series data not loaded. Check file path or column name.")
#         elif isinstance(time_series, pd.Series):
#             time_series = time_series.to_frame().reset_index(drop=True)
#
#         if len(time_series) > 3:
#             time_series = time_series.iloc[3:].reset_index(drop=True)
#             print(time_series)
#             print("Loaded Time Series")
#         else:
#             print("Warning: Not enough data to remove first 3 rows.")
#
#         print(time_series)
#         print("Loaded Time Series")
#         if time_series is not None:
#             adf_result = adfuller(time_series)
#             print(adf_result)
#             if adf_result[0] <= 0.05:
#                 print("The time series is stationary (Reject H0).")
#                 print(stock)
#                 # Possible values of lookback period to try out
#                 lookback = [int(x) for x in np.linspace(start=2, stop=15, num=5)]
#
#                 # Possible values of standard deviation period to try out
#                 std_dev = [round(x, 2) for x in np.linspace(start=0.5, stop=2.5, num=5)]
#
#                 path = "temp/" + stock + ".csv"
#                 # Read data
#                 import pandas as pd
#                 import numpy as np
#
#                 # Load the data, removing NaN index and setting correct header
#                 df = pd.read_csv(path, index_col=0, header=0)
#
#                 # Remove the first row if it's redundant (contains column names as values)
#                 df = df.iloc[1:].reset_index(drop=True)
#
#                 # Convert index to integer
#                 df['index'] = df['index'].astype(float).astype(int)
#
#                 # Set 'index' as DataFrame index
#                 df = df.set_index('index')
#
#                 # Convert all columns to numeric
#                 df = df.apply(pd.to_numeric, errors='coerce')
#
#                 # Display cleaned data
#                 print(df.head())
#
#                 # Split data into training and testing sets
#                 train_length = int(len(df) * 0.7)
#                 train_set = df.iloc[:train_length].copy()
#                 test_set = df.iloc[train_length:].copy()
#
#                 # Ensure no NaN index before proceeding
#                 df = df.dropna().reset_index()
#
#                 # Set the index back again
#                 df = df.set_index('index')
#
#                 # Display cleaned data again
#                 print(df.head())
#
#                 # Define the lookback and std_dev range (assuming they are predefined lists)
#                 lookback = [10, 20, 30]  # Example values
#                 std_dev = [1, 2, 3]  # Example values
#
#                 # Ensure train_set is not empty
#                 if train_set.empty:
#                     raise ValueError("Training set is empty after preprocessing!")
#
#                 # Initialize matrix for performance evaluation
#                 matrix = np.zeros([len(lookback), len(std_dev)])
#
#                 # Loop through parameter combinations and store results
#                 for i in range(len(lookback)):
#                     for j in range(len(std_dev)):
#                         matrix[i][j] = mrs_pnl_pre(lookback[i], std_dev[j], train_set) * 100
#
#                 # Display the resulting performance matrix
#                 print(matrix)
#
#                 # print(matrix)
#                 # import seaborn
#                 #
#                 # seaborn.heatmap(matrix, cmap='RdYlGn',
#                 #                 xticklabels=std_dev, yticklabels=lookback)
#                 # plt.show()
#
#                 opt = np.where(matrix == np.max(matrix))
#                 opt_lookback = lookback[opt[0][0]]
#                 opt_std_dev = std_dev[opt[1][0]]
#                 print('Lookback Optimal', opt_lookback)
#                 print('Standard Deviation Optimal', opt_std_dev)
#                 dff = pd.read_csv("temp/" + stock + ".csv")
#                 dfff = dff.rename(columns={'Close': 'prices'})
#                 print(dfff)
#                 dffff = mrs_pnl_pre(lookback=opt_lookback, std_dev=opt_std_dev, df=dfff)
#                 dffff.cumpnl.plot()
#                 plt.savefig("plot.png", format="png", dpi=300, bbox_inches="tight")
#                 return render_template("results.html", message="Backtest Results for" + stock, backtest_p="./static/logo.png")
#             else:
#                 pass
#     else:
        # return render_template("backtest.html", stocks=["360ONE", "3MINDIA", "ABB", "ACC", "AIAENG", "APLAPOLLO", "AUBANK", "AADHARHFC", "AARTIIND", "AAVAS", "ABBOTINDIA", "ACE", "ADANIENSOL", "ADANIENT", "ADANIGREEN", "ADANIPORTS", "ADANIPOWER", "ATGL", "AWL", "ABCAPITAL", "ABFRL", "ABREL", "ABSLAMC", "AEGISLOG", "AFFLE", "AJANTPHARM", "AKUMS", "APLLTD", "ALKEM", "ALKYLAMINE", "ALOKINDS", "ARE&M", "AMBER", "AMBUJACEM", "ANANDRATHI", "ANANTRAJ", "ANGELONE", "APARINDS", "APOLLOHOSP", "APOLLOTYRE", "APTUS", "ACI", "ASAHIINDIA", "ASHOKLEY", "ASIANPAINT", "ASTERDM", "ASTRAZEN", "ASTRAL", "ATUL", "AUROPHARMA", "AVANTIFEED", "DMART", "AXISBANK", "BASF", "BEML", "BLS", "BSE", "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BAJAJHLDNG", "BALAMINES", "BALKRISIND", "BALRAMCHIN", "BANDHANBNK", "BANKBARODA", "BANKINDIA", "MAHABANK", "BATAINDIA", "BAYERCROP", "BERGEPAINT", "BDL", "BEL", "BHARATFORG", "BHEL", "BPCL", "BHARTIARTL", "BHARTIHEXA", "BIKAJI", "BIOCON", "BIRLACORPN", "BSOFT", "BLUEDART", "BLUESTARCO", "BBTC", "BOSCHLTD", "BRIGADE", "BRITANNIA", "MAPMYINDIA", "CCL", "CESC", "CGPOWER", "CIEINDIA", "CRISIL", "CAMPUS", "CANFINHOME", "CANBK", "CAPLIPOINT", "CGCL", "CARBORUNIV", "CASTROLIND", "CEATLTD", "CELLO", "CENTRALBK", "CDSL", "CENTURYPLY", "CERA", "CHALET", "CHAMBLFERT", "CHEMPLASTS", "CHENNPETRO", "CHOLAHLDNG", "CHOLAFIN", "CIPLA", "CUB", "CLEAN", "COALINDIA", "COCHINSHIP", "COFORGE", "COLPAL", "CAMS", "CONCORDBIO", "CONCOR", "COROMANDEL", "CRAFTSMAN", "CREDITACC", "CROMPTON", "CUMMINSIND", "CYIENT", "DLF", "DOMS", "DABUR", "DALBHARAT", "DATAPATTNS", "DEEPAKFERT", "DEEPAKNTR", "DELHIVERY", "DEVYANI", "DIVISLAB", "DIXON", "LALPATHLAB", "DRREDDY", "EIDPARRY", "EIHOTEL", "EASEMYTRIP", "EICHERMOT", "ELECON", "ELGIEQUIP", "EMAMILTD", "EMCURE", "ENDURANCE", "ENGINERSIN", "EQUITASBNK", "ERIS", "ESCORTS", "EXIDEIND", "NYKAA", "FEDERALBNK", "FACT", "FINEORG", "FINCABLES", "FINPIPE", "FSL", "FIVESTAR", "FORTIS", "GRINFRA", "GAIL", "GVT&D", "GMRAIRPORT", "GRSE", "GICRE", "GILLETTE", "GLAND", "GLAXO", "GLENMARK", "MEDANTA", "GODIGIT", "GPIL", "GODFRYPHLP", "GODREJAGRO", "GODREJCP", "GODREJIND", "GODREJPROP", "GRANULES", "GRAPHITE", "GRASIM", "GESHIP", "GRINDWELL", "GAEL", "FLUOROCHEM", "GUJGASLTD", "GMDCLTD", "GNFC", "GPPL", "GSFC", "GSPL", "HEG", "HBLENGINE", "HCLTECH", "HDFCAMC", "HDFCBANK", "HDFCLIFE", "HFCL", "HAPPSTMNDS", "HAVELLS", "HEROMOTOCO", "HSCL", "HINDALCO", "HAL", "HINDCOPPER", "HINDPETRO", "HINDUNILVR", "HINDZINC", "POWERINDIA", "HOMEFIRST", "HONASA", "HONAUT", "HUDCO", "ICICIBANK", "ICICIGI", "ICICIPRULI", "ISEC", "IDBI", "IDFCFIRSTB", "IFCI", "IIFL", "INOXINDIA", "IRB", "IRCON", "ITC", "ITI", "INDGN", "INDIACEM", "INDIAMART", "INDIANB", "IEX", "INDHOTEL", "IOC", "IOB", "IRCTC", "IRFC", "IREDA", "IGL", "INDUSTOWER", "INDUSINDBK", "NAUKRI", "INFY", "INOXWIND", "INTELLECT", "INDIGO", "IPCALAB", "JBCHEPHARM", "JKCEMENT", "JBMA", "JKLAKSHMI", "JKTYRE", "JMFINANCIL", "JSWENERGY", "JSWINFRA", "JSWSTEEL", "JPPOWER", "J&KBANK", "JINDALSAW", "JSL", "JINDALSTEL", "JIOFIN", "JUBLFOOD", "JUBLINGREA", "JUBLPHARMA", "JWL", "JUSTDIAL", "JYOTHYLAB", "JYOTICNC", "KPRMILL", "KEI", "KNRCON", "KPITTECH", "KSB", "KAJARIACER", "KPIL", "KALYANKJIL", "KANSAINER", "KARURVYSYA", "KAYNES", "KEC", "KFINTECH", "KIRLOSBROS", "KIRLOSENG", "KOTAKBANK", "KIMS", "LTF", "LTTS", "LICHSGFIN", "LTIM", "LT", "LATENTVIEW", "LAURUSLABS", "LEMONTREE", "LICI", "LINDEINDIA", "LLOYDSME", "LUPIN", "MMTC", "MRF", "LODHA", "MGL", "MAHSEAMLES", "M&MFIN", "M&M", "MAHLIFE", "MANAPPURAM", "MRPL", "MANKIND", "MARICO", "MARUTI", "MASTEK", "MFSL", "MAXHEALTH", "MAZDOCK", "METROBRAND", "METROPOLIS", "MINDACORP", "MSUMI", "MOTILALOFS", "MPHASIS", "MCX", "MUTHOOTFIN", "NATCOPHARM", "NBCC", "NCC", "NHPC", "NLCINDIA", "NMDC", "NSLNISP", "NTPC", "NH", "NATIONALUM", "NAVINFLUOR", "NESTLEIND", "NETWEB", "NETWORK18", "NEWGEN", "NAM-INDIA", "NUVAMA", "NUVOCO", "OBEROIRLTY", "ONGC", "OIL", "OLECTRA", "PAYTM", "OFSS", "POLICYBZR", "PCBL", "PIIND", "PNBHOUSING", "PNCINFRA", "PTCIL", "PVRINOX", "PAGEIND", "PATANJALI", "PERSISTENT", "PETRONET", "PFIZER", "PHOENIXLTD", "PIDILITIND", "PEL", "PPLPHARMA", "POLYMED", "POLYCAB", "POONAWALLA", "PFC", "POWERGRID", "PRAJIND", "PRESTIGE", "PGHH", "PNB", "QUESS", "RRKABEL", "RBLBANK", "RECLTD", "RHIM", "RITES", "RADICO", "RVNL", "RAILTEL", "RAINBOW", "RAJESHEXPO", "RKFORGE", "RCF", "RATNAMANI", "RTNINDIA", "RAYMOND", "REDINGTON", "RELIANCE", "ROUTE", "SBFC", "SBICARD", "SBILIFE", "SJVN", "SKFINDIA", "SRF", "SAMMAANCAP", "MOTHERSON", "SANOFI", "SAPPHIRE", "SAREGAMA", "SCHAEFFLER", "SCHNEIDER", "SCI", "SHREECEM", "RENUKA", "SHRIRAMFIN", "SHYAMMETL", "SIEMENS", "SIGNATURE", "SOBHA", "SOLARINDS", "SONACOMS", "SONATSOFTW", "STARHEALTH", "SBIN", "SAIL", "SWSOLAR", "SUMICHEM", "SPARC", "SUNPHARMA", "SUNTV", "SUNDARMFIN", "SUNDRMFAST", "SUPREMEIND", "SUVENPHAR", "SUZLON", "SWANENERGY", "SYNGENE", "SYRMA", "TBOTEK", "TVSMOTOR", "TVSSCS", "TANLA", "TATACHEM", "TATACOMM", "TCS", "TATACONSUM", "TATAELXSI", "TATAINVEST", "TATAMOTORS", "TATAPOWER", "TATASTEEL", "TATATECH", "TTML", "TECHM", "TECHNOE", "TEJASNET", "NIACL", "RAMCOCEM", "THERMAX", "TIMKEN", "TITAGARH", "TITAN", "TORNTPHARM", "TORNTPOWER", "TRENT", "TRIDENT", "TRIVENI", "TRITURBINE", "TIINDIA", "UCOBANK", "UNOMINDA", "UPL", "UTIAMC", "UJJIVANSFB", "ULTRACEMCO", "UNIONBANK", "UBL", "UNITDSPR", "USHAMART", "VGUARD", "VIPIND", "DBREALTY", "VTL", "VARROC", "VBL", "MANYAVAR", "VEDL", "VIJAYA", "VINATIORGA", "IDEA", "VOLTAS", "WELCORP", "WELSPUNLIV", "WESTLIFE", "WHIRLPOOL", "WIPRO", "YESBANK", "ZFCVINDIA", "ZEEL", "ZENSARTECH", "ZOMATO", "ZYDUSLIFE", "ECLERX"], options=["Algorithm"])

import os
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from flask import request, render_template


@app.route("/backtest", methods=["GET", "POST"])
def backtest():
    if request.method == "POST":
        # Get user input
        stock = request.form.get("stock", "").strip()
        backtest_type = request.form.get("type", "").strip()

        # Validate input
        if not stock:
            return apology("Please enter Stock", "/backtest", "Go back to backtest")
        if not backtest_type:
            return apology("Please enter Type", "/backtest", "Go back to backtest")

        ticker = stock + ".NS"
        try:
            # Download data
            data = yf.download(tickers=ticker, interval="1m", period="1d")
            if data.empty:
                return apology("No data available for the stock.", "/backtest", "Go back to backtest")

            # Process data
            data = data.iloc[3:].reset_index(drop=True)
            output_file_path = os.path.join("temp", f"{stock}.csv")
            data.to_csv(output_file_path, index=False)

            # Load time series
            file_path = os.path.join("temp", f"{stock}.csv")
            column_name = "Close"
            time_series = load_time_series(file_path, column_name)

            if time_series is None or time_series.empty:
                return apology("Error: Time series data not loaded.", "/backtest", "Go back to backtest")

            time_series = time_series.to_frame().reset_index(drop=True)

            # Ensure sufficient data for processing
            if len(time_series) > 3:
                time_series = time_series.iloc[3:].reset_index(drop=True)
            else:
                return apology("Not enough data to process.", "/backtest", "Go back to backtest")

            # Perform ADF test
            adf_result = adfuller(time_series["Close"])
            p_value = adf_result[1]
            print(p_value)
            if p_value > 0.05:
                return apology("Time series is not stationary. Backtest cannot proceed.", "/backtest",
                               "Go back to backtest")

            # Hyperparameters
            lookback = [10, 20, 30]
            std_dev = [1, 2, 3]

            # Load and clean CSV data
            df = pd.read_csv(file_path, index_col=0, header=0)
            df = df.iloc[1:].reset_index(drop=True)
            df["index"] = df.index.astype(int)
            df.set_index("index", inplace=True)
            df = df.apply(pd.to_numeric, errors="coerce").dropna()

            # Split into training and testing sets
            train_length = int(len(df) * 0.7)
            train_set = df.iloc[:train_length].copy()

            if train_set.empty:
                return apology("Training set is empty after preprocessing!", "/backtest", "Go back to backtest")

            # Optimization loop
            matrix = np.zeros((len(lookback), len(std_dev)))
            print("Hit 7")
            for i, lb in enumerate(lookback):
                for j, sd in enumerate(std_dev):
                    matrix[i, j] = mrs_pnl_pre(lb, sd, train_set) * 100

            # Find optimal parameters
            opt_idx = np.unravel_index(matrix.argmax(), matrix.shape)
            opt_lookback = lookback[opt_idx[0]]
            opt_std_dev = std_dev[opt_idx[1]]

            # Apply best parameters to backtest
            df = pd.read_csv(file_path)
            df.rename(columns={"Close": "prices"}, inplace=True)

            result = mrs_pnl_pre(lookback=opt_lookback, std_dev=opt_std_dev, df=df)

            # Plot and save result
            plt.figure()
            result.cumpnl.plot()
            plt.savefig("static/plot.png", format="png", dpi=300, bbox_inches="tight")

            return render_template("results.html", message=f"Backtest Results for {stock}",
                                   backtest_p="static/plot.png")

        except Exception as e:
            return apology(f"An error occurred: {str(e)}", "/backtest", "Go back to backtest")

    else:
        return render_template("backtest.html",
                               stocks=["360ONE", "3MINDIA", "ABB", "ACC", "AIAENG", "APLAPOLLO", "AUBANK", "AADHARHFC",
                                       "AARTIIND", "AAVAS", "ABBOTINDIA", "ACE", "ADANIENSOL", "ADANIENT", "ADANIGREEN",
                                       "ADANIPORTS", "ADANIPOWER", "ATGL", "AWL", "ABCAPITAL", "ABFRL", "ABREL",
                                       "ABSLAMC", "AEGISLOG", "AFFLE", "AJANTPHARM", "AKUMS", "APLLTD", "ALKEM",
                                       "ALKYLAMINE", "ALOKINDS", "ARE&M", "AMBER", "AMBUJACEM", "ANANDRATHI",
                                       "ANANTRAJ", "ANGELONE", "APARINDS", "APOLLOHOSP", "APOLLOTYRE", "APTUS", "ACI",
                                       "ASAHIINDIA", "ASHOKLEY", "ASIANPAINT", "ASTERDM", "ASTRAZEN", "ASTRAL", "ATUL",
                                       "AUROPHARMA", "AVANTIFEED", "DMART", "AXISBANK", "BASF", "BEML", "BLS", "BSE",
                                       "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BAJAJHLDNG", "BALAMINES",
                                       "BALKRISIND", "BALRAMCHIN", "BANDHANBNK", "BANKBARODA", "BANKINDIA", "MAHABANK",
                                       "BATAINDIA", "BAYERCROP", "BERGEPAINT", "BDL", "BEL", "BHARATFORG", "BHEL",
                                       "BPCL", "BHARTIARTL", "BHARTIHEXA", "BIKAJI", "BIOCON", "BIRLACORPN", "BSOFT",
                                       "BLUEDART", "BLUESTARCO", "BBTC", "BOSCHLTD", "BRIGADE", "BRITANNIA",
                                       "MAPMYINDIA", "CCL", "CESC", "CGPOWER", "CIEINDIA", "CRISIL", "CAMPUS",
                                       "CANFINHOME", "CANBK", "CAPLIPOINT", "CGCL", "CARBORUNIV", "CASTROLIND",
                                       "CEATLTD", "CELLO", "CENTRALBK", "CDSL", "CENTURYPLY", "CERA", "CHALET",
                                       "CHAMBLFERT", "CHEMPLASTS", "CHENNPETRO", "CHOLAHLDNG", "CHOLAFIN", "CIPLA",
                                       "CUB", "CLEAN", "COALINDIA", "COCHINSHIP", "COFORGE", "COLPAL", "CAMS",
                                       "CONCORDBIO", "CONCOR", "COROMANDEL", "CRAFTSMAN", "CREDITACC", "CROMPTON",
                                       "CUMMINSIND", "CYIENT", "DLF", "DOMS", "DABUR", "DALBHARAT", "DATAPATTNS",
                                       "DEEPAKFERT", "DEEPAKNTR", "DELHIVERY", "DEVYANI", "DIVISLAB", "DIXON",
                                       "LALPATHLAB", "DRREDDY", "EIDPARRY", "EIHOTEL", "EASEMYTRIP", "EICHERMOT",
                                       "ELECON", "ELGIEQUIP", "EMAMILTD", "EMCURE", "ENDURANCE", "ENGINERSIN",
                                       "EQUITASBNK", "ERIS", "ESCORTS", "EXIDEIND", "NYKAA", "FEDERALBNK", "FACT",
                                       "FINEORG", "FINCABLES", "FINPIPE", "FSL", "FIVESTAR", "FORTIS", "GRINFRA",
                                       "GAIL", "GVT&D", "GMRAIRPORT", "GRSE", "GICRE", "GILLETTE", "GLAND", "GLAXO",
                                       "GLENMARK", "MEDANTA", "GODIGIT", "GPIL", "GODFRYPHLP", "GODREJAGRO", "GODREJCP",
                                       "GODREJIND", "GODREJPROP", "GRANULES", "GRAPHITE", "GRASIM", "GESHIP",
                                       "GRINDWELL", "GAEL", "FLUOROCHEM", "GUJGASLTD", "GMDCLTD", "GNFC", "GPPL",
                                       "GSFC", "GSPL", "HEG", "HBLENGINE", "HCLTECH", "HDFCAMC", "HDFCBANK", "HDFCLIFE",
                                       "HFCL", "HAPPSTMNDS", "HAVELLS", "HEROMOTOCO", "HSCL", "HINDALCO", "HAL",
                                       "HINDCOPPER", "HINDPETRO", "HINDUNILVR", "HINDZINC", "POWERINDIA", "HOMEFIRST",
                                       "HONASA", "HONAUT", "HUDCO", "ICICIBANK", "ICICIGI", "ICICIPRULI", "ISEC",
                                       "IDBI", "IDFCFIRSTB", "IFCI", "IIFL", "INOXINDIA", "IRB", "IRCON", "ITC", "ITI",
                                       "INDGN", "INDIACEM", "INDIAMART", "INDIANB", "IEX", "INDHOTEL", "IOC", "IOB",
                                       "IRCTC", "IRFC", "IREDA", "IGL", "INDUSTOWER", "INDUSINDBK", "NAUKRI", "INFY",
                                       "INOXWIND", "INTELLECT", "INDIGO", "IPCALAB", "JBCHEPHARM", "JKCEMENT", "JBMA",
                                       "JKLAKSHMI", "JKTYRE", "JMFINANCIL", "JSWENERGY", "JSWINFRA", "JSWSTEEL",
                                       "JPPOWER", "J&KBANK", "JINDALSAW", "JSL", "JINDALSTEL", "JIOFIN", "JUBLFOOD",
                                       "JUBLINGREA", "JUBLPHARMA", "JWL", "JUSTDIAL", "JYOTHYLAB", "JYOTICNC",
                                       "KPRMILL", "KEI", "KNRCON", "KPITTECH", "KSB", "KAJARIACER", "KPIL",
                                       "KALYANKJIL", "KANSAINER", "KARURVYSYA", "KAYNES", "KEC", "KFINTECH",
                                       "KIRLOSBROS", "KIRLOSENG", "KOTAKBANK", "KIMS", "LTF", "LTTS", "LICHSGFIN",
                                       "LTIM", "LT", "LATENTVIEW", "LAURUSLABS", "LEMONTREE", "LICI", "LINDEINDIA",
                                       "LLOYDSME", "LUPIN", "MMTC", "MRF", "LODHA", "MGL", "MAHSEAMLES", "M&MFIN",
                                       "M&M", "MAHLIFE", "MANAPPURAM", "MRPL", "MANKIND", "MARICO", "MARUTI", "MASTEK",
                                       "MFSL", "MAXHEALTH", "MAZDOCK", "METROBRAND", "METROPOLIS", "MINDACORP", "MSUMI",
                                       "MOTILALOFS", "MPHASIS", "MCX", "MUTHOOTFIN", "NATCOPHARM", "NBCC", "NCC",
                                       "NHPC", "NLCINDIA", "NMDC", "NSLNISP", "NTPC", "NH", "NATIONALUM", "NAVINFLUOR",
                                       "NESTLEIND", "NETWEB", "NETWORK18", "NEWGEN", "NAM-INDIA", "NUVAMA", "NUVOCO",
                                       "OBEROIRLTY", "ONGC", "OIL", "OLECTRA", "PAYTM", "OFSS", "POLICYBZR", "PCBL",
                                       "PIIND", "PNBHOUSING", "PNCINFRA", "PTCIL", "PVRINOX", "PAGEIND", "PATANJALI",
                                       "PERSISTENT", "PETRONET", "PFIZER", "PHOENIXLTD", "PIDILITIND", "PEL",
                                       "PPLPHARMA", "POLYMED", "POLYCAB", "POONAWALLA", "PFC", "POWERGRID", "PRAJIND",
                                       "PRESTIGE", "PGHH", "PNB", "QUESS", "RRKABEL", "RBLBANK", "RECLTD", "RHIM",
                                       "RITES", "RADICO", "RVNL", "RAILTEL", "RAINBOW", "RAJESHEXPO", "RKFORGE", "RCF",
                                       "RATNAMANI", "RTNINDIA", "RAYMOND", "REDINGTON", "RELIANCE", "ROUTE", "SBFC",
                                       "SBICARD", "SBILIFE", "SJVN", "SKFINDIA", "SRF", "SAMMAANCAP", "MOTHERSON",
                                       "SANOFI", "SAPPHIRE", "SAREGAMA", "SCHAEFFLER", "SCHNEIDER", "SCI", "SHREECEM",
                                       "RENUKA", "SHRIRAMFIN", "SHYAMMETL", "SIEMENS", "SIGNATURE", "SOBHA",
                                       "SOLARINDS", "SONACOMS", "SONATSOFTW", "STARHEALTH", "SBIN", "SAIL", "SWSOLAR",
                                       "SUMICHEM", "SPARC", "SUNPHARMA", "SUNTV", "SUNDARMFIN", "SUNDRMFAST",
                                       "SUPREMEIND", "SUVENPHAR", "SUZLON", "SWANENERGY", "SYNGENE", "SYRMA", "TBOTEK",
                                       "TVSMOTOR", "TVSSCS", "TANLA", "TATACHEM", "TATACOMM", "TCS", "TATACONSUM",
                                       "TATAELXSI", "TATAINVEST", "TATAMOTORS", "TATAPOWER", "TATASTEEL", "TATATECH",
                                       "TTML", "TECHM", "TECHNOE", "TEJASNET", "NIACL", "RAMCOCEM", "THERMAX", "TIMKEN",
                                       "TITAGARH", "TITAN", "TORNTPHARM", "TORNTPOWER", "TRENT", "TRIDENT", "TRIVENI",
                                       "TRITURBINE", "TIINDIA", "UCOBANK", "UNOMINDA", "UPL", "UTIAMC", "UJJIVANSFB",
                                       "ULTRACEMCO", "UNIONBANK", "UBL", "UNITDSPR", "USHAMART", "VGUARD", "VIPIND",
                                       "DBREALTY", "VTL", "VARROC", "VBL", "MANYAVAR", "VEDL", "VIJAYA", "VINATIORGA",
                                       "IDEA", "VOLTAS", "WELCORP", "WELSPUNLIV", "WESTLIFE", "WHIRLPOOL", "WIPRO",
                                       "YESBANK", "ZFCVINDIA", "ZEEL", "ZENSARTECH", "ZOMATO", "ZYDUSLIFE", "ECLERX"],
                               options=["Algorithm"])


@app.route("/terms")
def terms():
    return render_template("terms.html")