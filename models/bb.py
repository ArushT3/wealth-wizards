import pandas as pd
from fyers_apiv3 import fyersModel
import yfinance as yf
import schedule
import time
from datetime import datetime


client_id = open("./client_id.txt", 'r').read()
access_token = open("./access_token.txt", 'r').read()


fyers = fyersModel.FyersModel(client_id=client_id, is_async=False, token=access_token, log_path="")

def get_net_quantity(data, symbol_to_find):
    if 'netPositions' in data:
        for position in data['netPositions']:
            if position['symbol'] == symbol_to_find:
                return position['netQty']
    return None

def get_symbol_list(data):
    if 'netPositions' in data:
        return [position['symbol'] for position in data['netPositions']]
    return []



def placeOrder(inst ,t_type,qty,order_type,price=0, price_stop=0):
    global type1, side1
    symb = inst[4:]
    dt = datetime.now()
    print(dt.hour,":",dt.minute,":",dt.second ," => ",t_type," ",symb," ",qty," ",order_type," @ price =  ",price)
    if order_type== "MARKET":
        type1 = 2
        price = 0
        price_stop = 0
    elif order_type== "LIMIT":
        type1 = 1
        price_stop = 0
    elif order_type== "SL-LIMIT":
        type1 = 4

    if t_type== "BUY":
        side1=1
    elif t_type== "SELL":
        side1=-1

    data =  {
        "symbol":inst,
        "qty":qty,
        "type":type1,
        "side":side1,
        "productType":"INTRADAY",
        "limitPrice":price,
        "stopPrice":price_stop,
        "validity":"DAY",
    }

    try:
        orderid = fyers.place_order(data)
        print(dt.hour,":",dt.minute,":",dt.second ," => ", symb , orderid)
        return orderid
    except Exception as e:
        print(dt.hour,":",dt.minute,":",dt.second ," => ", symb , "Failed : {} ".format(e))

def my_task():
    with open("stocks.txt", "r") as file:
        # Loop through each line in the file
        for line in file:
            print(line.strip().split()[0] + ".NS")
            ticker = line.strip().split()[0] + ".NS"
            data = yf.download(tickers=ticker, interval="1m", period="1d")
            data.head()
            data.reset_index(inplace=True)
            data.insert(0, "Datetime", data.pop("Datetime"))
            output_file_path = "temp/" + line.strip() + ".csv"
            data.to_csv(output_file_path, index=False)

            file_path = "temp/" + line.strip() + ".csv"
            with open(file_path, 'r') as filed:
                lines = filed.readlines()
            lines = lines[:1] + lines[2:]
            modified_file_path = "temp/" + line.strip() + ".csv"
            with open(modified_file_path, 'w') as filed:
                filed.writelines(lines)

    with open("stocks.txt", "r") as file:
        for line in file:
            lin = line.strip().split()
            print("hi")
            print(lin)
            file_name = "temp/" + lin[0] + ".csv"
            data = pd.read_csv(file_name, parse_dates=['Datetime'])
            data = data.sort_values('Datetime')

            window = int(lin[1])
            dev = float(lin[2])
            upper_band = []
            lower_band = []
            sma_list = []

            for i in range(len(data)):
                if i >= window - 1:
                    sum_close = 0
                    for j in range(window):
                        sum_close += data.iloc[i - j]['Close']
                    sma = sum_close / window
                    sma_list.append(sma)

                    sum_squared_diff = 0
                    for j in range(window):
                        diff = data.iloc[i - j]['Close'] - sma
                        sum_squared_diff += diff ** 2
                    std_dev = (sum_squared_diff / window) ** 0.5

                    upper_band.append(sma + dev * std_dev)
                    lower_band.append(sma - dev * std_dev)
                else:
                    sma_list.append(None)
                    upper_band.append(None)
                    lower_band.append(None)

            data['SMA'] = sma_list
            data['UpperBand'] = upper_band
            data['LowerBand'] = lower_band
            last_row_tail = data.tail(1)
            print(last_row_tail)
            dataa = fyers.positions()
            symbol_list = get_symbol_list(dataa)
            last_row_column_value = data.iloc[-1]["Close"]
            net_quantity = get_net_quantity(dataa, ("NSE:" + lin[0] + "-EQ"))
            if (("NSE:" + lin[0] + ":EQ") in symbol_list) and net_quantity != 0:
                if net_quantity < 0:
                    if last_row_column_value >= last_row_tail["SMA"].iloc[0]:
                        placeOrder("NSE:" + lin[0] + "-EQ", 'BUY', 2, "MARKET")
                else:
                    if last_row_column_value <= last_row_tail["SMA"].iloc[0]:
                        placeOrder("NSE:" + lin[0] + "-EQ", 'SELL', 2, "MARKET")
            else:
                if last_row_tail["Close"].iloc[0] > last_row_tail["UpperBand"].iloc[0]:
                    placeOrder("NSE:" + lin[0] + "-EQ", 'SELL', 2, "MARKET")
                if last_row_tail["Close"].iloc[0] < last_row_tail["LowerBand"].iloc[0]:
                    placeOrder("NSE:" + lin[0] + "-EQ", 'BUY', 2, "MARKET")
    print(f"Task executed at {datetime.now()}")

# Schedule the task every minute between 9:20 AM and 3:15 PM
def schedule_task():
    current_time = datetime.now().strftime("%H:%M")
    if "09:20" <= current_time <= "15:15":
        my_task()

# Run the schedule
while True:
    schedule.every(1).minutes.do(schedule_task)  # Schedule task every minute
    schedule.run_pending()
    time.sleep(1)  # Prevent high CPU usage

