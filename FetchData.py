import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta

current_date = datetime.now()

six_months_back = current_date - timedelta(days=30*7)

one_year_back = current_date - timedelta(days=30*12)


months_list = []

# Generate the strings for the last 6 months
for i in range(7):  # Increase the range to 7 to include the current month
    # Calculate the month
    month_date = six_months_back + timedelta(days=30*i)
    # Append the formatted string to the list
    months_list.append(month_date.strftime('%Y-%m'))

months_list.pop(0)

STOCK_TICKER = "AAPL"
STOCK_NAME = "AAPL"

data_dict = {'time': [],
             'open': [],
             'high': [],
             'low': [],
             'close': [],
             'volume': []}

for month in months_list:
    url = (f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={STOCK_TICKER}&interval=60min'
                f'&month={month}&outputsize=full&adjusted=true&datatype=json&extended_hours=false&apikey=demo')
    r = requests.get(url)
    data = r.json()

    for t, d in data['Time Series (60min)'].items():
        data_dict['time'].append(t)
        data_dict['open'].append(d['1. open'])
        data_dict['high'].append(d['2. high'])
        data_dict['low'].append(d['3. low'])
        data_dict['close'].append(d['4. close'])
        data_dict['volume'].append(d['5. volume'])

stocks_df = pd.DataFrame(data_dict)
stocks_df.sort_values(by=['time'], inplace=True)
stocks_df.to_csv(f'{STOCK_NAME}_data_train.csv', index_label='index')