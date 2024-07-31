---
layout: post
title: "Forecasting Bitcoin Prices using LSTM Neural Networks"
description: "Customer churn is a core component of marketing analytics and marketing-focused data science."
date: 2023-03-31
feature_image: images/bitcoin.jpg
---

This project explores the capabilities of Long Short-Term Memory (LSTM) neural networks in predicting Bitcoin prices. Having explored day trading (and failing), I aimed to develop an accurate and robust forecasting model that could provide valuable insights into short-term Bitcoin price movements, while having fun in the process. If you are a cryptocurrency trader with a basic understanding of Python or a data science geek, then this article is for you!

<!--more-->

The results of my project were encouraging. Using 5-fold cross-validation, the Mean Absolute Percentage Error (MAPE), which is a measure of error in a model, was reduced by approximately 18.34% compared to the initial model, reaching a value of 8.57%! This indicated that the LSTM model’s predictions deviated from the actual prices of Bitcoin by an average of 8.57%. Furthermore, the rolling origin cross-validation approach yielded even more impressive results, with the MAPE further decreasing to 3.08%! However, does this mean that the model is actually performing well? We'll explore the process, results, and implications of this project. Don’t worry if you don’t understand what these models mean or do, I will explain everything in more detail throughout this article.

## Contents

The article is split into five main sections:

1.  [Setting up the Data](#setting-up-the-data)
2.  [Basic Stacked LSTM Model](#basic-stacked-lstm-model)
3.  [5-fold Cross-Validation](#5-fold-cross-validation)
4.  [Rolling Origin Cross-Validation](#rolling-origin-cross-validation)
5.  [Conclusion](#conclusion)

<br>

## Setting up the Data

Like any coding project, let's first import the necessary libraries:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import talib
import requests
import requests
import csv
import json
```

#### Requesting Data from CryptoCompare

I retrieved Bitcoin price data from CryptoCompare’s API using the requests library. To sign up for a free API key for crypto pricing data, visit: https://www.cryptocompare.com/

The function below fetches data for the past 364 days using the limit parameter and the current time using the toTs parameter. The response is then converted to JSON format, and the relevant data is extracted using the Data key. A pandas DataFrame is then created from the extracted data, with the time column converted to datetime format. Finally, the function returns the DataFrame containing the historical data.

<Details markdown="block">
<summary>Click here to view the code</summary>

```python
def fetch_historical_data():
    url = "https://min-api.cryptocompare.com/data/v2/histoday"
    params = {
        'fsym': 'BTC',
        'tsym': 'USD',
        'limit': 364,
        'toTs': int(pd.Timestamp.utcnow().timestamp()),
        'api_key': API_KEY
    }
    response = requests.get(url, params=params)
    json_data = response.json()
    print(json_data)  # Print json_data to debug

    historical_data = json_data['Data']['Data']
    df = pd.DataFrame(historical_data)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

# Test the function
df = fetch_historical_data()
df.head()
```
</Details>

#### Feature Engineering

The code block below creates technical indicators for Bitcoin’s historical daily price data and then prepares the data for our LSTM model. Technical indicators such as SMA, EMA, RSI, MACD, Bollinger Bands, ATR, Stochastic Oscillator, CCI, OBV, ROC, PSAR, and VWAP (I will explain more about these indicators in the next section) are calculated using the TA-Lib library. Additional price-derived features such as daily return and log return and time-based features such as day of the week are also calculated. Missing values are then dropped, and only the desired features are selected for further analysis. The data is then scaled using the MinMaxScaler, and the scaled data is split into training and testing sets with a 75/25 ratio. The resulting train_data and test_data DataFrames are used for model training and evaluation, respectively.

<Details markdown="block">
<summary>Click here to view the code</summary>

```python
# Technical Indicators
df['SMA'] = talib.SMA(df['close'], timeperiod=14)  # Simple Moving Average (14 days)
df['EMA'] = talib.EMA(df['close'], timeperiod=14)  # Exponential Moving Average (14 days)
df['RSI'] = talib.RSI(df['close'], timeperiod=14)  # Relative Strength Index (14 days)
df['MACD'], df['MACD_signal'], _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)  # Moving Average Convergence Divergence
df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['close'], timeperiod=20)  # Bollinger Bands (20 days)
df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)  # Average True Range (14 days)
df['%K'], df['%D'] = talib.STOCH(df['high'], df['low'], df['close'], fastk_period=14, slowk_period=3, slowd_period=3)  # Stochastic Oscillator
df['CCI'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)  # Commodity Channel Index (14 days)
df['OBV'] = talib.OBV(df['close'], df['volumeto'])  # On Balance Volume
df['ROC'] = talib.ROC(df['close'], timeperiod=10)  # Rate of Change (10 days)
df['PSAR'] = talib.SAR(df['high'], df['low'], acceleration=0.02, maximum=0.2)  # Parabolic Stop and Reverse

# Calculate VWAP
typical_price = (df['high'] + df['low'] + df['close']) / 3
cumulative_typical_price_volume = (typical_price * df['volumeto']).cumsum()
cumulative_volume = df['volumeto'].cumsum()

df['VWAP'] = cumulative_typical_price_volume / cumulative_volume

# Price-derived features
df['daily_return'] = df['close'].pct_change()  # Daily return
df['log_return'] = np.log(df['close'] / df['close'].shift(1))  # Log return

# Time-based features
df['day_of_week'] = df['time'].dt.dayofweek  # Day of the week (0: Monday, 1: Tuesday, ..., 6: Sunday)

# Drop missing values
df.dropna(inplace=True)

# Select only the columns that you want to use as features
feature_columns = ['SMA', 'EMA', 'RSI', 'MACD', 'MACD_signal', 
                   'BB_upper', 'BB_middle', 'BB_lower', 'daily_return', 'log_return', 
                   'day_of_week', 'ATR', '%K', 'CCI', 'OBV', 'ROC', 'PSAR', 'VWAP']
numerical_data = df[feature_columns]

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_numerical_data = scaler.fit_transform(numerical_data)

# Create a new DataFrame with scaled data
scaled_data = pd.DataFrame(scaled_numerical_data, columns=numerical_data.columns, index=numerical_data.index)

# Split the data into training and testing sets
train_data, test_data = scaled_data[:int(len(df) * 0.75)], scaled_data[int(len(df) * 0.75):]
```
</Details>





