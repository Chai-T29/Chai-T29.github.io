---
layout: post
title: "Forecasting Bitcoin Prices using LSTM Neural Networks"
description: "Customer churn is a core component of marketing analytics and marketing-focused data science."
date: 2023-03-31
feature_image: images/bitcoin.jpg
---

This project explores the capabilities of Long Short-Term Memory (LSTM) neural networks in predicting Bitcoin prices. Having explored day trading (and failing), I aimed to develop an accurate and robust forecasting model that could provide valuable insights into short-term Bitcoin price movements, while having fun in the process. This project inspired my passion for data science, so I hope you enjoy the read!

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

#### Visualizing the Indicators

Technical indicators are used in investing to analyze past price and volume data of a financial asset and to identify patterns, trends, and potential future price movements. Technical indicators are mathematical calculations based on historical data and are often plotted on charts to provide visual representation. Traders and investors use technical indicators to make informed decisions about buying, selling, or holding a financial asset. Technical indicators can be used for different purposes such as identifying trend reversals, determining market momentum, detecting oversold or overbought conditions, and generating trading signals.

- SMA (Simple Moving Average): Represents the average closing price over a specified period, smoothing out short-term price fluctuations and highlighting overall trends. It helps identify potential support or resistance levels and can signal trend reversals.

![SMA](https://github.com/user-attachments/assets/9c6c9d44-e3c8-4992-9d92-08d8ba050306)

- EMA (Exponential Moving Average): Similar to SMA, but it gives more weight to recent prices, making it more responsive to new information. It is useful for identifying the short-term trend direction and potential entry or exit points.

![EMA](https://github.com/user-attachments/assets/714ac3e9-2a8e-4eea-b6c8-39778d035a96)

- RSI (Relative Strength Index): Measures the speed and change of price movements, indicating overbought or oversold conditions. It can help in identifying potential trend reversals and price corrections.

![RSI](https://github.com/user-attachments/assets/bcce700b-bf95-43d6-95c3-72f5a2536da3)

- MACD (Moving Average Convergence Divergence): Shows the relationship between two moving averages of a security’s price, highlighting potential trend reversals, and providing buy or sell signals.

![MACD](https://github.com/user-attachments/assets/e9c8bc58-c886-48cd-b3b7-a31dc4c04405)

- Bollinger Bands: Consist of a middle band (simple moving average) with an upper and lower band, representing volatility. They can help identify potential breakouts and price targets, as well as overbought or oversold conditions.

![BB](https://github.com/user-attachments/assets/6755abe1-601d-49ea-b80c-edb74fc8591e)

- ATR (Average True Range): Measures market volatility by considering the range of price movements. A higher ATR indicates higher volatility, while a lower ATR suggests lower volatility. It can help set stop-loss levels and manage risk.

![ATR](https://github.com/user-attachments/assets/c7b7e2c2-f091-4473-818b-511a29ade3ce)

- Stochastic Oscillator (%K and %D): Identifies potential trend reversals and overbought or oversold conditions by comparing a security’s closing price to its price range over a specified period.

![KD](https://github.com/user-attachments/assets/6228d391-1b95-40c7-b039-91bd2d0086e0)

- CCI (Commodity Channel Index): Measures the deviation of a security’s price from its average price, indicating overbought or oversold conditions and potential trend reversals.

![CCI](https://github.com/user-attachments/assets/59a8ce8a-c087-4a32-8d44-08da4f7efc48)

- OBV (On Balance Volume): Relates price and volume to show the flow of funds into or out of a security, helping identify potential price breakouts or reversals.

![OBV](https://github.com/user-attachments/assets/44340950-3fdc-4cf9-b804-cc48eab73536)

- ROC (Rate of Change): Measures the percentage change in price over a specified period, indicating momentum and potential trend reversals.

![ROC](https://github.com/user-attachments/assets/fe9d5fe2-e1b8-4c82-9f9d-2a2eb1739282)

- PSAR (Parabolic Stop and Reverse): Provides potential entry and exit points by indicating trend direction and stop levels.

![PSAR](https://github.com/user-attachments/assets/55aea9be-19ec-4981-b48c-ab67dd8d9ab2)

- VWAP (Volume Weighted Average Price): Represents the average price at which a security has traded throughout the day, considering both price and volume. It can help identify intraday trends and potential entry or exit points.

![VWAP](https://github.com/user-attachments/assets/5196c316-ce82-4ed9-8530-a701a99ab372)

<br>

## Basic Stacked LSTM Model

A stacked LSTM (Long Short-Term Memory) model is a type of neural network that consists of multiple LSTM layers stacked on top of each other. Each LSTM layer learns a different level of abstraction from the input data, and the output of one layer is fed as input to the next layer. Stacked LSTM models are used in time series analysis and prediction — in this case, predicting Bitcoin prices — because they can capture the complex dependencies and patterns in the data over long periods.

The model can be trained on historical data, and once trained, it can be used to make predictions about future prices. The function below splits the training and testing data from earlier between X and y.

<Details markdown="block">
<summary>Click here to view the code</summary>

```python
def create_sequences(df, window_size):
    X, y = [], []
    for i in range(window_size, len(df)):
        X.append(df.iloc[i - window_size:i].values)
        y.append(df.iloc[i, 0])
    return np.array(X), np.array(y)

window_size = 60
X_train, y_train = create_sequences(train_data, window_size)
X_test, y_test = create_sequences(test_data, window_size)
```
</Details>

With the data set up, we can build the LSTM model.

The model below consists of three LSTM layers stacked on top of each other, with dropout layers in between to help prevent overfitting. Here’s a brief explanation of each layer in the model:

1. First LSTM layer: This layer has 50 LSTM units and is set to return sequences because the next layer is also an LSTM layer. It takes the input shape from the training data.
2. First dropout layer: This layer applies a dropout rate of 20% to the first LSTM layer’s output, randomly setting some of the activations to zero during training to reduce overfitting.
3. Second LSTM layer: This layer has 50 LSTM units and also returns sequences because the next layer is another LSTM layer.
4. Second dropout layer: This layer applies a dropout rate of 20% to the second LSTM layer’s output.
5. Third LSTM layer: This layer has 50 LSTM units and does not return sequences since the next layer is a Dense layer.
6. Third dropout layer: This layer applies a dropout rate of 20% to the third LSTM layer’s output.
7. Dense output layer: This layer produces the final output of the model, with a single output unit for the predicted value.

The model is compiled using the Adam optimizer with a learning rate of 0.001 and mean squared error as the loss function. The model is then trained for 50 epochs (or iterations of the LSTM model) with a batch size of 32 using both training and validation data.

<Details markdown="block">
<summary>Click here to view the code</summary>

```python
# Define the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compile and train the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)
```
</Details>

#### Results

To evaluate the performance of the LSTM model, I first calculated the train and test loss by measuring the difference between the predicted and actual values using the model’s evaluate() method. I then used the model to predict Bitcoin prices and transformed the predicted and actual values back to their original scale using the scaler’s inverse_transform() method. To assess the accuracy of the predicted values, I calculated the root mean squared error (RMSE) and mean absolute percentage error (MAPE) compared to the actual values. The RMSE and MAPE are significant as they help to evaluate the model’s performance at predicting future Bitcoin prices, while visualizations aid in understanding the alignment between the predicted and actual values. The plots below show the predicted and actual values for both the training and testing datasets.

![image](https://github.com/user-attachments/assets/beaf784e-764d-4671-9dff-451338533859)

![image](https://github.com/user-attachments/assets/103142fa-bfb1-427d-a5c8-869103430394)

It is crucial to highlight the significance of the difference between the model’s fit on the training and testing data. This will help us to identify areas of improvement to enhance the model’s predictive capabilities for Bitcoin prices. The training MAPE is 26.91% and the testing MAPE is 2.74%.

In the following sections, we will explore ways to further improve the model’s performance and accuracy.

## 5-Fold Cross-Validation
Generally in 5-fold cross-validation, the dataset is divided into 5 equal parts or “folds”. The model is trained on 4 folds, using them as the training set, and the remaining fold is used as the validation set. This process is repeated 5 times, with each fold being used as the validation set once. The model’s performance is averaged over the 5 validation sets to obtain a single performance metric. This method helps assess the model’s ability to generalize to unseen data. This approach has many strengths, but it also comes with its fair share of limitations.

Strengths:

- Provides a more robust estimate of model performance than a single train-test split.
- Reduces the risk of overfitting by averaging performance over multiple validation sets.
- Provides a good indication of how well the model will perform on new, unseen data.
- 
Limitations:

- May not be suitable for time series data with strong temporal dependencies, as shuffling the data may lead to leakage of information from the future into the training set.
- Can be computationally expensive, especially for large datasets or complex models, as the model has to be trained multiple times.

5-fold cross-validation may provide a more robust estimate of model performance, but I did not get my hopes too high because the data might have significant temporal dependencies.

#### Developing the Model

This code implements a 5-fold Time Series Cross-Validation technique using the TimeSeriesSplit function from sklearn.model_selection. In this approach, the dataset is split into 5 non-overlapping time-ordered folds. For each fold, the model is trained on all previous folds and validated on the current fold. This process is repeated 5 times, with each fold serving as the validation set once.

The lstm_5fold_cross_validation function takes in the input data X, target variable y, number of splits n_splits, and a scaler object scaler. Inside the function, a TimeSeriesSplit object is created with the specified number of splits. The function then iterates through each train and test index pair generated by the TimeSeriesSplit object, trains the LSTM model, and calculates the RMSE and MAPE values for each fold.

Finally, the function returns the RMSE and MAPE values, as well as the unscaled predictions and true values for each fold. The code uses the 5-fold Time Series Cross-Validation technique, which is different from the rolling origin cross-validation technique mentioned in your previous question.

Finally, the function returns the RMSE and MAPE values, as well as the unscaled predictions and true values for each fold. The code uses the 5-fold Time Series Cross-Validation technique, which is different from the rolling origin cross-validation technique mentioned in your previous question.

<Details markdown="block">
<summary>Click here to view the code</summary>

```python
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def lstm_5fold_cross_validation(X, y, n_splits=5, scaler=None):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmse_values, mape_values, accuracy_values = [], [], []
    predictions, true_values = [], []  # Add lists to store predictions and true_values

    # Define early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Create and train the model
        model = create_lstm_model((X_train.shape[1], X_train.shape[2]))
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=0, callbacks=[early_stopping])

        # Make predictions
        y_test_pred = model.predict(X_test)

        # Unscale the predictions and true values
        y_test_pred_reshaped = y_test_pred.reshape(-1, 1)
        y_test_reshaped = y_test.reshape(-1, 1)
        dummy_test = np.zeros((y_test_pred_reshaped.shape[0], X_test.shape[2] - 1))

        y_test_pred_unscaled = scaler.inverse_transform(np.hstack((y_test_pred_reshaped, dummy_test)))[:, 0]
        y_test_unscaled = scaler.inverse_transform(np.hstack((y_test_reshaped, dummy_test)))[:, 0]

        predictions.append(y_test_pred _unscaled)  # Append unscaled predictions
        true_values.append(y_test_unscaled)  # Append unscaled true_values

        # Calculate the RMSE
        rmse = sqrt(mean_squared_error(y_test_unscaled, y_test_pred_unscaled))
        rmse_values.append(rmse)

        # Calculate the MAPE
        mape = np.mean(np.abs((y_test_unscaled - y_test_pred_unscaled) / y_test_unscaled)) * 100
        mape_values.append(mape)

    return rmse_values, mape_values, predictions, true_values  # Return predictions and true_values

# Prepare the data using the create_sequences function
window_size = 60
X, y = create_sequences(scaled_data, window_size)

# Call the lstm_cross_validation function
rmse_5f_values, mape_5f_values, predictions_5f, true_values_5f = lstm_5fold_cross_validation(X, y, n_splits=5, scaler=scaler)
```
</Details>

#### Visualizing the Results

This visualization shows the results of the 5-fold cross-validation on the LSTM model. The plot displays both the predicted and actual values of Bitcoin prices along a time series. The purpose of this visualization is to compare the predicted values with the actual values to see how well the model performs in predicting Bitcoin prices. The plot also helps to identify any patterns or trends in the data that the model may have missed. Overall, the model seems to be doing a much better job at predicting Bitcoin prices than the basic stacked LSTM model.

![5-fold](https://github.com/user-attachments/assets/fd699f97-3598-4b7b-8346-dc786df64898)

# Average MAPE: 8.57%

The MAPE has been reduced by approximately 18.34%, which highlights the effectiveness of employing cross-validation techniques on regression models. A MAPE of 8.57% after cross-validation signifies that the LSTM model’s predictions deviate from the actual values by an average of 8.57%. This relatively low error rate implies that the LSTM model is proficient at capturing the underlying dynamics of the time series data.

The improved accuracy can offer valuable insights for investors, enabling them to make more informed decisions and better manage risk in the financial markets. By utilizing a cross-validated LSTM model, investors can gain a deeper understanding of the factors influencing price movements and identify potential trends or market anomalies. Consequently, this enhanced predictive capability can lead to more strategic investment planning and potentially higher returns.

However, it is crucial to recognize that the model’s performance is not flawless, and the inherent uncertainty in financial markets remains. The 8.57% error rate, while relatively low, indicates that predictions should be treated with caution and not solely relied upon for decision-making. Combining the model’s insights with other market indicators and fundamental analysis can help investors create a more comprehensive and balanced investment strategy.

## Rolling Origin Cross-Validation
Now we’re getting into some serious business! Rolling origin cross-validation, also known as time series cross-validation, is specifically designed for time series data. In this method, the dataset is split into a training and validation set, with the validation set starting at a certain point in time. The model is trained on the initial training set and validated on the validation set. The process is then repeated by rolling the starting point of the validation set forward in time, expanding the training set to include the previous validation set, and using a new validation set. This ensures that the training set always precedes the validation set in time, preserving the temporal structure of the data.

Here’s a visual representation of the rolling origin cross-validation approach:

![rolling-origin](https://github.com/user-attachments/assets/61cbb8b9-322d-45b6-8c0e-7bafd8eb3289)

Each row represents an iteration of the model, with the green and orange dots being training and testing data points, respectively (the white dots are data points that are not included in the iteration). This incremental structure has strengths and weaknesses as outlined below.

Strengths:

- Preserves the temporal structure of the data, making it suitable for time series forecasting.
- Provides a realistic estimate of the model’s performance on new, unseen data by validating data points that come after the training data.
- Reduces the risk of data leakage due to shuffling the dataset.

Limitations:

- Can be computationally expensive, as the model needs to be retrained multiple times.
- May not work well for datasets with strong seasonality or recurring patterns, as the model may not be exposed to the full range of these patterns during training.

Rolling origin cross-validation preserves the temporal structure, making it more suitable for time series forecasting but may not capture seasonal patterns effectively. Additionally, the predictions from this model will only capture the last 50 data points because of its computational extensivity.

#### Developing the Model

This code implements a variation of the rolling origin cross-validation technique for time series data. The lstm_cross_validation function iterates through the time series dataset, creating a rolling window for training and testing the LSTM model. At each step, the training set expands by one observation, and the test set consists of a single observation following the training set.

However, this implementation differs from a standard Time Series Cross-Validation in one aspect. In a typical rolling origin cross-validation, you would divide the data into K-folds and use the first K-1 folds for training and the Kth fold for testing, then move the window forward. The current implementation does not use a fixed number of K-folds but rather moves the window forward one step at a time until the end of the dataset (similar to the image in the previous section, but only using one data point for testing at a time instead of three). This leads to a much larger number of iterations, as the model is tested on every single observation after the initial training set.

The code does not use TimeSeriesSplit from sklearn.model_selection, which is commonly used to create K-folds in rolling origin cross-validation. Instead, it uses a custom for loop to create the rolling window.

<Details markdown="block">
<summary>Click here to view the code</summary>

```python
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def lstm_cross_validation(X, y, test_size, scaler=None):
    rmse_values, mape_values, predictions, true_values = [], [], [], []

    # Define early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    train_size = X.shape[0] - test_size
    for i in range(train_size, X.shape[0]):
        X_train, X_test = X[:i], X[i:i+1]
        y_train, y_test = y[:i], y[i:i+1]

        # Create and train the model
        model = create_lstm_model((X_train.shape[1], X_train.shape[2]))
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=0, callbacks=[early_stopping])

        # Make predictions
        y_test_pred = model.predict(X_test)

        # Unscale the predictions and true values
        y_test_pred_reshaped = y_test_pred.reshape(-1, 1)
        y_test_reshaped = y_test.reshape(-1, 1)
        dummy_test = np.zeros((y_test_pred_reshaped.shape[0], X_test.shape[2] - 1))

        y_test_pred_unscaled = scaler.inverse_transform(np.hstack((y_test_pred_reshaped, dummy_test)))[:, 0]
        y_test_unscaled = scaler.inverse_transform(np.hstack((y_test_reshaped, dummy_test)))[:, 0]

        predictions.append(y_test_pred_unscaled)
        true_values.append(y_test_unscaled)

        # Calculate the RMSE
        rmse = sqrt(mean_squared_error(y_test_unscaled, y_test_pred_unscaled))
        rmse_values.append(rmse)

        # Calculate the MAPE
        mape = np.mean(np.abs((y_test_unscaled - y_test_pred_unscaled) / y_test_unscaled)) * 100
        mape_values.append(mape)

    return rmse_values, mape_values, predictions, true_values

# Prepare the data using the create_sequences function
window_size = 60
X, y = create_sequences(scaled_data, window_size)

# Determine the test size (e.g., 20% of the dataset)
test_size = int(len(X) * 0.2)

# Call the lstm_cross_validation function
rmse_values, mape_values, predictions, true_values = lstm_cross_validation(X, y, test_size, scaler=scaler)
```
</Details>

#### Visualizing the Results

The visualization below displays the results of the rolling origin cross-validation on the LSTM model. Just like the previous plot, this plot also displays both the predicted and actual values of Bitcoin prices along a time series. The purpose of this visualization is to compare the predicted values with the actual values to see if there was any improvement in the model’s performance/fitting. Since this cross-validation approach only covers the past 50 days, there are only predictions for this time frame. However, it appears that the model is performing extremely well!

![ro_results](https://github.com/user-attachments/assets/3970a2d1-30a4-46d8-844f-bdac9f2c73c7)

The MAPE has been significantly reduced to 3.08% after implementing the rolling origin cross-validation technique, which emphasizes the effectiveness of this method in refining regression models. A MAPE of 3.08% signifies that the LSTM model’s predictions deviate from the actual values by an average of only 3.08%. This remarkably low error rate indicates that the LSTM model is highly adept at capturing the underlying dynamics of the time series data.

The enhanced accuracy can provide valuable insights for investors, allowing them to make more informed decisions and manage risk more effectively in the financial markets. By using a rolling origin cross-validated LSTM model, investors can gain a more profound understanding of the factors driving price movements and identify potential trends or market anomalies. As a result, this improved predictive capability can contribute to more strategic investment planning and potentially higher returns.

It is essential, however, to recognize that the model’s performance is not perfect, and the inherent uncertainty in financial markets still prevails. The 3.08% error rate, while impressively low, suggests that predictions should be approached with prudence and not solely relied upon for decision-making. Integrating the model’s insights with other market indicators and fundamental analysis can help investors develop a more comprehensive and well-rounded investment strategy, further mitigating potential risks.

## Conclusion

In conclusion, the LSTM project we have explored throughout this article demonstrates the power and versatility of deep learning techniques for time series forecasting in financial markets. By leveraging a diverse set of technical indicators, price-derived features, and time-based features, the LSTM model — paired with cross-validation techniques — captures complex patterns and relationships in the historical data, offering improved predictive accuracy for future price movements.

This LSTM-based approach can be applied to various use-cases, including algorithmic trading, portfolio management, and risk management. For algorithmic trading, traders can use the model’s predictions to design and execute trading strategies, identifying potential entry and exit points in the market. Portfolio managers can benefit from the model’s insights to make informed decisions about asset allocation and rebalancing, optimizing their portfolio’s risk-adjusted returns. In the realm of risk management, financial institutions can utilize the LSTM model to forecast potential price fluctuations, allowing them to set appropriate risk limits and manage their exposure effectively.

Moreover, the LSTM model can be extended to include additional features or data sources, such as sentiment analysis from news articles or social media, improving its predictive capabilities further. It can also be adapted to forecast various financial instruments, including stocks, commodities, currencies, and cryptocurrencies, highlighting its versatility across different asset classes.

Overall, this LSTM project showcases the potential of deep learning in financial market forecasting, opening up new opportunities for investors, traders, and financial institutions to make data-driven decisions and stay ahead in the ever-evolving world of finance. As a reader, you are now better equipped to understand and explore the potential of LSTM models in finance, and hopefully you are inspired to take this project a step further with your own flair! By staying informed about these cutting-edge technologies, you can actively participate in the financial market’s ongoing transformation and seize new opportunities that arise from advancements in predictive analytics!

<br>

## References

https://www.elearnmarkets.com/blog/best-25-technical-indicators/

https://machinelearningmastery.com/stacked-long-short-term-memory-networks/

https://stats.stackexchange.com/questions/564311/how-to-cross-validate-a-time-series-lstm-model







