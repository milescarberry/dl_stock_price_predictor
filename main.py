import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import pandas_datareader as web

import datetime as dt          # This is a core python module.

from sklearn.preprocessing import MinMaxScaler


from tensorflow.keras.models import Sequential     # A Sequential model contains a "linear" stack of neural layers.

from tensorflow.keras.layers import Dense, Dropout, LSTM                    # "LSTM" stands for Long Short-Term Memory Layer.

# "LSTM" :- Long Short-Term Memory Layer

# A Sequential model contains a "linear" stack of neural layers.


# Load Data :-


company = "TCS"


start  = dt.datetime(2011,02,07)                  # From what timestamp we want to start collecting the company stock data.

stop = dt.datetime(2021,02,07)                    # Till what timestamp we want to collect the company stock data.



data = web.DataReader(company, "yahoo", start, stop)              # "data" here points to a "dict" object. "data" is a "nested" structure. (There is a "dict" inside a "dict")

# We are gonna collect the company "stock" data using the yahoo_api.


# Prepare Data :-     (Pre-Process the data before feeding it into the neural network)


scaler = MinMaxScaler(feature_range = (0, 1))                 # MinMaxScaler() belongs to the sklearn.preprocessing module.


scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))                   # The 'Close' key inside the "data" dictionary points to a dictionary.


# We need the "scaled_data" array in a particular shape. That's why we are "reshaping" the data['Close'].values arrays.
