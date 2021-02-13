import numpy as np

from nsepy import get_history

import matplotlib.pyplot as plt

import pandas as pd

import pandas_datareader as web

import datetime as dt          # This is a core python module.

from sklearn.preprocessing import MinMaxScaler


from tensorflow.keras.models import Sequential     # A Sequential model contains a "linear" stack of neural layers.

from tensorflow.keras.layers import Dense, Dropout, LSTM                    # "LSTM" stands for Long Short-Term Memory Layer.

from tensorflow.keras.optimizers import Adam


# "LSTM" :- Long Short-Term Memory Layer

# A Sequential model contains a "linear" stack of neural layers.


# Load Data :-


company = "IDBI"


start  = dt.datetime(2019,1,1)                  # From what timestamp we want to start collecting the company stock data.

end = dt.datetime(2020,1,1)                    # Till what timestamp we want to collect the company stock data.



#data = web.DataReader(company, "yahoo", start, stop)              # "data" here points to a "dict" object. "data" is a "nested" structure. (There is a "dict" inside a "dict")

# We are gonna collect the company "stock" data using the yahoo_api.

data = get_history(symbol = company, start = start, end = end)


# Prepare Data :-     (Pre-Process the data before feeding it into the neural network)


scaler = MinMaxScaler(feature_range = (0, 1))                 # MinMaxScaler() belongs to the sklearn.preprocessing module.


# "scaler" is an instance of the sklearn.preprocessing.MinMaxScaler class.


scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))                   # The 'Close' key inside the "data" dictionary points to a dictionary.


# We need the "scaled_data" array in a particular shape. That's why we are "reshaping" the data['Close'].values arrays.



prediction_days = 60               # How many "days" should the "model" look back to predict the "closing stock price" of the next day.


x_train = []

y_train = []


for x in range(prediction_days, len(scaled_data)):

    x_train.append(scaled_data[x-prediction_days:x, 0])               # append the list containing 120 values from scaled_data into x_train.


    y_train.append(scaled_data[x, 0])                  # Then append the list containing the 121st value from scaled_data into y_train.





x_train , y_train = np.array(x_train), np.array(y_train)






x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))        # We are gonna add another dimension to our x_train array.




# We have "reshaped" our x_train array so that we can feed it into our neural network in the future.



# Now, let's build the model :-

# A "sequential" model comprises of a linear stack of neural layers.


model = Sequential()


model.add(LSTM(units = 100, return_sequences = True, input_shape = (x_train.shape[1], 1)))

model.add(Dropout(0.2))


# LSTM layer is a recurrent layer. It's not just gonna feed forward the information. It's also gonna feed back the information into the LSTM layer itself.


model.add((LSTM(units = 100, return_sequences = True)))

model.add(Dropout(0.2))


model.add((LSTM(units = 100)))                    # We are not gonna "return the sequences" here.

model.add(Dropout(0.2))


model.add(Dense(units = 1))                          # This is going to be our *output layer*. This layer contains the "prediction" of the next "closing stock price" of the company.


# The one neuron in the output layer is going to provide the prediction on the "next closing price" of our stock.



model.compile(optimizer = 'adam', loss = 'mean_squared_error')


model.fit(x_train, y_train, epochs = 75, batch_size = 32)




''' Test The Model Accuracy On Existing Data'''


# Load the test data :-


test_start = dt.datetime(2020,1,1)

test_end = dt.datetime.now()


# "test_start" to "test_end" is going to be the range of our test data. This test data has not been seen by our model.


#test_data = web.DataReader(company, 'yahoo', test_start, test_end)

test_data = get_history(symbol = company, start = test_start, end = test_end)


actual_closing_prices = test_data['Close'].values


total_dataset = pd.concat((data['Close'], test_data['Close']), axis = 0)


model_inputs = total_dataset[len(total_dataset)- len(test_data)- prediction_days::1].values

model_inputs = model_inputs.reshape((model_inputs.size, 1))


model_inputs = scaler.transform(model_inputs)


# scaler.transform(data_array) function is going to "scale" (compress) the values (closing stock prices) inside the array between 0 and 1. (So that the neural network of the model recognizes those values)



# Make Predictions On Our Test Data :-


x_test = []


for x in range(prediction_days, len(model_inputs)):

    x_test.append(model_inputs[x-prediction_days:x:1, 0])


x_test = np.array(x_test)


x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))            # We have added one more dimension to the x_test numpy array.


predicted_closing_prices = model.predict(x_test)                   # Pass in the input_data (x_test) for our model to work upon.



# Now, the predicted closing prices that we are going to get will be "scaled" (compressed between 0 and 1).

# So, we will "reverse scale" the predicted closing prices to get the actual values.

# So, we will "inverse_transform" our predicted_closing_prices.


predicted_closing_prices = scaler.inverse_transform(predicted_closing_prices)




# Now, let's plot the test predictions on a graph :-


plt.plot(actual_closing_prices, color = "blue", label = "Predicted Closing Price")

plt.plot(predicted_closing_prices, color = "red", label = "Actual Closing Price")

plt.title(f"{company} Share Price Graph")

plt.xlabel("Time")

plt.ylabel("Closing Share Price")

plt.legend()

plt.show()                     # Display the graph.
