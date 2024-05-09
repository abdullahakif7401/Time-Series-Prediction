import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error


def to_sequences(data, window_size):
    x = []
    y = []
    for i in range(len(data) - window_size):
        x.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(x), np.array(y)


def pinball_loss(y_true, y_pred, tau, delta=1.0):
    error = y_true - y_pred
    huber_loss = K.mean(K.maximum(K.abs(error) - delta, 0), axis=-1)
    return K.mean(K.maximum(tau * error, (tau - 1) * error), axis=-1) + huber_loss


df = pd.read_csv('LD2011_2014.txt', sep=';', decimal=',')
df.rename(columns={"Unnamed: 0": 'Datetime'}, inplace=True)
df.index = pd.to_datetime(df["Datetime"], format="%Y-%m-%d %H:%M:%S")
df = df.resample('h').sum()
df_valid = df.loc["2012-01-01":]
random_df = df_valid[random.sample(list(df_valid.columns[1:]), 30)]

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(random_df)

train_end_index = len(random_df[random_df.index <= "2012-12-31"])
validation_end_index = len(random_df[random_df.index <= "2013-01-31"])
test_end_index = len(random_df[random_df.index <= "2013-02-28"])

train_data = scaled_data[:train_end_index]
validation_data = scaled_data[train_end_index:validation_end_index]
test_data = scaled_data[validation_end_index:test_end_index]

window_size = 3
X_train, y_train = to_sequences(train_data, window_size)
X_val, y_val = to_sequences(validation_data, window_size)
X_test, y_test = to_sequences(test_data, window_size)

model = Sequential([
    LSTM(units=256, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    LSTM(units=256, return_sequences=True),
    LSTM(units=256),
    Dense(units=30, activation='sigmoid')
])

tau = 0.5
model.compile(loss=lambda y_true, y_pred: pinball_loss(y_true, y_pred, tau), optimizer=Adam(learning_rate=1e-4))
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=256)

y_pred = model.predict(X_test)
print("Root Mean Squared Error: " + str(np.sqrt(mean_squared_error(y_test, y_pred))))
plt.plot(y_test[:, 0], label='True')
plt.plot(y_pred[:, 0], label='Predicted')
plt.xlabel('Hours')
plt.ylabel('Electricity Load (kWh)')
plt.legend()
plt.show()