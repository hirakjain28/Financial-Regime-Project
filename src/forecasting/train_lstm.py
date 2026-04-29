import pandas as pd
import numpy as np
import os
import joblib

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

BASE_DIR = os.path.dirname(__file__)

df = pd.read_csv(r"F:\Study\Projects\Financial-Regime-Project\data\processed\nifty_clean.csv")

prices = df["Close"].values.reshape(-1,1)

# returns
returns = np.diff(prices, axis=0) / prices[:-1]

scaler = MinMaxScaler()
scaled = scaler.fit_transform(returns)

X = []
y = []

lookback = 20

for i in range(lookback, len(scaled)):
    X.append(scaled[i-lookback:i])
    y.append(scaled[i])

X = np.array(X)
y = np.array(y)

model = Sequential()
model.add(LSTM(64, input_shape=(lookback,1)))
model.add(Dense(1))

model.compile(loss="mse", optimizer="adam")
model.fit(X, y, epochs=20, batch_size=32)

model.save(os.path.join(BASE_DIR,"lstm_return.h5"))
joblib.dump(scaler, os.path.join(BASE_DIR,"return_scaler.save"))

print("Return LSTM Saved")