import numpy as np
import os
import joblib

from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(__file__)

model = load_model(
    os.path.join(BASE_DIR,"lstm_return.h5"),
    compile=False
)

scaler = joblib.load(
    os.path.join(BASE_DIR,"return_scaler.save")
)

def lstm_forecast(data):

    prices = data["Close"].values.reshape(-1,1)

    returns = np.diff(prices, axis=0) / prices[:-1]

    last_seq = returns[-20:]

    scaled = scaler.transform(last_seq)

    X = scaled.reshape(1,20,1)

    pred_scaled = model.predict(X, verbose=0)

    pred_return = scaler.inverse_transform(pred_scaled)[0][0]

    current_price = prices[-1][0]
    pred_return = np.clip(pred_return, -0.05, 0.05)
    pred_price = current_price * (1 + pred_return)

    return pred_price