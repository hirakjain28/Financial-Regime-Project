import joblib
import os
import numpy as np

BASE_DIR = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE_DIR, "arima_model.pkl"))

def arima_forecast(data):

    prices = data["Close"].values.astype(float)
    current_price = prices[-1]

    pred = model.forecast(steps=1)
    pred_return = float(pred.iloc[0])

    # Clamp return safely
    pred_return = np.clip(pred_return, -0.05, 0.05)

    # NORMAL return conversion
    pred_price = current_price * (1 + pred_return)

    return float(pred_price)