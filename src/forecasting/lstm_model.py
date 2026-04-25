import os
import joblib
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "lstm_model.h5")
scaler_path = os.path.join(BASE_DIR, "scaler.save")

model = load_model(model_path, compile=False)
scaler = joblib.load(scaler_path)

def lstm_forecast(series):

    data = series.values.reshape(-1, 1)
    data_scaled = scaler.transform(data)

    window = 20

    if len(data_scaled) < window:
        return series.iloc[-1]

    last_window = data_scaled[-window:]
    last_window = last_window.reshape(1, window, 1)

    pred = model.predict(last_window, verbose=0)

    return scaler.inverse_transform(pred)[0][0]