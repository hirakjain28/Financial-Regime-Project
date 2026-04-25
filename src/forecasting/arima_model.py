import joblib
import os

BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "arima_model.pkl")

model = joblib.load(model_path)

def arima_forecast(df):
    forecast = model.forecast(steps=1)
    return float(forecast.iloc[0])