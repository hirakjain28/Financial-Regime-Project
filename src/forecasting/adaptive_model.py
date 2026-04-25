import numpy as np
from src.forecasting.arima_model import arima_forecast
from src.forecasting.lstm_model import lstm_forecast

def adaptive_forecast(df):
    start = len(df) - 200
    window_size = 200
    forecasts = []
    forecast_dates = []

    for i in range(start, len(df)):

        if i % 10 != 0:   # optional speed boost
            continue

        window = df.iloc[i-window_size:i]

        series = window['Returns'].dropna()
        last_price = window['Close'].iloc[-1]
        
        pred_arima = arima_forecast(series)
        pred_lstm = lstm_forecast(series)

        pred_return = 0.7 * pred_arima + 0.3 * pred_lstm
        pred_return = np.clip(pred_return, -0.03, 0.03)
        pred_price = last_price * np.exp(pred_return)
        
        if not np.isfinite(pred_price):
            pred_price = last_price

        forecasts.append(pred_price)
        forecast_dates.append(df.index[i])

        print(f"Step {i} done")

    return forecasts, forecast_dates