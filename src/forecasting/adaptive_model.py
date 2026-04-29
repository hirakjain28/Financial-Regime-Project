from src.forecasting.arima_model import arima_forecast
from src.forecasting.lstm_model import lstm_forecast

def adaptive_forecast(df):

    forecasts = []
    dates = []

    start = len(df) - 100

    for i in range(start, len(df)):

        data_slice = df.iloc[:i+1]

        arima_pred = arima_forecast(data_slice)
        lstm_pred = lstm_forecast(data_slice)

        final_pred = 0.5 * arima_pred + 0.5 * lstm_pred

        forecasts.append(final_pred)
        dates.append(df.index[i])

    return forecasts, dates