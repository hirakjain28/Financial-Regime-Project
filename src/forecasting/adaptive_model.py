from src.forecasting.arima_model import arima_forecast

def adaptive_forecast(df):
    forecasts = []

    for i in range(50, min(150, len(df))):
        window = df.iloc[:i]

        regime = window['Regime'].iloc[-1]

        series = window['Close']

        # Simple logic (can improve later)
        if regime == 0:
            pred = arima_forecast(series)
        else:
            pred = arima_forecast(series)  # replace with LSTM later

        forecasts.append(pred)
    
    print(f"Processing step {i}")

    return forecasts