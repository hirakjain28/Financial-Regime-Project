import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.arima.model import ARIMA

def arima_forecast(series):
    model = ARIMA(series, order=(1,1,1))
    fitted = model.fit()
    forecast = fitted.forecast(steps=1)
    return forecast.iloc[0]