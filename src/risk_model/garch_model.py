from arch import arch_model

def garch_forecast(returns):
    model = arch_model(returns * 100, vol='Garch', p=1, q=1)
    fitted = model.fit(disp='off')

    forecast = fitted.forecast(horizon=1)
    variance = forecast.variance.iloc[-1, 0]

    return variance ** 0.5  # volatility