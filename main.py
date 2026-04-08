import sys
import os
sys.path.append(os.path.abspath("."))

import matplotlib.pyplot as plt
from src.data_loader import load_data
from src.preprocessing import add_features
from src.regime_model import train_hmm, predict_regime
from src.forecasting.adaptive_model import adaptive_forecast
from src.risk_model.garch_model import garch_forecast
from src.risk_model.var_model import calculate_var
from src.evaluation.metrics import rmse, mae

df = load_data()
df = add_features(df)

features = df[['Returns', 'Volatility']]

hmm_model = train_hmm(features)
df['Regime'] = predict_regime(hmm_model, features)

forecasts = adaptive_forecast(df)

print("Forecasts generated:", len(forecasts))

returns = df['Returns'].dropna()

volatility = garch_forecast(returns)
var = calculate_var(returns)

print("Forecasted Volatility:", volatility)
print("Value at Risk (95%):", var)

actual = df['Close'].iloc[-len(forecasts):].values
predicted = forecasts

print("RMSE:", rmse(actual, predicted))
print("MAE:", mae(actual, predicted))

plt.figure(figsize=(12,6))
plt.plot(df.index[-len(forecasts):], actual, label='Actual')
plt.plot(df.index[-len(forecasts):], predicted, label='Predicted')

plt.title("Actual vs Predicted Prices")
plt.legend()
plt.show()

plt.figure(figsize=(12,6))
plt.scatter(df.index, df['Close'], c=df['Regime'], cmap='viridis')
plt.title("Market Regimes")
plt.show()

