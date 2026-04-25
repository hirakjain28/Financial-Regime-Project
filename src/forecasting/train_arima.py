import pandas as pd
import joblib
from statsmodels.tsa.arima.model import ARIMA

# Load dataset
df = pd.read_csv("data/processed/nifty_clean.csv")

# Use latest relevant rows (recommended)
data = df["Close"].dropna().tail(3000)

# Train model
model = ARIMA(data, order=(5,1,0))
fitted_model = model.fit()

# Save model
joblib.dump(fitted_model, "src/forecasting/arima_model.pkl")

print("ARIMA model trained and saved successfully.")