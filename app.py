import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


from src.preprocessing import add_features
from src.regime_model import train_hmm, predict_regime
from src.forecasting.adaptive_model import adaptive_forecast
from src.backtesting.backtest import backtest_strategy


df = pd.read_csv("data/processed/nifty_clean.csv", index_col='Date', parse_dates=True)

# Add features
df = add_features(df)

# Train HMM
features = df[['Returns', 'Volatility']]
model = train_hmm(features)

# Predict regime
df['Regime'] = predict_regime(model, features)


st.title("Financial Market Regime Detection System")

st.subheader("Closing Price")
st.line_chart(df['Close'])

st.subheader("Returns")
st.line_chart(df['Returns'])

st.subheader("Volatility")
st.line_chart(df['Volatility'])

st.subheader("Regime Visualization")

fig, ax = plt.subplots()
ax.scatter(df.index, df['Close'], c=df['Regime'], cmap='viridis')
st.pyplot(fig)


st.subheader("Model Predictions")

# Run model
forecasts, forecast_dates = adaptive_forecast(df)

actual = df.loc[forecast_dates]['Close']

print("Forecasts:", forecasts)
print("Actual:", actual.values)
print("Len forecasts:", len(forecasts))
print("Len actual:", len(actual))

# Plot
fig, ax = plt.subplots(figsize=(12,6))

# actual
ax.plot(forecast_dates,
        actual,
        color="blue",
        linewidth=2.5,
        label="Actual")

# predicted
ax.plot(forecast_dates[:len(forecasts)],
        forecasts,
        color="orange",
        linestyle="--",
        linewidth=2.5,
        marker="o",
        markersize=4,
        label="Predicted")

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

plt.xticks(rotation=45)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_title("Actual vs Predicted Prices", fontsize=18)
plt.tight_layout()

st.pyplot(fig)

fig, ax = plt.subplots(figsize=(12,6))

ax.plot(forecast_dates, actual,
        color="blue",
        linewidth=2.5,
        label="Actual")

ax.plot(forecast_dates[:len(forecasts)],
        forecasts,
        color="orange",
        linestyle="--",
        linewidth=2,
        alpha=0.9)

ax.scatter(forecast_dates[:len(forecasts)],
           forecasts,
           color="red",
           s=45,
           zorder=5,
           label="Predicted")

plt.xticks(rotation=45)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()

st.pyplot(fig)

st.subheader("Backtesting Results")

results = backtest_strategy(df, forecasts, forecast_dates)

st.write("Total Profit:", results["total_profit"])
st.write("Accuracy:", results["accuracy"])
st.write("Win Rate:", results["win_rate"])

# Plot profit curve

cumulative_profit = np.cumsum(results["profits"])

fig, ax = plt.subplots()
ax.plot(cumulative_profit)
ax.set_title("Cumulative Profit")

st.pyplot(fig)