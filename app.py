import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.preprocessing import add_features
from src.regime_model import train_hmm, predict_regime

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