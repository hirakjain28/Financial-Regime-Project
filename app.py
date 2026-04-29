import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.preprocessing import add_features
from src.regime_model import train_hmm, predict_regime
from src.forecasting.adaptive_model import adaptive_forecast
from src.backtesting.backtest import backtest_strategy

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AI Financial Regime Detection System",
    page_icon="📈",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.metric-box {
    background: #111827;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
}
h1,h2,h3 {
    color: white;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(
        "data/processed/nifty_clean.csv",
        index_col="Date",
        parse_dates=True
    )
    return df

df = load_data()

# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------
df = add_features(df)

features = df[['Returns', 'Volatility']]
model = train_hmm(features)
df['Regime'] = predict_regime(model, features)

# --------------------------------------------------
# FORECAST
# --------------------------------------------------
forecasts, forecast_dates = adaptive_forecast(df)
actual = df.loc[forecast_dates]["Close"]

# --------------------------------------------------
# BACKTEST
# --------------------------------------------------
results = backtest_strategy(df, forecasts, forecast_dates)

cumulative_profit = np.cumsum(results["profits"])

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("📈 AI Financial Market Regime Detection Dashboard")
st.caption("Hybrid Forecasting + HMM Regime Detection + Smart Backtesting Engine")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("⚙ Control Panel")
view = st.sidebar.selectbox(
    "Select View",
    ["Dashboard", "Forecasting", "Backtesting", "Analytics"]
)

# --------------------------------------------------
# KPI SECTION
# --------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("NIFTY Close", f"{df['Close'].iloc[-1]:,.2f}")

with col2:
    regime_text = "Stable" if df["Regime"].iloc[-1] == 0 else "Volatile"
    st.metric("Current Regime", regime_text)

with col3:
    st.metric("Total Profit", f"{results['total_profit']:.2f}")

with col4:
    st.metric("Accuracy", f"{results['accuracy']*100:.2f}%")

# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------
if view == "Dashboard":

    st.subheader("📊 Market Price Overview")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["Close"],
        mode="lines",
        name="Close Price"
    ))

    fig.update_layout(
        template="plotly_dark",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🧠 Regime Visualization")

    fig = px.scatter(
        df,
        x=df.index,
        y="Close",
        color="Regime",
        template="plotly_dark",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# FORECASTING
# --------------------------------------------------
elif view == "Forecasting":

    st.subheader("🔮 Actual vs Predicted")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=actual,
        mode="lines",
        name="Actual",
        line=dict(width=3)
    ))

    fig.add_trace(go.Scatter(
        x=forecast_dates[:len(forecasts)],
        y=forecasts,
        mode="lines+markers",
        name="Predicted",
        line=dict(dash="dash")
    ))

    fig.update_layout(
        template="plotly_dark",
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# BACKTESTING
# --------------------------------------------------
elif view == "Backtesting":

    st.subheader("💰 Cumulative Profit Curve")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=cumulative_profit,
        mode="lines",
        name="Equity Curve"
    ))

    fig.update_layout(
        template="plotly_dark",
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Win Rate", f"{results['win_rate']*100:.2f}%")

    with col2:
        st.metric("Trades", len([p for p in results["profits"] if p != 0]))

    with col3:
        st.metric("Avg Profit", f"{results['avg_profit']:.2f}")

# --------------------------------------------------
# ANALYTICS
# --------------------------------------------------
elif view == "Analytics":

    st.subheader("📉 Returns Distribution")

    fig = px.histogram(
        df,
        x="Returns",
        nbins=50,
        template="plotly_dark",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("⚡ Volatility Trend")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["Volatility"],
        mode="lines",
        name="Volatility"
    ))

    fig.update_layout(
        template="plotly_dark",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# FOOTER AI INSIGHT
# --------------------------------------------------
st.markdown("---")

latest_return = df["Returns"].iloc[-1]

if latest_return > 0:
    signal = "BUY"
else:
    signal = "SELL"

st.subheader("🤖 AI Insight")

st.success(
    f"Current regime appears {regime_text}. "
    f"Model suggests **{signal}** bias based on latest momentum and regime structure."
)