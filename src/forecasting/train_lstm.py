import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import joblib

def train_lstm(series):
    
    returns = np.log(series / series.shift(1)).dropna()

    data = returns.values.reshape(-1, 1)

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    window = 20

    for i in range(window, len(data_scaled)):
        X.append(data_scaled[i-window:i])
        y.append(data_scaled[i])

    X, y = np.array(X), np.array(y)

    model = Sequential()
    model.add(LSTM(64, input_shape=(X.shape[1], 1)))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, batch_size=16, verbose=1)

    model.save("lstm_model.h5")
    joblib.dump(scaler, "scaler.save")
    
    
if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv(r"F:\Study\Projects\Financial-Regime-Project\data\processed\nifty_clean.csv")  # adjust path
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    series = df['Close']

    train_lstm(series)