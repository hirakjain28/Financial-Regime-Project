import numpy as np

def add_features(df):
    df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    return df.dropna()