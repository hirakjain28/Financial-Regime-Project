import pandas as pd

def load_data():
    df = pd.read_csv("data/processed/nifty_clean.csv", index_col='Date', parse_dates=True)

    df = df.asfreq('B')  # Business days
    df = df.fillna(method='ffill')

    return df