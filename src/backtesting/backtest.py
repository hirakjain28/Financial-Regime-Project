import numpy as np

def backtest_strategy(df, forecasts, forecast_dates):

    profits = []
    positions = []

    cooldown = 0

    for i in range(len(forecasts) - 1):

        current_price = float(df.loc[forecast_dates[i], "Close"])
        next_price = float(df.loc[forecast_dates[i + 1], "Close"])
        predicted_price = float(forecasts[i])

        regime = int(df.loc[forecast_dates[i], "Regime"])

        pred_return = (predicted_price - current_price) / current_price

        # Dynamic threshold
        if regime == 0:
            threshold = 0.003
        else:
            threshold = 0.004

        cost = 0.0008 * current_price

        profit = 0

        # Reduce cooldown
        if cooldown > 0:
            cooldown -= 1

        # Only trade if cooldown finished
        if cooldown == 0:

            if pred_return > threshold:
                profit = (next_price - current_price) - cost
                positions.append("BUY")
                cooldown = 4

            elif pred_return < -threshold:
                profit = (current_price - next_price) - cost
                positions.append("SELL")
                cooldown = 4

            else:
                positions.append("HOLD")

        else:
            positions.append("HOLD")

        profits.append(profit)

    # Metrics
    total_profit = np.sum(profits)
    avg_profit = np.mean(profits)

    trades = [p for p in profits if p != 0]
    num_trades = len(trades)

    accuracy = sum(p > 0 for p in trades) / max(1, num_trades)
    win_rate = accuracy

    sharpe = np.mean(profits) / (np.std(profits) + 1e-6)

    cumulative = np.cumsum(profits)
    drawdown = np.max(np.maximum.accumulate(cumulative) - cumulative)

    print("Total Trades:", num_trades)
    print("Profit per trade:", total_profit / max(1, num_trades))
    print("Sharpe-like:", sharpe)
    print("Max Drawdown:", drawdown)

    return {
        "total_profit": total_profit,
        "avg_profit": avg_profit,
        "accuracy": accuracy,
        "win_rate": win_rate,
        "profits": profits,
        "positions": positions
    }