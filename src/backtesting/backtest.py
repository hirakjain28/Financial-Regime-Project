import numpy as np

def backtest_strategy(df, forecasts, forecast_dates):
    profits = []
    positions = []

    for i in range(len(forecasts) - 1):

        current_price = df.loc[forecast_dates[i]]['Close']
        next_price = df.loc[forecast_dates[i+1]]['Close']
        predicted_price = forecasts[i]
        regime = df.loc[forecast_dates[i]]['Regime']
        recent_return = np.log(current_price / df.iloc[i-1]['Close'])

        threshold = 0.0015
        cost = 0.001 * current_price
        stop_loss = 0.006 * current_price   # 1%

        pred_return = np.log(predicted_price / current_price)
        confidence = abs(pred_return)
        
        if abs(pred_return) < 0.0008:
            profits.append(0)
            positions.append("HOLD")
            continue

        if regime == 0:   # stable
            size = min(max(confidence, 0.7), 1.5)

        else:   # volatile
            size = min(max(confidence, 0.3), 1.0)
            
        position_size = size
        
        if regime == 1:
            threshold = 0.003

        pred_return = np.log(predicted_price / current_price)

        if pred_return > threshold:
            raw_profit = next_price - current_price
            raw_profit = max(raw_profit, -stop_loss)
            profit = position_size * (raw_profit - cost)
            positions.append("BUY")

        elif pred_return < -threshold:
            raw_profit = current_price - next_price
            raw_profit = max(raw_profit, -stop_loss)
            profit = position_size * (raw_profit - cost)
            positions.append("SELL")

        else:
            profit = 0
            positions.append("HOLD")

        profits.append(profit)

    total_profit = np.sum(profits)
    avg_profit = np.mean(profits)
    
    trades = [p for p in profits if p != 0]
    accuracy = sum(p > 0 for p in trades) / max(1, len(trades))
    win_rate = sum(p > 0 for p in profits) / len(profits) 
    num_trades = sum(1 for p in positions if p != "HOLD")
    print("Total Trades:", num_trades)
    print("Profit per trade:", total_profit / max(1, num_trades))
    sharpe = np.mean(profits) / (np.std(profits) + 1e-6)
    print("Sharpe-like:", sharpe)
    
    cumulative = np.cumsum(profits)
    drawdown = np.max(np.maximum.accumulate(cumulative) - cumulative)

    print("Max Drawdown:", drawdown)

    if positions and positions[-1] == "BUY" and pred_return > threshold:
        profit = 0   # avoid repeated buy
        
    return {
        "total_profit": total_profit,
        "avg_profit": avg_profit,
        "accuracy": accuracy,
        "win_rate": win_rate,
        "profits": profits,
        "positions": positions
    }  