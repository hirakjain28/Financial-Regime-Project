import numpy as np

def calculate_var(returns, confidence_level=0.95):
    mean = np.mean(returns)
    std = np.std(returns)

    var = mean - std * 1.65  # 95% approx

    return var