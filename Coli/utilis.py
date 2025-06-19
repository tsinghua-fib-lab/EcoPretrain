from sklearn.metrics import r2_score, mean_absolute_percentage_error
import numpy as np


def compute_metrics(x, x_hat):
    N, T, L = x.shape
    C = 3 
    metrics = np.zeros((N, T, C))

    for i in range(N):
        for j in range(T):
            y_true = x[i, j]
            y_pred = x_hat[i, j]
            
            r2 = r2_score(y_true, y_pred)
            nmse = np.mean((y_true - y_pred) ** 2) / np.mean(y_true ** 2)
            mape = mean_absolute_percentage_error(y_true, y_pred)

            metrics[i, j] = [r2, nmse, mape]

    return metrics