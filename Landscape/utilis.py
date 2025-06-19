from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np


def compute_metrics(x, x_hat, num_classes):
    N, T, L = x.shape
    C = 4 
    metrics = np.zeros((N, T, C))

    for i in range(N):
        for j in range(T):
            y_true = x[i, j]
            y_pred = x_hat[i, j]
            
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

            metrics[i, j] = [accuracy, precision, recall, f1]

    return metrics

def compute_metrics_bi(x, x_hat, num_classes):
    N, T, L  = x.shape
    C = 4 
    metrics = np.zeros((N, T, C))

    for i in range(N):
        for j in range(T):
            y_true = x[i, j]
            y_pred = x_hat[i, j]
            
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
            recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)

            metrics[i, j] = [accuracy, precision, recall, f1]

    return metrics