#fonction de perte(fonction de perte quadratique )
import math

def mse(y_true, y_pred):
    n = len(y_true)
    squared_errors = [(y_pred[i] - y_true[i])**2 for i in range(n)]
    return sum(squared_errors) / n

def mse_prime(y_true, y_pred):
    n = len(y_true)
    return [(2/n) * (y_pred[i] - y_true[i]) for i in range(n)]
"""

# Fonction de perte basée sur le logarithme
def mse(y_true, y_pred):
    loss = 0
    for i in range(len(y_true)):
        loss += -(y_true[i] * math.log(y_pred[i]) + (1 - y_true[i]) * math.log(1 - y_pred[i]))
    return loss / len(y_true)

# Dérivée de la fonction de perte basée sur le logarithme
def mse_prime(y_true, y_pred):
    grad = []
    for i in range(len(y_true)):
        grad.append((y_pred[i] - y_true[i]) / (y_pred[i] * (1 - y_pred[i])))
    return grad
"""