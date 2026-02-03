import numpy as np

def sum_squares_error(y, t):        # 오차제곱합
    return 0.5 * np.sum((y - t)**2)

def cross_entropy_error(y, t):      # 교차 엔트로피
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

def cross_entropy_loss_func(y, t):      #미니배치를 지원하는 교차 엔트로피
    if y.ndim ==1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

def sigmoid(x):
    return 1 / (1 + np.exp(-x))