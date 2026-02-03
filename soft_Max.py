import numpy as np

def softmax(x):
    c = np.max(x)
    e_x = np.exp(x - c)
    sum_exp_a = np.sum(e_x)
    y = e_x / sum_exp_a
    return y

a = np.array([0.3, 2.9, 4.0])

y = softmax(a)