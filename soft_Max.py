import numpy as np

def softmax(x):
    e_x = np.exp(x)
    sum_exp = np.sum(np.exp(x))
    y = e_x / sum_exp
    return y

