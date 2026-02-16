import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np

from mnist import load_mnist
from CH05_TwoLayerNet_Back import TwoLayerNet


# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

# 각 가중치 차이의 절댓값을 구한 후, 그 절댓값의 평균을 낸다.
for key in grad_backprop.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(key + ":" + str(diff))

