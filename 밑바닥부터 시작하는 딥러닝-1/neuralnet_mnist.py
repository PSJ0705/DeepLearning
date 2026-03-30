import pickle
import sys, os
from binascii import a2b_qp

from fontTools.ttLib.tables.T_S_I__0 import tsi0Format

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from mnist import load_mnist
from soft_Max import softmax
from PIL import Image

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_data():     # mnist 테스트 이미지 준비
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)

    return x_test, t_test


def init_network():     # 미리 학습된 가중치 파일을 장착
    with open(os.path.dirname(__file__) + "/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network

def predict(network, x):        # 입력 데이터를넣고 계산해서 확률을 예측
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

x, t = get_data()
network = init_network()

batch_size = 100    #배치 크기
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i + batch_size]
    t_batch = predict(network, x_batch)
    p = np.argmax(t_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i + batch_size])

print("Accuracy : " + str(float(accuracy_cnt) / len(x)))