"""
2층 신경망(은닉층 1개)
"""

import sys, os

import numpy as np

sys.path.append(os.path.join(sys.path[0], '..'))

from loss_function import *
from gradient_1d import numerical_gradient
from soft_Max import *

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):         # 가중치 초기화를 수행, 인수는 순서대로 입력층의 뉴런수, 은닉층의 뉴런수, 출력층의 뉴런수
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)      # 가중치를 난수로 초기화
        self.params['b1'] = np.zeros(hidden_size)                                           # 편향을 0으로 초기화
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)     # 가중치를 난수로 초기화
        self.params['b2'] = np.zeros(output_size)                                           # 편향을 0으로 초기화

    def predict(self, x):           # 예측(추론)을 수행한다, 인수 x는 이미지 데이터
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)            # 은닉층 활성화 함수(시그모이드 : 입력값을 0과 1사이의 값으로 변환, S자 곡선)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)             # 출력층 활성화 함수(소프트맥스) : 여러 개의 입력값을 받아 각각을 0~1 사이의 값으로 정규화, 전체 출력의 합이 1이 됨)
                                    # 소프트 맥수 함수를 통해 확률로 변환됨
        return y

    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):           # 손실 함수의 값을 구한다, 인수 x는 이미지 데이터, t는 정답 레이블
        y = self.predict(x)

        return cross_entropy_loss_func(y, t)        # 교차 엔트로피의 오차(손실값)를 구함

    def accuracy(self, x, t):       # 정확도를 계산한다
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x : 입력 데이터, t : 정답 레이블
    def numerical_gradient(self, x, t):         # 기울기를 계산한다.
        loss_W = lambda W: self.loss(x, t)      # 손실함수를 W에 대한 함수로 정의

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads


"""
net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
print(net.params['W1'].shape)           # TwoLayerNet 클래스의 매개변수 params 변수에는 이 신경망에 필요한 매개변수가 모두 저장된다.
print(net.params['b1'].shape)
print(net.params['W2'].shape)
print(net.params['b2'].shape)
print("=======")

x = np.random.rand(100,784)      # 더미 데이터 입력(100장 분량)
                                 # grads 변수에는  params 변수에 대응하는 각 매개변수의 기울기가 저장된다.

t = np.random.rand(100,10)          # 더미 정답 레이블(100장 분량)

grads = net.numerical_gradient(x, t)        # 기울기 계산


print(grads['W1'].shape)
print(grads['b1'].shape)
print(grads['W2'].shape)
print(grads['b2'].shape)
"""