import sys, os
from dataclasses import asdict

from tensorflow.python.ops.gen_sparse_ops import add_sparse_to_tensors_map_eager_fallback

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np

from CH05_Layers import *
from gradient_1d import numerical_gradient
from collections import OrderedDict

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):

        # 가중치(W)와 편향(b) 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # 계층 생성 및 조립
        self.layers = OrderedDict()     # 순서가 있는 딕셔너리(넣은 순서대로 저장됨)
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = ReLu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()      # 마지막 층은 따로 보관

    # 추론 (순전파) : 입력 데이터 x를 받아서 예측값을 내놓음
    def predict(self, x):
        for layer in self.layers.values():  # 등록된 계층들을 순서대로 통과시킴
            x = layer.forward(x)            # x가 Affine1 -> ReLU1 -> Affine2를 거치며 변함
             # SoftMax 층은 통과하지 않음
        return x

    # x : 입력 데이터, t : 정답 레이블
    # 손실 구하기(채점)
    def loss(self, x, t):
        y = self.predict(x)         # 1. 예측을 수행함
        return self.lastLayer.forward(y, t)     # 2. 마지막 층(SoftMax)에서 오차 계산

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
        if t.ndim != 1 : t = np.argmax(t, axis = 1)

        accuracy = np.sum(y == t) / y.shape[0]

        return accuracy

    # x : 입력 데이터, t : 정답 레이블
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x,t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    # 가중치 매개변수의 기울기 구하기(역전파 - 핵심)
    def gradient(self, x, t):
        # 1. 순전파
        self.loss(x,t)      # 먼저 앞으로 한번 쏨(그래야 각 층에 x,y값 등이 저장됨)

        # 2. 역전파
        dout = 1
        dout = self.lastLayer.backward(dout)

        # 3. 계층 순서 뒤집기
        layers = list(self.layers.values())
        layers.reverse()    # [Affine1, Relu1, Affine2] -> [Affine2, Relu1, Affine1]

        # 4. 뒤에서부터 차례대로 역전파 실행
        for layer in layers:
            dout = layer.backward(dout)     # 미분값(dout)이 뒤로 전달됨


        # 5. 결과 저장
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW # Affine1 층에 저장된 dW 가져오기
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW # Affine2 층에 저장된 dW 가져오기
        grads['b2'] = self.layers['Affine2'].db

        return grads



