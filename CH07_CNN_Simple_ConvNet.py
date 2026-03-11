"""
:parameter

input_dim : 입력 데이터(채널 수, 높이, 너비)의 차원

conv_param : 합성곱 계층의 하이퍼파라미터(딕셔너리), 딕셔너리의 키는 다음과 같다.
- filter_num : 필터 수
- filter_size : 필터 크기
- stride : 스트라이드
- pad : 패딩
- hidden_size : 은닉층(완전연결)의 뉴런 수
- output_size : 출력층(완전연결)의 뉴런 수
- weight_init_std : 초기화 때의 가중치 표준편차
"""

import numpy as np
from collections import OrderedDict
from CH07_CNN_Conv import *
from CH05_Layers import *
from CH07_CNN_Pooling import *

class SimpleConvNet:
    def __init__(self, input_dim = (1, 28, 28), conv_param = {'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1}, hidden_size = 100, output_size = 10, weight_init_std = 0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]

        # 출력 크기 계산 (설계도 그리기)
        # 합성곱 계층의 출력 크기 계산 공식 (OH = H + 2P -FH / S) + 1 )
        conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1

        # 풀링 계층을 통과한 후의 데이터 크기 (2 * 2 풀링을 사용하므로 가로세로가 반토막(/2)나게 되고, 이것을 1차원으로 쫙 폈을 때의 전체 뉴런 개수를 미리 계산해둠)
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))


        # 가중치 저장고 만들기
        # 신경망이 학습하면서 계속 업데이트 해야 할 가중치와 평향을 params라는 딕셔너리에 보관
        # W1/b1은 합성곱층, W2/b2와 W3/b3는 완전연결(Affine) 층의 매개변수
        self.params = { }
        self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)


        # 컨베이터 벨트 조립
        self.layers = OrderedDict()     # OrderDict() : 순서가 기억되는 사전
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = ReLu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride = 2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = ReLu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])
        self.last_layer = SoftmaxWithLoss()


    # predict & loss (순전파, 작동시키기)
    def predict(self, x):
    # 추론을 수행함, OrderDict() 있는 부품(layer)들을 하나씩 꺼내서, 데이터 x를 forward 함수에 넣고 계속 다음층으로 넘긴다. 끝까지 통과하고 냐온 날것의 점수들을 반환함
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    # loss(오차계산) : predict로 구한 예측 점수(y)와 실제 정답(t)를 최종 검사 역할인 last_layer에게 넘겨주어, 오차 점수를 계산해냄
    def loss(self, x, t):       # x : 입력 데이터, t : 정답 레이블
        y = self.predict(x)
        return self.last_layer.loss(y, t)

    # gradient(역전파, 스스로 학습)
    def gradient(self, x, t):
        # 순전파, 일단 끝까지 돌려봄
        self.loss(x, t)

        # 역전파, 맨 마지막 층(softmax)부터 거꾸로 역전파를 시작
        dout = 1
        dout = self.last_layer.backward(dout)

        # 컨베이어 벨트의 순서를 뒤집는다(역주행)
        layers = list(self.layers.values())
        layers.reverse()

        # 뒤집힌 순서대로 부품을 꺼내며 backward를 실행함
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장, 각 부품이 찾아낸 '가중치 수정 방향(기울기)를 수거해서 반환함
        grads = {}
        grads['W1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conv1'].db
        grads['W2'] = self.layers['Affine1'].dW
        grads['b2'] = self.layers['Affine1'].db
        grads['W3'] = self.layers['Affine2'].dW
        grads['b3'] = self.layers['Affine2'].db

        return grads






