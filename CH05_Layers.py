# ReLu 함수의 순전파와 역전파 (활성화 함수) 구현
# Sigmoid 함수의 순전파와 역전파 구현

import numpy as np
from tensorflow.python.framework.test_ops import none_eager_fallback
from soft_Max import softmax

from loss_function import cross_entropy_error


class ReLu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)        # x의 원소값이 0 이하인 인덱스는 True, 그 외에는 False
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))      # Sigmoid 함수
        self.out = out                  # 순전파의 출력을 인스턴스 변수 out에 보관했다가 역전파 계산에 그 값을 사용함

        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out)  * self.out

        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.y = None
        self.db = None

    def forward(self, x):
        self.x = x                          # 역전파에서 쓰기 위해 입력값을 기억해 둠
        out = np.dot(x, self.W) + self.b    # 핵심 연산 : 행렬 곱 + 편향 덧셈

        return out

    def backward(self, dout):
        dx = np.dot(dout,  self.W.T)        # .T : 전치행렬 (2,3) => (3,2)로 위치를 바꿈
        self.dW = np.dot(self.x.T, dout)    # 가중치 W를 얼마나 수정해야 할지 구해야 함
        self.db = np.sum(dout, axis=0)      # 편향 업데이트용

        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None        # 손실값
        self.y = None           # softMax의 출력값
        self.t = None           # 정답 레이블 (원-핫 인코딩)

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)         # 1. 확률로 변환
        self.loss = cross_entropy_error(self.y, self.t)     # 2. 오차 계산

        return self.loss

    def backward(self, dout = 1):
        batch_size = self.t.shape[0]        # 배치 크기
        dx = (self.y - self.t) / batch_size     # (예측값 - 정답) / 배치 크기 => 배치의 오차를 다 합친 값이므로, 데이터 1개당 평균 오차로 나눠서 전달해야 함

        return dx

x = np.array([[1.0, -0.5], [-2.0, 3.0]])

print(x)

mask = (x <= 0)
print(mask)