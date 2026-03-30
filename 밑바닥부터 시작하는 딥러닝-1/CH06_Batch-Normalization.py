# 배치 정규화
'''
<배치 정규화가 주목받는 이유>
 1.학습을 빨리 진행할 수 있다
 2.초깃값에 크게 의존하지 않는다.
 3.과대적합을 억제한다.

 => m 개의 입력 데이터에 대해
 1. 미니배치의 평균과 분산을 구하고,
 2. 정규화(표준화)를 수행하여 강제로 종 모양의 데이터로 깍아버리고,
 3. 정규화된 데이터에 대해 확대(y)와 이동(B)를 수행함, *y와 B는 학습하면서 적합한 값으로 조정
'''

import numpy as np

class BatchNormalization:
    def __init__(self, gamma, beta, momentum = 0.9):
        self.gamma = gamma      # 확대(Scale) 매개변수
        self.beta = beta        # 이동(Shift) 매개변수
        self.momentum = momentum        # 이동 평균을 구할 때 쓸 관성값

        self.running_mean = None
        self.running_var = None

    def forward(self, x, train_flg = True):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.ones(D)

        if train_flg:
            mu = x.mean(axis = 0)       # 평균 구하기
            xc = x - mu                 # x - 평균
            var = np.mean(xc ** 2, axis = 0)       # 미니배치 분산
            std = np.sqrt(var + 1e7)        # 표준편차 + (입실론)
            xn = xc / std               # 정규화(표준화)

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * xn
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * std
        else:
            xc = x - self.running_mean
            xn = xc / (self.running_var + 1e-7)

        out = self.gamma * xn + self.beta       # 확대와 이동

        return out

    def backward(self, dout):   # 역전파
        N, D = dout.shape       # N : 데이터 개수(배치 크기), D : 데이터 차원

        # --- 역전파 4단계 ---
        dbeta = dout.sum(axis = 0)
        dgamma = np.sum(self.xn * dout, axis = 0)

        # --- 역전파 3단계 ---
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis = 0)

        # --- 역전파 2단계---
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / N) * self.xc * dvar

        # --- 역전파 1단계 ---
        dmu = np.sum(dxc, axis = 0)
        dx = dxc - dmu / N

        # 가중치 업데이트 저장
        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx