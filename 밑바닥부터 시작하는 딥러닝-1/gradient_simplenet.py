import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))
import numpy as np

from loss_function import cross_entropy_error
from soft_Max import softmax
from gradient_1d import numerical_gradient, gradient_descent

class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)       # 정규분포를 랜덤한 숫자로 초기화, 입력값이 2개, 출력값이 3개라는 의미

    def predict(self, x):                   # 예측을 수행함
        return np.dot(x, self.W)            # 입력값과 가중치(W)를 행렬곱하여 결과를 내놓음

    def loss(self, x, t):                   # 손실함수의 값을 구함, x : 입력 데이터, t : 정답 레이블
        z = self.predict(x)                 # 예측 점수 계산
        y = softmax(z)                      # 확률로 변환
        loss = cross_entropy_error(y, t)    # 정답과 비교하여 오차 계산, loss값이 작을수록 똑똑한 신경망임

        return loss


net = SimpleNet()
print(net.W)                                # 가중치 매개변수

x = np.array([0.6, 0.9])                    # 입력 데이터
p = net.predict(x)                          # 예측 실행
print(p)

print(np.argmax(p))                         # 최댓값의 인덱스

t = np.array([0,0,1])
print(net.loss(x, t))                       # 정답레이블과 비교한 loss값 출력

f = lambda w: net.loss(x, t)
dW = gradient_descent(f, net.W)
print(dW)