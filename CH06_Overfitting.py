# 과대적합(Overfitting)
"""
과대적합 : 신경망이 훈련 데이터에만 지나치게 적응되어 그 외의 데이터에는 제대로 대응하지 못하는 상태

※ 과대적합이 일어나는 경우
1. 매개변수가 많고 표현력이 높은 모델
2. 훈련 데이터가 적음
"""
# Ex. 과대적합의 예시 코드

x_train = x_train[:300]     # 데이터를 300개만 사용함
t_train = t_train[:300]



# 가중치 감소(Weight Decay)

'''
과대적합 억제용으로 많이 이용해온 방법 중 하나.

=> 학습 과정에서 큰 가중치에 대해서는 그에 상응하는 큰 페널티를 부과하여 과대적합을 억제하는 방식
=> 본래 과대적합은 가중치 매개변수의 값이 커서 발생하는 경우가 많기 때문.

가중치를 작게 만드는 방법(L2 법칙)

수식 : 새로운 Loss = 원래 Loss + 1/2yW**2

W**2 : 가중치 행렬의 모든 원소를 제곱해서 더한 값, 가중치가 클수록 이 값도 커짐

y(람다) : 페널티를 얼마나 강하게 줄지 결정하는 '강도', 람다가 크면 가중치를 아주 강력하게 억제함

1/2 : 이 식은 역전(미분)할 때, W**2가 루트W가 되므로 2를 약분해서 깔끔하게 남기기 위해 붙여둔 편의상 숫자

1. 손실값을 잴 때 0.5 * 람다 * W^2을 더해준다.
2. 기울기를 잴 때 람다 * W를 더해준다.
'''

import numpy as np

# 가중치 감소 코드의 예시

class Weight_Decay:
    def __init__(self, x_train, t_train, weights_lambda = 0.1):
        self.lastLayer = None
        self.x_train = x_train
        self.t_train = t_train
        self.params = {}
        self.weights_lambda = weights_lambda

    def loss(self, x, t):
        loss = self.lastLayer.forward(x, t)     # 손실 계산

        weights_decay = 0       # 가중치 감소 계산

        W1 = self.params['W1']
        W2 = self.params['W2']
        weights_decay += 0.5 * self.weights_lambda * np.sum(W1**2)      # 가중치에 벌점을 구해서 더함
        weights_decay += 0.5 * self.weights_lambda * np.sum(W2**2)      # 0.5 * 람다 * (W의 제곱의 합)

        return loss + weights_decay     # 최종손실 = 원래 손실 + 벌점




# 드롭아웃(Dropout)

"""
신경망 모델이 복잡해지면 가중치 감소만으로는 대응하기 어려움 -> 이때 사용하는 기법이 드롭아웃

드롭아웃은 뉴런을 임의로 삭제함녀서 학습하는 방법

훈련 때 은닉층의 뉴런을 무작위로 골라 삭제(신호차단) 해버리고, 시험 때는 모든 뉴런에 신호를 전달한다.

단, 시험 때는 각 뉴런의 출력에 훈련 때 삭제 안 한 비율을 곱하여 출력한다.(뉴런의 출력이 너무 강해지기 때문)
"""

class Dropout:
    def __init__(self, dropout_ratio = 0.5):
        self.dropout_ratio = dropout_ratio      # 뉴런을 삭제시킬 비율(50%)
        self.mask = None                        # 순전파 때 어떤 뉴런이 삭제됐었는지 적어둘 공간

    def forward(self, x, train_flg = True):     # 훈련할 때(True) / 시험할 때(False)
        if train_flg:
            # 1. 입력데이터(x)와 똑같은 모양으로 0.0 ~ 1.0 사이의 난수를 배열함
            # 2. 그 난수가 dropout_ratio(0.5)보다 큰 것만 True, 아니라면 False로 만듬
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask        # 입력 x에 mask를 곱함
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        # 역전파 때는 ReLU 계층의 역전파와 똑같은 원리임, 순전파 때 신호를 통과시킨 뉴련은 역전파에서도 그대로 통과시키고, 통과시키지 않은 뉴런은 역전파에서도 신호를 차단함
        return dout * self.mask
