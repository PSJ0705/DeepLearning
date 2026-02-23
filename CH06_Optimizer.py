import numpy as np

class SGD:      # 확률 경사 하강법
    def __init__(self, lr = 0.01):
        self.lr = lr        # lr = 학습률(learning rate), 학습률을 인스턴스 변수로 유지

    def update(self, params, grads):    # params와 grads 함수는 딕셔너리 변수
        for key in params.keys():
            params[key] -= self.lr * grads[key]

class Momentum:     # 운동량, 경사로에서 공 굴리기
    def __init__(self, lr = 0.01, momentum = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None       # v = 물체의 속도

    def update(self, params, grads):
        if self.v is None:
            self.v = {}         # update 가 처음 호출되면 딕셔너리 변수로 저장함
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]

class AdaGrad:      # 많이 걸은 발은 짧게, 안 걸은 발은 길게 (맞춤형 보폭)
    def __init__(self, lr = 0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)

class Adam:     # 모멘텀(관성) + AdaGrad(보폭 조절)의 퓨전
    def __init__(self, lr = 0.01):
        self.lr = lr




