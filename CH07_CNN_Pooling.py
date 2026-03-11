# 풀링 구현
from CH07_im2col import *
import numpy as np

class Pooling:
    def __init__(self, pool_h, pool_w, stride = 1, pad = 0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

    def forward(self, x):

        # 0. 출력 크기를 계산
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        # 1. 입력 데이터를 전개(채널별로 한 줄로 펴기)
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        # im2col로 쫙 편 데이터를 다시 한번 reshape로 조작해줌 -> 풀링 창 크기만큼 한 줄에 들어가도록 조작
        col = col.reshape(-1, self.pool_h * self.pool_w)

        # 2. 행별 최댓값을 구한다 (axis = 1 (가로방향))
        out = np.max(col, axis = 1)

        # 3. 적절한 모양으로 성형한다 (원래 모양인 4차원 상자로 접는다)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        return out

def backward(self, dout):

    dout = dout.transpose(0, 2, 3, 1)       # 축을 순전파로 바꾼 뒤, 1차원으로 납작하게 만든다

    pool_size = self.pool_h * self.pool_w       # 풀링 구역 크기

    dmax = np.zeros((dout.size, pool_size))
    dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()

    dmax = dmax.reshape(dout.shape + (pool_size,))
    dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)

    dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

    return dx


