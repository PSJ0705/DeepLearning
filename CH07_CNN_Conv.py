# 합성곱 계층 구현하기

import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from CH07_im2col import *

class Convolution:
    def __init__(self, W, b, stride= 1, pad = 0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        self.x = None
        self.col = None
        self.col_W = None

    def forward(self, x):
        self.x = x
        # 1. 필터와 입력 데이터의 형태를 변수에 저장한다.
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape

        # 2. 출력 데이터의 가로(out_w), 세로(out_h) 크기를 계산한다.
        out_h = int((H + 2 * self.pad - FH)/self.stride) + 1
        out_w = int((W + 2 * self.pad - FW)/self.stride) + 1

        # 3. im2col을 이용해 4차원 입력 데이터(x)를 2차원 엑셀표로 편다.
        col = im2col(x, FH, FW, self.stride, self.pad)
        self.col = col

        # 4. 4차원 필터(W)도 2차원으로 편다
        col_W = self.W.reshape(FN, -1).T
        self.col_W = col_W
        """
        reshape(FN, -1) : 현재 필터는 (필터 개수, 채널, 세로, 가로) 모양이다. 이걸 reshape(FN, -1)로 묶어버리면,
        파이썬이 알아서 맨 앞은 필터 개수(FN)으로 놔두고, 나머지는 한즐로 쫙 펴서 2차원 표로 만들어준다.
        (-1은 나머지를 알아서 맞추라는 뜻)
        """

        # 5. 행렬곱셈 (2차원 이미지 표 * 2차원 필터 표) + 편향
        out = np.dot(col, col_W) + self.b

        # 6. 곱셈이 끝난 2차원 결과를 4차원 특징맵으로 복원
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        """
        transpose(0, 3, 1, 2) : 사용자가 원하는 4차원 상자의 순서는 (데이터 개수, 채널, 세로, 가로)이지만,
        np.dot의 결과물은 (N, out_h, out_w, FN)이라는 순서로 나온다.
        transpose(0, 3, 1, 2)는 0번 축은 그대로 두고, 3번 축을 두 번째로, 1번, 2번을 뒤로 밀으라는 뜻이다.
        """

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        # 1. 기울기(dout) 2차원으로 펴기
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        # 2. 편향의 기울기(db) 구하기
        self.db = np.sum(dout, axis=0)

        # 3. 필터 가중치의 기울기(dW) 구하기
        self.dW = np.dot(self.col.T, dout)

        # 3-1. 위에서 구한 2차원 형태의 가중치 기울기를 4차원 돋보기 모양으로 재구성하기(reshape)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        # 4. 아래층으로 흘려보낸 데이터의 기울기(dx) 구하기
        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx




