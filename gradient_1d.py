"""
(h + x)와 (h - x)의 미분을 이용해 수치 미분을 구하는 공식
"""
def numerical_diff(f, x):            # 수치 미분
    h = 1e-4  #0.0001
    return (f(x + h) - f(x - h)) / (2*h)

def function_1(x):                  # 수치 미분을 이용하기 위한 간단한 함수 미분의 예시
    return 0.01*x**2 + 0.1*x

def function_2(x):
    return x[0]**2 + x[1]**2        # 또는 np.sum(x**2)

def numerical_gradient(f, x):       # x0, x1의 편미분을 벡터로 정리한 기울기
    h = 1e-4
    grad = np.zeros_like(x)         # x와 형상이 같은 배열을 생성

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h) 계산
        x[idx] = tmp_val + h
        fxh1 = f(x)

        #f(x-h)계산
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val            # 값 복원
    return grad

def gradient_descent(f, init_x, lr = 0.01, step_num = 1000):        #경사 하강법, f는 최적화하려는 함수, init_x는 초깃값, learning rate는 학습률, step num = 경사법에 따른 반복회수
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)                 #함수의 기울기를 구함
        x -= lr * grad
    return x


import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)

plt.xlabel("x")
plt.ylabel("f(x)")

plt.plot(x, y)
plt.show()

