import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from mnist import load_mnist
from CH05_TwoLayerNet_Back import TwoLayerNet

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
network = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)

iters_num = 10000       # 반복횟수
train_size = x_train.shape[0]       # 60,000개
batch_size = 100        # 한번에 100개씩 묶어서 학습
learning_rate = 0.01       # 학습률

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 오차역전법으로 기울기를 구함
    grad = network.gradient(x_batch, t_batch)

    # 갱신
    for key in grad.keys():
        network.params[key] -= learning_rate * grad[key]        # 원래 가중치에서 (학습률 * 기울기)만큼 뺌

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)      # train_acc :  훈련 데이터 점수, test_acc : 시험 데이터 점수