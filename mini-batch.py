import sys, os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from mnist import dataset_dir, load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

print(x_train.shape, t_train.shape)
print(x_test.shape, t_test.shape)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

print(x_batch.shape, t_batch.shape)