import time
import numpy as np 
import npnn
import npnn.metrics as metric
from npnn.utils import get_mini_batch3d
from npnn.load import load_params
from datasets import load_cifar10


cifar = load_cifar10(train=True)

x_test = cifar.data
x_test /= 255.0
y_test = cifar.target

n_batch = 100
x_mini, y_mini = get_mini_batch3d(x_test, y_test, n_batch, shuffle=True)


net = load_params('trained_models\\cifar10_example_params')

total_acc = 0
for i in range(n_batch):
	net1_prediction = net.predict(x_mini[i])
	acc1 = metric.accuracy_score(y_mini[i], net1_prediction)
	print(net1_prediction)
	print(y_mini[i])
	print(f'Accuracy: {acc1}')
	total_acc += acc1
print(f'Average acc: {total_acc / n_batch}')