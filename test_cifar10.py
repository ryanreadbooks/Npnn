import time
import numpy as np
import npnn
from npnn.utils import get_mini_batch3d
from datasets import load_cifar10
import matplotlib.pyplot as plt
from npnn.load import load_params


# create network
net = npnn.nn.Sequential(
		npnn.basic.Conv2d(3, 32, kernel_size=3, stride=1, padding_mode='same'),
		npnn.basic.ReLU(),
		npnn.basic.Conv2d(32, 32, kernel_size=3, stride=1, padding_mode='same'),
		npnn.basic.ReLU(),		
		npnn.basic.AvgPool2d(2, 2),
		npnn.basic.Conv2d(32, 48, kernel_size=3, stride=1, padding_mode='same'),
		npnn.basic.ReLU(),		
		npnn.basic.AvgPool2d(2, 2),
		npnn.basic.Conv2d(48, 64, kernel_size=3, stride=1, padding_mode='same'),
		npnn.basic.ReLU(),
		npnn.basic.AvgPool2d(2, 2),
		npnn.basic.Flatten(),
		npnn.basic.Linear(1024, 512),
		npnn.basic.ReLU(),
		npnn.basic.Dropout(0.5),
		npnn.basic.Linear(512, 64),
		npnn.basic.ReLU(),
		npnn.basic.Linear(64, 10),
		npnn.basic.Softmax()		
	)

# define an optimizer
optim1 = npnn.optim.RMSprop(net, lr=0.001)

x_ax = []
train_loss = []
tracker = 0

# start training
for epoch in range(25):
	# iterate all examples
	for batch_num in range(1, 6):
		cifar = load_cifar10(train=True, n_batch=batch_num)
		x_train = cifar.data
		x_train /= 255.0
		y_train = cifar.target
		n_batch = 156
		x_mini, y_mini = get_mini_batch3d(x_train, y_train, n_batch, shuffle=True)

		for iteration in range(n_batch):
			net.forward(x_mini[iteration])
			loss = net.loss(y_mini[iteration])
			print(f'Current loss: {loss} on {iteration} times')
			net.backward()
			optim1.step()

			tracker += 1
			train_loss.append(loss)
			x_ax.append(tracker)
	print(f'****************************************************epoch {epoch} finished.\n')


current_time = str(time.time())[0:10]
save_path = f'test_cifar{current_time}'
npnn.save_params(net, save_path)

plt.figure()
plt.plot(x_ax, train_loss, label='training')
plt.grid()
plt.legend()
plt.show()