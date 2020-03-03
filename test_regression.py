import numpy as np 
import npnn
from npnn.utils import get_mini_batch2d
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 


# We use data from Sklearn boston housing problem
boston = load_boston()
raw_x = boston.data
raw_y = boston.target
seed = np.random.randint(np.random.randint(1,1e3))
x_train, x_test, y_train, y_test = train_test_split(raw_x, raw_y, test_size = 0.05, random_state = seed)

# Because in our model, the shape of input x is (numbers neurons, numbers of sample in one batch)
# We take transpose of sklearn data
x_train = x_train.T
x_test = x_test.T

# We can set batch
n_batch = 1
x_mini, y_mini = get_mini_batch2d(x_train, y_train, n_batch, shuffle = False)

net1 = npnn.nn.Sequential(
		npnn.basic.BatchNorm(n_in = 13, affine = False),
		npnn.basic.Linear(13,6),
		npnn.basic.Sigmoid(),
		npnn.basic.Linear(6,1),
		npnn.basic.MSELoss()
	)

# RMSprop optimizer
lr = 0.003
sgd_optimizer = npnn.optim.RMSprop(net1, lr = 0.001)

x_epoch = []
train_loss1 = []

for epoch in range(10000):
	for iteration in range(n_batch):
		net1.forward(x_mini[iteration])
		loss1 = net1.loss(y_mini[iteration])
		net1.backward()
		sgd_optimizer.step()
	
	if epoch % 50 == 0:
		x_epoch.append(epoch)
		train_loss1.append(loss1)

# Make prediction
net1_prediction = net1.predict(x_test)
print('Prediction SGD        : ',net1_prediction)
print('True target           : ',y_test)

# Plot loss iteration
plt.figure()
plt.plot(x_epoch, train_loss1, label = 'SGD')
plt.legend()
plt.grid()
plt.show()