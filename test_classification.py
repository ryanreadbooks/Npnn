import numpy as np 
import npnn
import npnn.metrics as metric
from npnn.utils import get_mini_batch2d
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 


# We use data from Sklearn iris data set
iris = load_iris()
raw_x = iris.data
raw_y = iris.target
seed = np.random.randint(np.random.randint(1,1e3))
x_train, x_test, y_train, y_test = train_test_split(raw_x, raw_y, test_size=0.2, random_state=seed)

# Because in our model, the shape of input x is (numbers neurons, numbers of sample in one batch)
# We take transpose of sklearn data
x_train = x_train.T
x_test = x_test.T

# we could set batch size if we want
n_batch = 1
x_mini, y_mini = get_mini_batch2d(x_train, y_train, n_batch, shuffle = False)

net1 = npnn.nn.Sequential(
		npnn.basic.BatchNorm(n_in = 4, affine = False),
		npnn.basic.Linear(4,6),
		npnn.basic.ReLU(),
		npnn.basic.Linear(6,3),
		npnn.basic.Softmax()
	)

optimizer1 = npnn.optim.RMSprop(net1, lr=0.03)

x_epoch = []
train_loss1 = []

for epoch in range(1000):
	for iteration in range(n_batch):
		net1.forward(x_mini[iteration])
		loss1 = net1.loss(y_mini[iteration])
		net1.backward()
		optimizer1.step()
	
	if epoch % 50 == 0:
		x_epoch.append(epoch)
		train_loss1.append(loss1)

net1_prediction = net1.predict(x_test)

print('Prediction SGD        : ',net1_prediction)
print('True target           : ',y_test)

plt.figure()
plt.plot(x_epoch, train_loss1, label = 'loss')
plt.legend()
plt.grid()
plt.show()