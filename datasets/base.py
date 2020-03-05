from os.path import dirname
import csv
import numpy as np
import pickle


class DataCollector:
	r"""
	Data Collector

	Attributes:
		data: data itself(numpy array)
		feature_name: 
		target: 
		target_name: 
	"""

	def __init__(self, **kwargs):
		self.data = None
		self.feature_name = None
		self.target = None
		self.target_name = None

	def __setitem__(self, key, value):
		self.__dict__[key] = value

	def __getitem__(self, key):
		try:
			return self.__dict__[key]
		except KeyError:
			raise KeyError(f'attribute {key} does not exist')


def load_mnist(return_tensor=True, train=True):
	r"""
	Load mnist dataset
	
	Args:
		return_tensor: return tensor or not	
		train: need training set or testing set
	Return:
		data dict
	"""
	root = dirname(__file__)
	if train:
		filepath = root + '\\data\\mnist\\train.csv'
		m = 9900
	else:
		filepath = root + '\\data\\mnist\\test2.csv'
		m = 1600

	data = np.zeros((784, m))
	target = []

	target_name = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
	with open(filepath, 'r') as csv_file:
		reader = csv.reader(csv_file)
		
		for index, row in enumerate(reader):
			data[:, index] = np.array(row[1:]).reshape((-1,))
			target.append(int(row[0])) 

	if return_tensor:
		data = np.transpose(data).reshape((m, 1, 28, 28))

	bunch = DataCollector()
	bunch.data = data
	bunch.target = np.array(target)
	bunch.target_name = target_name 
	bunch.feature_name = ()

	return bunch


def load_cifar10(train=True, n_batch=1):
	r"""
	Load cifar10 dataset
	
	Args:
		train: need training set or testing set
		n_batch:
	Returns:
		data dict
	
	10 category in dataset,
	0-airplane, 1-automobile, 2-bird, 3-cat, 4-deer, 5-dog, 6-frog, 7-horse, 8-ship, 9-truck, 
	"""	
	root = dirname(__file__)
	if train:
		filepath = root + '\\data\\cifar10\\data_batch_' + str(n_batch)
	else:
		filepath = root + '\\data\\cifar10\\test_batch'
	with open(filepath, 'rb') as bin_file:
		data = pickle.load(bin_file, encoding='bytes')

	bunch = DataCollector()
	bunch.data = data[b'data'].astype(np.float32)
	bunch.data.resize((10000, 3, 32, 32))
	bunch.target = np.array(data[b'labels'])
	bunch.target_name = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
	bunch.feature_name = ()

	return bunch


def load_iris():
	r"""
	Load iris dataset from Sklearn

	Args:

	Return:
		data dict
	"""

	root = dirname(__file__)
	filepath = root + '\\data\\iris.csv'

	data = np.zeros((4, 150))
	target = []
	target_name = ('setosa', 'versicolor', 'virginica')
	feature_name = ('sepal length (cm)', 'sepal width (cm)',
                    'petal length (cm)', 'petal width (cm)')

	numstr2float = lambda x: float(x)
	with open(filepath, 'r') as csv_file:
		reader = csv.reader(csv_file)
		next(reader)
		for i, sample in enumerate(reader):
			temp = list(map(numstr2float, sample))
			data[:, i] = np.array(temp[:4]).reshape(4,)
			target.append(int(temp[-1]))

	target = np.array(target)
	# shuffle
	index = np.random.permutation(data.shape[1])
	data = data[:, index]
	target = target[index]

	bunch = DataCollector()
	bunch.data = data
	bunch.target = target
	bunch.target_name = target_name
	bunch.feature_name = feature_name

	return bunch


if __name__ == '__main__':

	"""
	from visual_tools import show_cifar10
	cifar = load_cifar10()
	data = cifar.data
	target = cifar.target
	index = 50
	print(target[index])
	show_cifar10('explicit', data[index,:,:,:].transpose(1, 2, 0))
	"""

	load_iris()

