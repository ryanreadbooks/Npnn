import time
import numpy as np


def one_hot_encoder(y, m, n_of_class):
	r"""
	Convert true target into a vector

	Args:
	y -- True target of all examples
	m -- Number of examples

	Returns:
	y_encoded -- A matrix shape of (number of class, number of examples)
	"""

	y_encoded = np.zeros((n_of_class, m))
	for i in range(m):
		y_encoded[y[i], i] = 1

	return y_encoded


def get_mini_batch2d(x, y, n_batch, shuffle=True):
	r"""
	Get mini batch from raw x and y with two dimensions

	Args:
		x -- raw x data
		y-- raw y data
		n_batch -- The number of batch divided from raw x data

	Return:
		mini_batch -- Python list of mini batch
	"""

	assert x.shape[1] == y.shape[0], 'Size of data and target are not matched.'
	if shuffle:
		# Shuffle examples
		index = np.random.permutation(x.shape[1])
		x = x[:, index]
		y = y[index]
	else:
		pass
	return np.array_split(x, n_batch, axis=1), np.array_split(y, n_batch)


def get_mini_batch3d(x, y, n_batch, shuffle=True):
	r"""
	Get mini batch from raw x and y with 4 dimensions

	Args:
		x -- raw x data
		y-- raw y data
		n_batch -- The number of batch divided from raw x data

	Return:
		mini_batch -- mini batch数据，python的list格式
	"""
	assert len(x.shape) == 4, 'data is not of dimension 3'
	assert x.shape[0] == y.shape[0], f'{x.shape} {y.shape} size of data and target are not matched.'
	if shuffle:
		index = np.random.permutation(x.shape[0])
		x = x[index, :, :, :]
		y = y[index]
	else:
		pass
	return np.array_split(x, n_batch, axis=0), np.array_split(y, n_batch)
