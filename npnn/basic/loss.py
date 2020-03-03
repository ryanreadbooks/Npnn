import numpy as np 
from .layer import Layer
from ..utils import one_hot_encoder


class MSELoss(Layer):
	r"""
	Mean square error layer
	"""
	def __init__(self):
		super().__init__()
		self._grads = 0

	def forward(self, x):
		self._out = x
		return x

	def backward(self, acc_grads):
		return self._grads

	def loss(self, y_true, n_class):
		r"""
		Calculate mse loss

		Args:
			y_hat:	output of the network
			y_true: training data
		"""
		m = y_true.shape[0]
		y_true = y_true.reshape(self._out.shape)
		if not (self._out.shape == y_true.shape):
			raise ValueError('Shape of input target vector do not matched.')
		loss = np.sum((y_true - self._out) ** 2) / (2 * m)
		self._grads = -(y_true - self._out) / m

		return loss


def cal_cross_entropy_loss(y_hat, y_true, n_class):
	r"""
	Calculate cross entropy loss

	Args:
		y_hat:	output of the network
		y_true:	true label
		n_class:numbers of class for one hot encoder

	Returns:
		loss: loss value
		grad: original gradients
	"""

	eps = 1e-7
	m = y_true.shape[0]
	y = one_hot_encoder(y_true, y_hat.shape[1], n_class)
	if not (y_hat.shape == y.shape):
		raise ValueError(f'Shape of input target vector do not matched {y_hat.shape}, {y.shape}')
	loss = -(np.sum(y * np.log(y_hat + eps))) / m
	grads = y_hat - y

	return loss, grads