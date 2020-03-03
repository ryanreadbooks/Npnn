import numpy as np 
import npnn.functional as F
from .layer import Layer


class Flatten(Layer):
	r"""	
	Flatten layer. Placed between convolutional layer and fully connected layer

	Class methods:
		forward: flatten the data
		backward: unflatten gradients
	"""
	def __init__(self):
		self._bridge = ()

	def __str__(self):
		return 'Flatten()'

	def _return_self(self):
		return {'Flatten':None}

	def forward(self, x):
		r"""
		Args:
			x: input tensor
		"""

		m, c_in, h_in, w_in = self._bridge = x.shape
		# think of it as using im2col with a kernel that has the same shape of input 
		flatten_data, _, _, _, _ = F.im2col2d(x, (h_in, w_in), (1, 1))
		return flatten_data

	def backward(self, acc_grads):
		r"""
		Process gradients from fully connnected layer then feed it the conv layer

		Args:
			acc_grads: shape=(numbers of neuron of the first fully connected layer, M)
		"""

		tensor_grad = F.col2im2d(acc_grads, self._bridge, (self._bridge[2], self._bridge[3]), (1, 1))

		return tensor_grad
