from collections import OrderedDict
import numpy as np
from .layer import Layer
from ..nn.init import kaiming_uniform


class Linear(Layer):
	r"""
	Linear layer

	Attributes:
		w: weight matrix, shape=(num of neurons in current layer × num of neurons in previous layer)
		b: bias
		n_in: num of neurons in previous layer
		n_out: num of neurons in current layer
		w_grad: derivatives of w
		b_grad: derivatives of b
		x: input x
		require_update: default=True

	Class methods:
		_init_params: initialization
		forward: forward propagation
		backward: backpropagation
		state_dict: return parameters
	"""

	def __init__(self, n_in, n_out, require_update=True):
		super().__init__()
		self.n_in = n_in
		self.n_out = n_out
		self.require_update = require_update
		self._init_params()
		self.w_grad = np.zeros_like(self.w, dtype=np.float32)
		self.b_grad = np.zeros_like(self.b, dtype=np.float32)
		self.x = 0

	def __str__(self):
		_format_string = f'Linear(n_in={self.n_in}, n_out={self.n_out}, require_update={self.require_update})'
		return _format_string

	def _return_self(self):
		return {'Linear':None, 'n_in':self.n_in, 'n_out':self.n_out, 'require_update':self.require_update}

	def state_dict(self):
		return OrderedDict({'weight': self.w, 'bias': self.b})

	def _init_params(self):
		self.w = kaiming_uniform((self.n_out, self.n_in), self.n_in)
		self.b = np.zeros((self.n_out, 1))
		self.w_grad = 0
		self.b_grad = 0

	def zero_grads(self):
		self.w_grad = np.zeros_like(self.w, dtype=np.float32)
		self.b_grad = np.zeros_like(self.b, dtype=np.float32)

	def forward(self, x):
		r"""
		Forward propagation
		Math: Z = wx + b

		Args:
			x: input, shape=(num of neurons in previous layer × M)
		"""
		assert (self.w.shape[1] == x.shape[0]), f'dimensions {self.w.shape}, {x.shape}do not match, check shape of input.'
		self.x = x
		z = self.w @ x + self.b
		return z

	def backward(self, acc_grads):
		if self.require_update:
			m = acc_grads.shape[1]
			self.w_grad = (acc_grads @ self.x.T) / m
			self.b_grad = np.sum(acc_grads, axis=1, keepdims=True) / m
		to_previous = self.w.T @ acc_grads
		return to_previous
