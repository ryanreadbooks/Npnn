import time
import numpy as np
from .layer import Layer


class _BaseConv(Layer):
	r"""
	Base class for convolutional layer class
			
	"""

	def __init__(self, in_channels, out_channels, kernel_size, stride, padding, padding_mode=None, require_update=True):
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		if isinstance(kernel_size, tuple):
			self.kernel_size = kernel_size
		elif isinstance(kernel_size, int):
			self.kernel_size = (kernel_size, kernel_size)
		if isinstance(stride, tuple):
			self.stride = stride
		elif isinstance(stride, int):
			self.stride = (stride, stride)
		if not padding_mode:
			pass
		elif not padding_mode == 'same':
			raise TypeError(f'padding mode of {padding_mode} is not supported, only same mode is supported')
		self.padding_mode = padding_mode
		if isinstance(padding, tuple):
			self.padding = padding
		elif isinstance(padding, int):
			self.padding = (padding, padding)
		self.require_update = require_update

		self.w = None
		self.b = None
		self.w_grad = None
		self.b_grad = None

	def zero_grads(self):

		self.w_grad = np.zeros_like(self.w, dtype=np.float32)
		self.b_grad = np.zeros_like(self.b, dtype=np.float32)

	def _init_params(self):
		pass

	def forward(self, x):
		raise NotImplementedError('Have not implemented this method, have to overwrite it.')

	def backward(self, acc_grads):
		raise NotImplementedError('Have not implemented this method, have to overwrite it.')
