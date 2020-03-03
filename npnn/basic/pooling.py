from collections import OrderedDict
import numpy as np
from .layer import Layer
from .conv_base import _BaseConv
from ..tools import timer
from ..functional import col2im2d


class _BasePool(Layer):
	r"""
	Base class for pooling layer

	Attributes:
		kernel_size: size of pool kernel, default=3
		stride:	strides, default=1
		padding: padding width, not supported yet

	Class methods:
		_pool_forward_base: base forward operation
		_pool_backward_base: base backward operation
	"""

	def __init__(self, kernel_size, stride=2, padding=0):
		if isinstance(kernel_size, tuple):
			self.kernel_size = kernel_size
		elif isinstance(kernel_size, int):
			self.kernel_size = (kernel_size, kernel_size)
		if isinstance(stride, tuple):
			self.stride = stride
		elif isinstance(stride, int):
			self.stride = (stride, stride)
		if isinstance(padding, tuple):
			self.padding = padding
		elif isinstance(padding, int):
			self.padding = (padding, padding)
		self._bridge = ()

	def state_dict(self):
		return OrderedDict()

	def _pool_forward_base(self, input_x, operation):
		r"""
		Args:
			x: input data
			operation: maxpool(0) or avgpool(1)
		"""

		m, c_in, h_in, w_in = input_x.shape
		k_h, k_w = self.kernel_size[0], self.kernel_size[1]
		s_h, s_w = self.stride[0], self.stride[1]

		# calculate output shape
		h_out = int(np.floor((h_in - k_h) / s_h + 1))
		w_out = int(np.floor((w_in - k_w) / s_w + 1))
		self._bridge = (h_out, w_out, m, c_in, h_in, w_in)

		new_shape = (m, c_in, h_out, w_out, k_h, k_w)
		strides = input_x.strides
		new_strides = (*strides[0:2], strides[2] * s_h, strides[3] * s_w, *strides[-2:])
		pre_pool_data = np.lib.stride_tricks.as_strided(input_x, shape=new_shape, strides=new_strides)
		temp = pre_pool_data.reshape(m, c_in, h_out * w_out, k_h * k_w).transpose(0, 1, 3, 2)
		unfolded_x = np.concatenate(np.concatenate(temp, axis=2), axis=0)
		unfolded_x = unfolded_x.T.reshape((-1, k_h * k_w))

		if not operation:
			# max index
			max_index_buf = np.argmax(unfolded_x, axis=1)
			out_tensor = pre_pool_data.max(axis=(4,5))
		else:
			out_tensor = pre_pool_data.mean(axis=(4,5))

		return out_tensor if operation else (out_tensor, max_index_buf)

	def _pool_backward_base(self, acc_grads, operation):
		r"""
		Args:
			acc_grads: accumulated gradients
			operation: maxpool(0) or avgpool(1)
		"""

		h_out, w_out, m, c_in, h_in, w_in = self._bridge
		k_h, k_w = self.kernel_size[0], self.kernel_size[1]
		s_h, s_w = self.stride[0], self.stride[1]
		shape0 = h_out * w_out * m * c_in
		shape1 = k_w * k_h
		shape = (shape0, shape1)
		
		if not operation:
			to_previous = np.zeros(shape, dtype=np.float16)
			to_previous[np.arange(self._max_index.size), self._max_index.flatten()] = acc_grads.transpose(0,2,3,1).flatten()
		else:
			to_previous = np.zeros((shape0, 1), dtype=np.float16)
			to_previous[np.arange(shape0), 0] = (acc_grads.transpose(0,2,3,1).flatten() / shape1)
			to_previous = np.repeat(to_previous, shape1, axis=1)

		to_previous = to_previous.reshape(-1, shape1 * c_in).T
		to_previous = col2im2d(to_previous, (m, c_in, h_in, w_in), (k_h, k_w), (s_h, s_w))		
		return to_previous


class MaxPool2d(_BasePool):
	r"""
	Max pool class
	"""

	def __init__(self, kernel_size=2, stride=2, padding=0):
		super(MaxPool2d, self).__init__(kernel_size, stride, padding)
		self._max_index = None

	def __str__(self):
		return f'MaxPool2d(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})'

	def _return_self(self):
		return {'MaxPool2d':None, 'kernel_size':self.kernel_size, 'stride':self.stride, 'padding':self.padding}	

	def forward(self, x):
		r"""call father function"""
		out_tensor, self._max_index = self._pool_forward_base(x, 0)
		return out_tensor

	def backward(self, acc_grads):
		r"""call father function"""
		to_previous = self._pool_backward_base(acc_grads, 0)

		return to_previous


class AvgPool2d(_BasePool):
	r"""
	Average pool class
	"""

	def __init__(self, kernel_size=2, stride=2, padding=0):
		super(AvgPool2d, self).__init__(kernel_size, stride, padding)

	def __str__(self):
		return f'AvgPool2d(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})'

	def _return_self(self):
		return {'AvgPool2d':None, 'kernel_size':self.kernel_size, 'stride':self.stride, 'padding':self.padding}	

	def forward(self, x):
		r"""call father function"""
		out_tensor = self._pool_forward_base(x, 1)

		return out_tensor

	def backward(self, acc_grads):
		r"""call father function"""
		to_previous = self._pool_backward_base(acc_grads, 1)

		return to_previous
