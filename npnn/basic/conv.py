from collections import OrderedDict
import numpy as np
from .conv_base import _BaseConv
from ..nn.init import kaiming_normal 
import npnn.functional as F


class Conv2d(_BaseConv):
	r"""
	Convolutional layer for 4 dimensional data

	Attributes:
		in_channels: numbers of input channel
		out_channels: number of output channel(numbers of filters)
		kernel_size: size of kernel(filter), default=3
		stride: stride, default=1
		padding: padding(pad 0), default=0, 
		padding_mode: support same padding, dafault=None
		b: bias, shape=(out_channels,)
		w: weights of kernel, shape=(out_channels, in_channels, kernel_size_H, kernel_size_W)
		require_update: default=True
		w_grad: derivatives of weights
		b_grad: derivatives of bias
	
	Class methods:
		_init_params: initialization
		_restore_kernel2tensor: restore kernel to the format of tensor
		forward: forward propagation
		backward: backward propagation
		state_dict: return parameters

	"""

	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, padding_mode=None, require_update=True):
		super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, padding_mode, require_update)
		self._init_params()
		self.x = 0
		self._bridge = {'x_t_shape':None, 'out_padding':()}

	def __str__(self):
		string_format = f'Conv2d(in_channels={self.in_channels}, out_channels={self.out_channels}, ' \
						f'kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, ' \
						f'padding_mode={self.padding_mode}, require_update={self.require_update})'
		return string_format

	def _return_self(self):
		return {'Conv2d':None ,'in_channels':self.in_channels, 'out_channels':self.out_channels, 'kernel_size':self.kernel_size,
				'stride':self.stride, 'padding':self.padding, 'padding_mode':self.padding_mode, 'require_update':self.require_update}

	def _init_params(self):
		r"""
		Initialize weights and bias and gradients
		"""
		k_size = (self.out_channels, self.kernel_size[0] * self.kernel_size[1] * self.in_channels)
		fan_out = self.out_channels * self.kernel_size[0] * self.kernel_size[1]
		self.w = kaiming_normal(k_size, fan_out)
		self.b = np.zeros((self.out_channels, 1))
		self.w_grad = np.zeros_like(self.w, dtype=np.float32)
		self.b_grad = np.zeros_like(self.b, dtype=np.float32)

	def state_dict(self):

		return OrderedDict({'weight': self._restore_kernel2tensor(), 'bias': self.b})
		
	def _restore_kernel2tensor(self):

		return self.w.reshape((self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]))

	def forward(self, x):
		r"""
		forward propagation

		Args:
			x: input x, shape=(M, C, H, W)
		"""

		# pad input if padding is set
		if self.padding_mode == 'same':
			x, self._bridge['out_padding'] = F.pad(x, self.padding, 'define', 'same', stride=self.stride, kernel_size=self.kernel_size)
		elif not self.padding_mode and self.padding != (0, 0):
			x, self._bridge['out_padding'] = F.pad(x, self.padding, 'define', 'optional')

		# use im2col to calculate forward
		self.x, h_out, w_out, m, self._bridge['x_t_shape'] = F.im2col2d(x, self.kernel_size, self.stride)
		z = self.w @ self.x + self.b
		# reshape result from im2col to format of tensor
		out_tensor = np.array([k.reshape((self.out_channels, h_out, w_out)) for k in np.split(z, m, axis=1)])

		return out_tensor

	def backward(self, acc_grads):
		r"""

		Args:
			acc_grads: accumulated gradients, format of tensor
		"""

		# reshape grads to format of a 2 dimensional matrix
		m, c, h, w = acc_grads.shape
		grad = np.concatenate([k.reshape((c, h * w)) for k in np.split(acc_grads, m, axis=0)], axis=1)
		if self.require_update:
			self.w_grad = grad @ self.x.T
			self.b_grad = np.sum(grad, axis=1, keepdims=True)
		to_previous = F.col2im2d((self.w.T @ grad), self._bridge['x_t_shape'], self.kernel_size, self.stride)

		# has constant padding
		if self._bridge['out_padding']:
			o_p_h, o_p_w, k = self._bridge['out_padding']
			to_previous = to_previous[::, ::, o_p_h * k: -o_p_h, o_p_w * k: -o_p_w]

		return to_previous
