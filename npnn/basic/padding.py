import numpy as np
from .layer import Layer


class ConstantPad2d(Layer):
	r"""
	Constant padding layer

	Attributes:
		padding: padding width
		value: padding value
	"""

	def __init__(self, padding, value):
		if isinstance(padding, tuple):
			self.padding = padding
		elif isinstance(padding, int):
			self.padding = (padding, padding)
		self.value = value

	def __str__(self):
		return f'ConstantPad2d(padding={self.padding}, value={self.value})'

	def _return_self(self):
		return {'ConstantPad2d':None, 'padding':self.padding, 'value':self.value}

	def forward(self, x):
		return np.lib.pad(x, ((0, 0), (0, 0), (self.padding[0], self.padding[0]), (self.padding[1], \
							self.padding[1])), mode='constant', constant_values=self.value)

	def backward(self, acc_grads):
		return acc_grads[:, :, self.padding[0]: -self.padding[0], self.padding[1]: -self.padding[1]]


class EdgePad2d(Layer):
	r"""
	Edge padding layer

	Attributes:
		padding: padding width
	"""
	
	def __init__(self, padding):
		if isinstance(padding, tuple):
			self.padding = padding
		elif isinstance(padding, int):
			self.padding = (padding, padding)

	def __str__(self):
		return f'EdgePad2d(padding={self.padding})'

	def _return_self(self):
		return {'EdgePad2d':None, 'padding':self.padding}	

	def forward(self, x):
		return 	np.lib.pad(x, ((0, 0), (0, 0), (self.padding[0], self.padding[0]), (self.padding[1], \
							self.padding[1])), mode='edge')	

	def backward(self, acc_grads):
		r"""
		Gradients of padding position should be added to the edge
		"""
		
		acc_grads[:, :, self.padding[0], :] += np.sum(acc_grads[:, :, 0: self.padding[0], :], axis=2)
		acc_grads[:, :, -self.padding[0]-1, :] += np.sum(acc_grads[:, :, -self.padding[0]:, :], axis=2)
		acc_grads[:, :, :, self.padding[1]] += np.sum(acc_grads[:, :, :, 0:self.padding[1]], axis=3)
		acc_grads[:, :, :, -self.padding[1]-1] += np.sum(acc_grads[:, :, :, -self.padding[1]:], axis=3)
		
		return acc_grads[:, :, self.padding[0]: -self.padding[0], self.padding[1]: -self.padding[1]]


class ReflectionPad2d(Layer):
	r"""
	Reflection padding layer

	Attributes:
		padding: padding width

	"""
	
	def __init__(self, padding):
		if isinstance(padding, tuple):
			self.padding = padding
		elif isinstance(padding, int):
			self.padding = (padding, padding)

	def __str__(self):
		return f'ReflectionPad2d(padding={self.padding})'

	def _return_self(self):
		return {'ReflectionPad2d':None, 'padding':self.padding}

	def forward(self, x):
		return 	np.lib.pad(x, ((0, 0), (0, 0), (self.padding[0], self.padding[0]), (self.padding[1], \
							self.padding[1])), mode='reflect')			

	def backward(self, acc_grads):
		r"""
		Gradients go to where the padding value comes from
		"""
		acc_grads[:, :, self.padding[0] + 1: 2 * self.padding[0] + 1, :] += acc_grads[:, :, self.padding[0] - 1::-1, :]
		acc_grads[:, :, -2 * self.padding[0] - 1: self.padding[0] - 1, :] += acc_grads[:, :, :-self.padding[0] - 1:-1, :]
		acc_grads[:, :, :, self.padding[1] + 1: 2 * self.padding[1] + 1] += acc_grads[:, :, :, self.padding[1] - 1::-1]
		acc_grads[:, :, :, -2 * self.padding[1] - 1: self.padding[1] - 1] += acc_grads[:, :, :, :-self.padding[1] - 1:-1]

		return acc_grads[:, :, self.padding[0]: -self.padding[0], self.padding[1]: -self.padding[1]]
