from abc import ABCMeta, abstractmethod
from ..nn.sequential import Sequential
from ..basic.layer import Layer


class Optimizer(metaclass=ABCMeta):
	r"""
	Base class for optimizer class, overriding step method is needed

	"""

	def __init__(self, network, lr='required', weight_decay=0):
		if lr == 'required':
			raise TypeError('lr is required for optimizer, now missing lr.')			
		if lr < 0.0:
			raise ValueError('Invalid value for lr: {}'.format(lr))
		if weight_decay < 0.0:
			raise ValueError('Invalid value for weight_decay: {}'.format(weight_decay))
		# shared attributes
		self._updating_layer = []
		if isinstance(network, Sequential):
			self._updating_layer.extend(network.parameters())
		elif isinstance(network, Layer):
			self._updating_layer.append(network)
		else:
			raise TypeError('type of Sequential or Layer is required instead of %s' % type(network))
		self.lr = lr
		self.weight_decay = weight_decay
		
	@abstractmethod
	def step(self):
		raise NotImplementedError('method step not implemented.')
