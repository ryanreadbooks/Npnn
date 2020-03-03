from abc import ABCMeta, abstractmethod
from collections import OrderedDict


class Layer(metaclass=ABCMeta):
	r"""
	Base class for all layers

	Son class overrides forward, backward, state_dict, _return_self method
	"""

	def __init__(self):
		pass

	@abstractmethod
	def forward(self, x):
		pass

	@abstractmethod
	def backward(self, acc_grads):
		pass

	def state_dict(self):
		return OrderedDict()

	def _return_self(self):
		return {}
