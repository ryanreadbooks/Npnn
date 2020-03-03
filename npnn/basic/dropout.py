import numpy as np
from .layer import Layer


class Dropout(Layer):
	r"""
	Dropout layer

	Attributes:
		training: training mode, default=True
		p: the probability of keeping a neuron during training
	
	Class methods:
		forward: 
		backward: 
	"""

	def __init__(self, p='required'):
		super().__init__()
		if p == 'required':
			raise TypeError('p is needed')
		if not (0 <= p < 1):
			raise ValueError('Invalid value for p: {}'.format(p))
		self.p = p
		self.training = True
		self._deactivation = 0

	def __str__(self):
		return f'Dropout(p={self.p})'

	def _return_self(self):
		return {'Dropout':None, 'p':self.p}

	def forward(self, x):
		# if training
		if self.training:
			self._deactivation = np.random.binomial(1, self.p, size=x.shape[0]).reshape((-1, 1))
			return (x * self._deactivation) / self.p
		else:
			return x

	def backward(self, acc_grads):
		return (acc_grads * self._deactivation) / self.p
