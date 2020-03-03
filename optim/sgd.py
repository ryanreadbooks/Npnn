import numpy as np
from .optimizer import Optimizer


class SGD(Optimizer):
	r"""
	SGD optimizer class

	Attributes:
		network: sequential network
		lr: learning rate, required
		weight_decay: default=0
		momentum: momentum factor, default=0
		_updating_layer: updating layers
		_updating_layer_s:

	Class method:
		step: update parameters once
	"""

	def __init__(self, network, lr='required', weight_decay=0, momentum=0):
		super(SGD, self).__init__(network, lr, weight_decay)
		if not 0.0 <= momentum < 1:
			raise ValueError('Invalid value for momentum: {}'.format(momentum))
		self.momentum = momentum
		self._updating_layer_s = [{'w': np.zeros_like(layer.w),
								   'b': np.zeros_like(layer.b)} for layer in self._updating_layer]

	def step(self):
		r"""
		math:
		s_{dw}=\momentum\cdot s_{dw}+(1-\momentum)\cdot dW
		s_{dw}=\momentum\cdot s_{db}+(1-\momentum)\cdot db
		w:=w-\alpha\cdot s_{dw}
		b:=b-\alpha\cdot s_{db}
		"""
		decay = 1 - self.lr * self.weight_decay
		
		for ud_layer, s in zip(self._updating_layer, self._updating_layer_s):
			s['w'] = self.momentum * s['w'] + (1 - self.momentum) * ud_layer.w_grad
			s['b'] = self.momentum * s['b'] + (1 - self.momentum) * ud_layer.b_grad
			
			ud_layer.w *= decay
			ud_layer.w -= self.lr * s['w']
			ud_layer.b -= self.lr * s['b']
