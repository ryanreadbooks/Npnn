import numpy as np
from .optimizer import Optimizer


class Adadelta(Optimizer):
	"""
	Adadelta optimizer class

	Attributes:
		network: network sequential
		lr: learning rate, default=1.0
		rho: rho, default=0.9
		weight_decay: weight decay factor, optional(default=0）
		eps: default=1e-6
		_updating_layer: updating layers
		_updating_layer_r:
		_updating_delta:

	Class method:
		step: update parameters once
	"""

	def __init__(self, network, lr=1.0, rho=0.9, weight_decay=0, eps=1e-6):
		super(Adadelta, self).__init__(network, lr, weight_decay)
		self.rho = rho
		self.eps = eps
		self.lr = lr
		self._updating_layer_r = [{'w': np.zeros_like(layer.w),
								   'b': np.zeros_like(layer.b)} for layer in self._updating_layer]
		self._updating_delta = [{'w': np.zeros_like(layer.w),
								 'b': np.zeros_like(layer.b)} for layer in self._updating_layer]

	def step(self):
		r"""
		math:
		E[g^2]_t=\rho E[g^2]+(1-\rho)g^2_t\\
		\Delta x_t={\sqrt{E[\Delta{x^2}]+\epsilon}\over{E[g^2_t]+\epsilon}} \cdot g_t\\
		E[\Delta x]_t=\rho E[\Delta x^2]_{t-1} + (1-\rho)\Delta x_t^2\\
		x_{t+1}=x_t+\Delta x_t
		"""

		decay = 1 - self.lr * self.weight_decay
		for ud_layer, r, delta in zip(self._updating_layer, self._updating_layer_r, self._updating_delta):

			r['w'] = self.rho * r['w'] + (1 - self.rho) * (ud_layer.w_grad ** 2)
			r['b'] = self.rho * r['b'] + (1 - self.rho) * (ud_layer.b_grad ** 2)
			
			delta_w = (np.sqrt(delta['w'] + self.eps) / np.sqrt(r['w'] + self.eps)) * ud_layer.w_grad
			delta_b = (np.sqrt(delta['b'] + self.eps) / np.sqrt(r['b'] + self.eps)) * ud_layer.b_grad
			
			delta['w'] = self.rho * delta['w'] + (1 - self.rho) * (delta_w ** 2)
			delta['b'] = self.rho * delta['b'] + (1 - self.rho) * (delta_b ** 2)

			ud_layer.w *= decay
			ud_layer.w -= delta_w
			ud_layer.b -= delta_b
