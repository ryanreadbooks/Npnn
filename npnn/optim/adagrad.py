import numpy as np 
from .optimizer import Optimizer


class Adagrad(Optimizer):
	"""
	Adagrad optimizer class

	Attributes:
		network: network sequential
		lr: learning rate, default=1.0
		weight_decay: weight decay factor, optional(default=0）
		eps: default=1e-10
		_updating_layer: updating layers
		_updating_layer_r:

	Class method:
		step: update parameters once
	"""

	def __init__(self, network, lr=0.001, weight_decay=0, eps=1e-10):
		super(Adagrad, self).__init__(network, lr, weight_decay)
		self.eps = eps
		
		self._updating_layer_r = [{'w':np.zeros_like(layer.w),
							'b':np.zeros_like(layer.b)} for layer in self._updating_layer]

	def step(self):
		r"""
		math:
		r\leftarrow r+g^2
		\theta=\theta-\alpha \cdot{g\over {\sqrt{r}+\epsilon}}
		"""
		
		decay = 1 - self.lr * self.weight_decay
		for ud_layer, r in zip(self._updating_layer, self._updating_layer_r):
			r['w'] = r['w'] + np.power(ud_layer.w_grad, 2)
			r['b'] = r['b'] + np.power(ud_layer.b_grad, 2)
			
			den_w = np.sqrt(r['w']) + self.eps
			den_b = np.sqrt(r['b']) + self.eps
			delta_w = self.lr * (ud_layer.w_grad / den_w)
			delta_b = self.lr * (ud_layer.b_grad / den_b)
			
			ud_layer.w *= decay
			ud_layer.w -= delta_w			
			ud_layer.b -= delta_b
