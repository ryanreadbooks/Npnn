import numpy as np
from .optimizer import Optimizer


class RMSprop(Optimizer):
	r"""
	Root Mean Square Propagation optimizer
	
	Attributes:
		network: network sequential
		lr: learning rate, default=0.003
		weight_decay: weight decay factor, optional(default=0）
		eps: default=1e-8
		beta: default=0.99
		_updating_layer: updating layers
		_updating_layer_r:
	
	Class method:
		step: update parameters once
	"""

	def __init__(self, network, lr=0.003, weight_decay=0, beta=0.99, eps=1e-8):
		super(RMSprop, self).__init__(network, lr, weight_decay)
		if not 0 <= beta < 1:
			raise ValueError('Invalid value for beta:{}'.format(beta))
		if eps < 0.0:
			raise ValueError('Invalid value for eps:{}'.format(eps))
		self.beta = beta
		self.eps = eps
		self._updating_layer_r = [{'w': np.zeros_like(layer.w),
								   'b': np.zeros_like(layer.b)} for layer in self._updating_layer]

	def step(self):
		r"""
		math:
		r_{dW}=\beta\cdot r_{dw}+(1-\beta)\cdot(dW)^2
		r_{db}=\beta\cdot r_{db}+(1-\beta)\cdot(db)^2
		w:=w-\alpha\cdot{dW\over\sqrt{r_{dW}}+\epsilon}
		b:=b-\alpha\cdot{db\over\sqrt{r_{db}}+\epsilon}
		"""

		decay = 1 - self.lr * self.weight_decay
		for ud_layer, r in zip(self._updating_layer, self._updating_layer_r):
			
			r['w'] = self.beta * r['w'] + (1 - self.beta) * (ud_layer.w_grad ** 2)
			r['b'] = self.beta * r['b'] + (1 - self.beta) * (ud_layer.b_grad ** 2)
			temp_r_w = np.sqrt(r['w']) + self.eps
			temp_r_b = np.sqrt(r['b']) + self.eps
			
			delta_w = self.lr * (ud_layer.w_grad / temp_r_w)
			delta_b = self.lr * (ud_layer.b_grad / temp_r_b)
			
			ud_layer.w *= decay
			ud_layer.w -= delta_w			
			ud_layer.b -= delta_b
