import numpy as np
from .optimizer import Optimizer


class Adam(Optimizer):
	"""
	Adam optimizer class

	Attributes:
		network: network sequential
		lr: learning rate, default=0.005
		weight_decay: weight decay factor, optional(default=0）
		eps: default=1e-7
		betas: default=(0.9,0.98), format of tuple
		_updating_layer: updating layers		
		_t:
		_updating_layer_s: accumulated gradients
		_updating_layer_r: accumulated square values of gradients

	Class method:
		step: update parameters once
	"""

	def __init__(self, network, lr=0.005, weight_decay=0, betas=(0.9, 0.98), eps=1e-7):
		super(Adam, self).__init__(network, lr, weight_decay)
		if not isinstance(betas, (list, tuple)):
			raise TypeError('Invalid type for betas, betas must be list or tuple.')
		if not (0.0 <= betas[0] < 1) or not (0.0 <= betas[1] < 1):
			raise ValueError('Invalid value for betas: {}'.format(betas))
		if eps < 0.0:
			raise ValueError('Invalid value for eps: {}'.format(eps))
		self.beta = betas[0]
		self.momentum = betas[1]
		self.eps = eps
		self._t = 0
		# 初始化一阶矩和二阶矩
		self._updating_layer_s = [{'w': np.zeros_like(layer.w),
								   'b': np.zeros_like(layer.b)} for layer in self._updating_layer]
		self._updating_layer_r = [{'w': np.zeros_like(layer.w),
								   'b': np.zeros_like(layer.b)} for layer in self._updating_layer]

	def step(self):
		r"""
		np:
		t\leftarrow t+1
		s=\beta_{1}\cdot s_+(1-\beta_{1})\cdot g
		r=\beta_{2}\cdot r+(1-\beta_{2})\cdot (g)^2
		\hat{s}\leftarrow {s\over 1-\beta_{1}^t}
		\hat{r}\leftarrow {r\over 1-\beta_{2}^t}
		\theta\leftarrow \theta-\alpha{\hat{s}\over\sqrt{\hat{r}}+\epsilon}
		"""

		decay = 1 - self.lr * self.weight_decay
		self._t += 1
		for ud_layer, s, r in zip(self._updating_layer, self._updating_layer_s, self._updating_layer_r):
			# 更新一阶矩
			s['w'] = self.momentum * s['w'] + (1 - self.momentum) * ud_layer.w_grad
			s['b'] = self.momentum * s['b'] + (1 - self.momentum) * ud_layer.b_grad
			# 更新二阶矩
			r['w'] = self.beta * r['w'] + (1 - self.beta) * (ud_layer.w_grad ** 2)
			r['b'] = self.beta * r['b'] + (1 - self.beta) * (ud_layer.b_grad ** 2)
			# 修正一阶矩偏差
			den_s = 1 - self.momentum ** self._t
			corrected_s_w = s['w'] / den_s
			corrected_s_b = s['b'] / den_s
			# 修正二阶矩偏差
			den_r = 1 - self.beta ** self._t
			corrected_r_w = r['w'] / den_r
			corrected_r_b = r['b'] / den_r
			# 计算更新
			delta_w = self.lr * (corrected_s_w / (np.sqrt(corrected_r_w) + self.eps))
			delta_b = self.lr * (corrected_s_b / (np.sqrt(corrected_r_b) + self.eps))
			# 更新参数
			ud_layer.w *= decay
			ud_layer.w -= delta_w			
			ud_layer.b -= delta_b
