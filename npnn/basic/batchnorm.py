from collections import OrderedDict
import numpy as np
from .layer import Layer


class BatchNorm(Layer):
	r"""
	Batch Normalization Layer that can be placed between hidden layers or be the first layer

	Attributes:
		n_in: numbers of input features
		w: γ value in affine, if affine=False，then w=1
		b: β value in affine, if affine=False，then b=0
		eps: constant to prevent denominator from being 0，default=1e-5
		momentum: momentum factor, default=0.1，for calculating running_mean and running_var
		affine: default=True, True to use γ and β
		w_grad: derivatives of γ
		b_grad: derivatives of β
		running_mean: mean during the training process for evaluation
		running_var: variance during the training process for evaluation
		training: default=True

	Class methods:
		forward: forward propagation
		backward: backpropagation
		state_dict: return parameters
	"""

	def __init__(self, n_in, eps=1e-5, momentum=0.1, affine=True):
		super().__init__()
		self.n_in = n_in
		self.eps = eps
		self.momentum = momentum
		self.affine, self.require_update = affine, affine

		self.w = np.ones((self.n_in, 1))
		self.b = np.zeros((self.n_in, 1))

		self.w_grad = np.zeros_like(self.w, dtype=np.float32)
		self.b_grad = np.zeros_like(self.b, dtype=np.float32)

		self.running_mean = np.zeros((self.n_in, 1))
		self.running_var = np.ones((self.n_in, 1))
		self.training = True
		self.x = 0
		self._temp = 0
		self._temp_cof = 0

	def __str__(self):
		return f'BatchNorm(n_in={self.n_in}, eps={self.eps}, momentum={self.momentum}, affine={self.affine})'

	def _return_self(self):
		return {'BatchNorm':None, 'n_in':self.n_in, 'eps':self.eps,  
				'momentum':self.momentum, 'affine':self.affine}

	def state_dict(self):
		r"""get parameters"""

		return OrderedDict({'weight': self.w, 'bias': self.b, 'running_mean': self.running_mean,
							'running_var': self.running_var})

	def zero_grads(self):

		self.w_grad = np.zeros_like(self.w, dtype=np.float32)
		self.b_grad = np.zeros_like(self.b, dtype=np.float32)

	def forward(self, x):
		self.x = x
		if self.training:
			mean = np.mean(x, axis=1).reshape((-1, 1))
			var = np.var(x, axis=1).reshape((-1, 1))
			# calculate running mean and running var
			self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
			self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var

			self._temp = (x - mean) / (np.sqrt(var + self.eps))
			self._temp_cof = self.w / np.sqrt(var + self.eps)
		# calculate with running mean, running var
		else:
			self._temp = (x - self.running_mean) / (np.sqrt(self.running_var + self.eps))

		return self.w * self._temp + self.b

	def backward(self, acc_grads):
		batch_size = acc_grads.shape[1]
		self.w_grad = (np.sum(acc_grads * self._temp, axis=1) / batch_size).reshape((-1, 1))
		self.b_grad = (np.sum(acc_grads, axis=1) / batch_size).reshape((-1, 1))
		return self._temp_cof * (acc_grads - self._temp * self.w_grad - self.b_grad)


class BatchNorm2d(Layer):
	r"""
	Batch normalization for data of 4 dimensions
	"""	
	def __init__(self, n_in, eps=1e-5, momentum=0.9, affine=True):
		super().__init__()
		self.n_in = n_in
		self.eps = eps
		self.momentum = momentum
		self.affine = affine
		self.require_update = affine

	def forward(self, x):
		raise NotImplementedError('Not implemented yet')

	def backward(self, acc_grads):
		raise NotImplementedError('Not implemented yet')
		