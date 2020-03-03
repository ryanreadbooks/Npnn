from abc import abstractmethod
import numpy as np
from .layer import Layer
from .loss import cal_cross_entropy_loss


class Activation(Layer):
	r"""
	Base class for activation son class

	Attributes
		x: inpux
		_out: output after activation
	"""

	def __init__(self):
		super().__init__()
		self.x = 0
		self._out = 0

	@abstractmethod
	def forward(self, x):
		pass

	@abstractmethod
	def backward(self, acc_grads):
		pass


class ReLU(Activation):

	def __int__(self):
		super().__init__()

	def __str__(self):
		return 'ReLU()'

	def _return_self(self):
		return {'ReLU':None}

	def forward(self, x):
		self.x = x
		self._out = np.maximum(0, x)
		return self._out

	def backward(self, acc_grads):
		# acc_grads: accumulated gradients from last layer in backprop
		_grad = np.where(self.x >= 0, 1, 0)

		return _grad * acc_grads


class Sigmoid(Activation):

	def __int__(self):
		super().__init__()

	def __str__(self):
		return 'Sigmoid()'

	def _return_self(self):
		return {'Sigmoid':None}	

	def forward(self, x):
		self.x = x
		self._out = 1 / (1 + np.exp(-x))
		return self._out

	def backward(self, acc_grads):
		_grad = self._out * (1 - self._out)
		return _grad * acc_grads


class Tanh(Activation):

	def __int__(self):
		super().__init__()

	def __str__(self):
		return 'Tanh()'

	def _return_self(self):
		return {'Tanh':None}

	def forward(self, x):
		self.x = x
		self._out = np.tanh(x)
		return self._out

	def backward(self, acc_grads):
		_grad = 1 - (self._out ** 2)
		return _grad * acc_grads


class Softmax(Activation):
	r"""
	Softmax activation function with loss
	"""

	def __int__(self):
		super().__init__()
		self._grads = 0

	def __str__(self):
		return 'Softmax()'

	def _return_self(self):
		return {'Softmax':None}	

	def forward(self, x):
		self.x = x
		self._out = np.exp(x - np.max(x, axis=0)) / np.sum(np.exp(x - np.max(x, axis=0)), axis=0)
		return self._out

	def backward(self, acc_grads):
		r"""
		Combine softmax with cross entropy loss
		"""

		return self._grads

	def loss(self, y, n_class):
		r"""
		compute cross entropy loss and the original gradients 
		"""

		loss, self._grads = cal_cross_entropy_loss(self._out, y, n_class)
		return loss

class ELU(Activation):
	r"""
	Math: max(0,x) + min(0, alpha * (exp(x)-1))
	"""

	def __init__(self, alpha=1):
		super().__init__()
		self.alpha = 1

	def __str__(self):
		return 'ELU()'

	def _return_self(self):
		return 	{'ELU':None, 'alpha':self.alpha}
		
	def forward(self, x):
		self.x = x
		self._out = np.maximum(0, x) + np.minimum(0, self.alpha * (np.exp(x) - 1))
		return self._out

	def backward(self, acc_grads):
		_grad = np.where(self.x >= 0, 1, self.alpha * np.exp(self.x))
		return _grad * acc_grads


class LeakyReLU(Activation):
	r"""
	Math: max(alpha*x, x), alpha=0.01(default)
	"""

	def __init__(self, alpha=0.01):
		super().__init__()
		self.alpha = alpha

	def __str__(self):
		return 'LeakyReLU()'

	def _return_self(self):
		return 	{'LeakyReLU':None, 'alpha':self.alpha}

	def forward(self, x):
		self.x = x
		self._out = np.maximum(self.alpha * x, x)
		return self._out

	def backward(self, acc_grads):
		_grad = np.where(self.x >= 0, 1, self.alpha)
		return _grad * acc_grads


class GeLU(Activation):
	r"""
	GeLU: Gaussian error linear unit
	Math: GeLU(x) = x * φ(x) ≈ x*0.5*(1 + tanh(sqrt(2/pi) * (x + 0.044715 * x ** 3)))
		derivative： GeLU'(x) = φ(x) + x * φ'(x)

	Attribute:
		_Phi_x: cumulated distribution function of standard normal distribution
	"""

	def __init__(self):
		super().__init__()
		self._Phi_x = 0

	def __str__(self):
		return 'GeLU()'

	def _return_self(self):
		return {'GeLU':None}	

	def forward(self, x):
		self.x = x
		self._Phi_x = 0.5 * (1 + np.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * (x ** 3)))))
		self._out = x * self._Phi_x
		return self._out

	def backward(self, acc_grads):
		# probability density function of standard normal distribution
		_phi_x = np.exp(-(self.x ** 2) / 2) / np.sqrt(2 * np.pi)
		_grad = self._Phi_x + self.x * _phi_x
		return _grad * acc_grads
