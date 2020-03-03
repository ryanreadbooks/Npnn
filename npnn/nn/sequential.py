from collections import OrderedDict
import numpy as np
from ..basic.layer import Layer
from ..basic.linear import Linear
from ..basic.batchnorm import BatchNorm
from ..basic.dropout import Dropout
from ..basic.conv import Conv2d


class Sequential:
	r"""
	Sequential class, which is used to build the whole neural network

	Attributes:
		_structure: structure of the network base on layer, define layer-0 to be the first layer
		training: training mode, default=True, False for eval

	Class methods:
		state_dict: return parameters
		parameters: get parameters of layer that needs update 
		forward: forward propagation of the whole network
		backward: backpropagation of the whole network
		loss: calculate loss based loss layer(the last layer of the network)
		train: set training mode
		eval: set eval mode(for Dropout and BatchNorm)
		predict: make predictions
	"""

	def __init__(self, *args):
		self._create_struct(args)
		self.training = True

	def __str__(self):
		_format_string = ''
		for index, layer in enumerate(self._structure):
			_format_string = _format_string + f'Layer {(index)}:{layer.__str__()}' + '\n'
		return _format_string

	def _create_struct(self, args):
		r"""
		Create a list that makes structure
		"""

		self._structure = []
		for layer in args:
			if isinstance(layer, Layer):
				pass
			else:
				raise TypeError(f'{type(layer)} is not a part of net')
			self._structure.append(layer)

	def state_dict(self):

		state_dict = OrderedDict()
		for index, layer in enumerate(self._structure):
			layer_state_dict = layer.state_dict()
			if layer_state_dict:
				for key, value in layer_state_dict.items():
					state_dict[f'{index}.{key}'] = value

		return state_dict

	def parameters(self):
		r"""
		Get parameters of layers that needs update
		"""

		updating_layer = []

		for layer in self._structure:
			if isinstance(layer, (Linear, BatchNorm, Conv2d)):
				if layer.require_update:
					updating_layer.append(layer)

		return updating_layer

	def eval(self):

		self.training = False
		for layer in self._structure:
			if isinstance(layer, (BatchNorm, Dropout)):
				layer.training = False

	def train(self):

		self.training = True
		for layer in self._structure:
			if isinstance(layer, (BatchNorm, Dropout)):
				layer.training = True

	def forward(self, x):
		r"""
		Args:
			x: training x
		"""

		for layer in self._structure:
			x = layer.forward(x)
		return x

	def backward(self):
		r"""
		Backpropagate
		"""
		acc_grads = 0
		for layer in self._structure[::-1]:
			acc_grads = layer.backward(acc_grads)

	def predict(self, x):

		self.eval()
		# for classification problem
		if not self._structure[-2].n_out == 1:
			y_p = np.argmax(self.forward(x), axis=0)
		# for regression problem
		else:
			y_p = self.forward(x)
		return y_p

	def loss(self, y):
		r"""
		Calculate loss according to the loss layer		
		参数：
			y: training data
		"""

		loss = self._structure[-1].loss(y, self._structure[-2].n_out)

		return loss

	def zero_grads(self):
		r"""
		Clear grads in layers
		"""

		for layer in self.parameters():
			layer.zero_grads()

