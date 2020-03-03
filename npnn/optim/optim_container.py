import threading
import numpy as np
from ..nn.sequential import Sequential
from .adadelta import Adadelta
from .adagrad import Adagrad
from .adam import Adam
from .rmsprop import RMSprop
from .sgd import SGD

__types__ = ['SGD', 'Adagrad', 'Adadelta', 'RMSprop', 'Adam']

class OptimContainer():
	r"""
	Optimizer container
	Support different optimizers for different layers

	Args:
		network: sequential network
		group: optimizer group
			e.g. group=[{'type':'Adam', 'lr':0.001, 'betas':(0.9, 0.999)},
						{'type':'RMSprop', 'lr':0.005, 'beta':0.9},
						{'type':'SGD', 'lr':0.01, 'momentum':0.99},
						{'type':'Adagrad', 'lr':0.01}]
		
	Class method:
		step: update parameters once
	"""

	def __init__(self, network, group):
		if not isinstance(network, Sequential):
			raise TypeError('type of Sequential is required instead of %s' % type(network))
		if not (len(network.parameters()) == len(group)):
			raise TypeError('numbers of updating layers of network and optimizers do not match.')
		self._updating_layer = network.parameters()
		self.optim_group = []
		self._group_checker_creator(group)

	def _group_checker_creator(self, group):
		r"""
		Check legitimacy
		"""

		for i, optim in enumerate(group):
			if optim['type'] not in __types__:
				raise TypeError('optimizer of type {} is not supported.'.format(optim['type']))
			if optim['type'] == 'SGD':
				if 'weight_decay' in optim:
					wd = optim['weight_decay']
				else:
					wd = 0
				if 'momentum' in optim:
					momentum = optim['momentum']
				else:
					momentum = 0
				self.optim_group.append(SGD(self._updating_layer[i], lr=optim['lr'], weight_decay=wd, momentum=momentum))
			elif optim['type'] == 'RMSprop':
				if 'weight_decay' in optim:
					wd = optim['weight_decay']
				else:
					wd = 0
				if 'beta' in optim:
					beta = optim['beta']
				else:
					beta = 0.99
				if 'eps' in optim:
					eps = optim['eps']
				else:
					eps = 1e-8
				self.optim_group.append(RMSprop(self._updating_layer[i], lr=optim['lr'], weight_decay=wd, beta=beta, eps=eps))
			elif optim['type'] == 'Adam':
				if 'weight_decay' in optim:
					wd = optim['weight_decay']
				else:
					wd = 0
				if 'betas' in optim:
					betas = optim['betas']
				else:
					betas = (0.9, 0.98)
				if 'eps' in optim:
					eps = optim['eps']
				else:
					eps = 1e-7
				self.optim_group.append(Adam(self._updating_layer[i], lr=optim['lr'], weight_decay=wd, betas=betas, eps=eps))
			elif optim['type'] == 'Adagrad':
				if 'weight_decay' in optim:
					wd = optim['weight_decay']
				else:
					wd = 0
				if 'eps' in optim:
					eps = optim['eps']
				else:
					eps = 1e-10		
				self.optim_group.append(Adagrad(self._updating_layer[i], lr=optim['lr'], weight_decay=wd, eps=eps))	
			elif optim['type'] == 'Adadelta':
				if 'weight_decay' in optim:
					wd = optim['weight_decay']
				else:
					wd = 0
				if 'rho' in optim:
					rho = optim['rho']
				else:
					rho = 0.9
				if 'eps' in optim:
					eps = optim['eps']
				else:
					eps = 1e-6
				self.optim_group.append(Adadelta(self._updating_layer[i], lr=optim['lr'], rho=rho, weight_decay=wd, eps=eps))	

	def step(self, parallel=False):
		r"""

		Args:
			parallel: use parallel updating, default=Fasle, maybe slower than usual
		"""
		if not parallel:
			for optm in self.optim_group:
				optm.step()
		else:
			for optim in self.optim_group:
				t = threading.Thread(target=optim.step)
				t.start()
