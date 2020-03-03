from numpy.random import uniform, normal
from math import sqrt


def normal_init(shape, mean=0, std=0.01):
	r"""
	Normal distribution

	Args:
		shape: shape of parameters
		mean: mean
		std: standard deviation
	"""

	return normal(mean, std, shape)


def xavier_uniform(shape, fan_in, fan_out, gain=1):
	r"""
	Xavier uniform initialization

	Args:
		shape: shape: shape of parameters
		fan_in: numbers of input(or feature maps)
		fan_out: numbers of output(or feature maps)
		gain: gain value
	"""
	u = gain * sqrt(6. / float(fan_in + fan_out))
	return uniform(-u, u, shape)


def xavier_normal(shape, fan_in, fan_out, gain=1):
	r"""
	Xavier normal initialization

	Args:
		shape: shape: shape of parameters
		fan_in: numbers of input(or feature maps)
		fan_out: numbers of output(or feature maps)
		gain: gain value
	"""

	std = gain * sqrt(2. / float(fan_in + fan_out))
	return normal(0, std, shape)


def kaiming_uniform(shape, fan, a=0):
	r"""
	Kaiming uniform
	"""

	gain = sqrt(3. / (1 + a ** 2))
	u = gain * sqrt(2. / fan)
	return uniform(-u, u, shape)


def kaiming_normal(shape, fan, a=0):
	r"""
	Kaiming normal initialization
	"""

	gain = sqrt(3. / (1 + a ** 2))
	std = gain * sqrt(2. / fan)
	return normal(0, std, shape)