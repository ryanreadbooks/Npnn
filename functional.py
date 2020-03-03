import numpy as np
from .tools import timer 


def im2col2d(input_x, kernel, stride):
	r"""
	Implement im2col

	Args:
		input_x: input data,shape=(m,c_in,h_in,w_in)
		kernel: what kernel to use
		stride:	what stride to use
	"""

	m, c_in, h_in, w_in = input_x.shape
	k_h, k_w = kernel[0], kernel[1]
	s_h, s_w = stride[0], stride[1]

	# out of bound handling, pad 0 to it
	p_h = int(s_h * (np.ceil((h_in - k_h) / s_h + 1) - 1) + k_h - h_in)
	p_w = int(s_w * (np.ceil((w_in - k_w) / s_w + 1) - 1) + k_w - w_in)
	if p_h or p_w:
		input_x = pad(input_x, (p_h, p_w), condition='kernel_overflow')
	# output size after padding
	h_out = int((h_in - k_h + p_h) / s_h + 1)
	w_out = int((w_in - k_w + p_w) / s_w + 1)

	new_shape = (m, c_in, h_out, w_out, k_h, k_w)
	strides = input_x.strides
	new_strides = (*strides[0:2], strides[2] * s_h, strides[3] * s_w, *strides[-2:])
	pre_conv_data = np.lib.stride_tricks.as_strided(input_x, shape=new_shape, strides=new_strides)	
	temp = pre_conv_data.reshape(m, c_in, h_out * w_out, k_h * k_w).transpose(0, 1, 3, 2)
	unfolded_x = np.concatenate(np.concatenate(temp, axis=2), axis=0)

	return unfolded_x, h_out, w_out, m, input_x.shape


def do_conv_without_unfolding_kernel(input_x, kernel, stride):

	m, c_in, h_in, w_in = input_x.shape
	k_h, k_w = kernel[0], kernel[1]
	s_h, s_w = stride[0], stride[1]

	p_h = int(s_h * (np.ceil((h_in - k_h) / s_h + 1) - 1) + k_h - h_in)
	p_w = int(s_w * (np.ceil((w_in - k_w) / s_w + 1) - 1) + k_w - w_in)
	if p_h or p_w:
		input_x = pad(input_x, (p_h, p_w), condition='kernel_overflow')

	h_out = int((h_in - k_h + p_h) / s_h + 1)
	w_out = int((w_in - k_w + p_w) / s_w + 1)

	new_shape = (m, c_in, h_out, w_out, k_h, k_w)
	strides = input_x.strides
	new_strides = (*strides[0:2], strides[2] * s_h, strides[3] * s_w, *strides[-2:])
	pre_conv_data = np.lib.stride_tricks.as_strided(input_x, shape=new_shape, strides=new_strides)
	out_tensor = np.tensordot(pre_conv_data, kernel, [(1, 4, 5), (1, 2, 3)]).swapaxes(1, 3).swapaxes(2, 3)
	
	return out_tensor


def col2im2d(unfolded_x, shape, kernel, stride):
	r"""
	Implement col2im

	Args:
		unfolded_x: data from im2col
		shape: original shape, which also is the shape to be restored
		kernel: what kernel to use
		stride: what stride to use
	"""

	m, c_in, h_in, w_in = shape

	k_h, k_w = kernel[0], kernel[1]
	s_h, s_w = stride[0], stride[1]

	# original output size
	h_out = int((h_in - k_h) / s_h + 1)
	w_out = int((w_in - k_w) / s_w + 1)

	# result location
	res = np.zeros((m, c_in, h_in, w_in))
	# counter location
	cnt_mat = np.zeros((m, c_in, h_in, w_in))

	for index_m_i, m_i in enumerate(np.split(unfolded_x, m, axis=1)):
		cnt = 0
		# window sliding
		for row in range(h_out):
			for col in range(w_out):
				res[index_m_i, ::, row * s_h: row * s_h + k_h, col * s_w: col * s_w + k_w] += m_i[:, cnt].reshape(
					(c_in, k_h, k_w))
				cnt_mat[index_m_i, :, row * s_h: row * s_h + k_h, col * s_w: col * s_w + k_w] += 1
				cnt += 1
	res /= cnt_mat

		return res


def pad(x, length, condition, padding_mode='optional', **kwargs):
	r"""
	Args:
		x: data that needs to be padded, shape=(m, c_in, h_in, w_in)
		length: width of padding(height, width)
		condition: self define usage condition
		padding_mode: default='optional'，means no same padding

	Returns:
		- ： output with padding
		out_pad_size: real pad size
	"""

	if condition == 'define':
		# Default: pad according to setting
		if padding_mode == 'optional':
			pad_width = ((0, 0), (0, 0), (length[0], length[0]), (length[1], length[1]))

		# use same conv, compute padding length
		elif padding_mode == 'same':
			s_h, s_w = kwargs['stride'][0], kwargs['stride'][1]
			k_h, k_w = kwargs['kernel_size'][0], kwargs['kernel_size'][1]
			n_h, n_w = x.shape[2], x.shape[3]
			p_h = (n_h * s_h - s_h + k_h - n_h) / 2
			p_w = (n_w * s_w - s_w + k_w - n_w) / 2

			if not (np.floor(p_h) == p_h) or not (np.floor(p_w) == p_w):
				raise ValueError(
					f'can not implement same convolution with current kernel{(k_h, k_w)}, stride{(s_h, s_w)}, and input shape')
			p_h, p_w = int(p_h), int(p_w)
			pad_width = ((0, 0), (0, 0), (p_h, p_h), (p_w, p_w))
		out_pad_size = (pad_width[2][1], pad_width[3][1], 1)
	elif condition == 'kernel_overflow':
		pad_width = ((0, 0), (0, 0), (0, length[0]), (0, length[1]))
		out_pad_size = (pad_width[2][1], pad_width[3][1], 0)

	return np.lib.pad(x, pad_width=pad_width, mode='constant'), out_pad_size
