import pickle
import re
import npnn


def load_params(path):
	r"""
	Load model
	"""

	with open(path, 'rb') as f:
		model = pickle.load(f)
	model_structure = model['structure']
	model_params = model['params']
	model_str = model['model_str']
	return _load_layer_and_params(model_structure, model_params, model_str)


def _load_layer_and_params(structure, params, str_description):
	r"""
	Load model
	"""

	sequential_args = []
	pattern = re.compile('[0-9]+')
	for layer_dict, layer_str_desp in zip(structure, str_description):
		args = list(layer_dict.values())
		n_layer = re.search(pattern, layer_str_desp).group()

		if 'Linear' in layer_dict:
			layer = npnn.basic.Linear(*args[1:])
			layer.w = params[f'{n_layer}.weight']
			layer.b = params[f'{n_layer}.bias']
		elif 'BatchNorm' in layer_dict:
			layer = npnn.basic.BatchNorm(*args[1:])
			layer.w = params[f'{n_layer}.weight']
			layer.b = params[f'{n_layer}.bias']
			layer.running_mean = params[f'{n_layer}.running_mean']
			layer.running_var = params[f'{n_layer}.running_var']
		elif 'Dropout' in layer_dict:
			layer = npnn.basic.Dropout(*args[1:])
		elif 'Conv2d' in layer_dict:
			layer = npnn.basic.Conv2d(*args[1:])
			layer.w = params[f'{n_layer}.weight']
			layer.b = params[f'{n_layer}.bias']
			out_channels = layer_dict['out_channels']
			kernel_size = layer_dict['kernel_size']
			in_channels = layer_dict['in_channels']
			layer.w.resize((out_channels, kernel_size[0] * kernel_size[1] * in_channels))
		elif 'MaxPool2d' in layer_dict:
			layer = npnn.basic.MaxPool2d(*args[1:])
		elif 'AvgPool2d' in layer_dict:
			layer = npnn.basic.AvgPool2d(*args[1:])
		elif 'ConstantPad2d' in layer_dict:
			layer = npnn.basic.ConstantPad2d(*args[1:])
		elif 'EdgePad2d' in layer_dict:
			layer = npnn.basic.EdgePad2d(*args[1:])
		elif 'ReflectionPad2d' in layer_dict:
			layer = npnn.basic.ReflectionPad2d(*args[1:])
		elif 'Flatten' in layer_dict:
			layer = npnn.basic.Flatten()
		elif 'ReLU' in layer_dict:
			layer = npnn.basic.ReLU()
		elif 'Sigmoid' in layer_dict:
			layer = npnn.basic.Sigmoid()
		elif 'Tanh' in layer_dict:
			layer = npnn.basic.Tanh()
		elif 'Softmax' in layer_dict:
			layer = npnn.basic.Softmax()
		elif 'ELU' in layer_dict:
			layer = npnn.basic.ELU()
		elif 'LeakyReLU' in layer_dict:
			layer = npnn.basic.LeakyReLU(*args[1:])
		elif 'GeLU' in layer_dict:
			layer = npnn.basic.GeLU()
		sequential_args.append(layer)
	return npnn.nn.Sequential(*sequential_args)
