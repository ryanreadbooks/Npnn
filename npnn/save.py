import pickle
from .nn.sequential import Sequential


def save_params(model, path):
	"""
	Save model

	Args:
		model: model
		path: saved path
	"""
	
	if not isinstance(model, Sequential):
		raise TypeError('only model of type Sequential can be saved.')

	file_name = path
	# params
	model_params = model.state_dict()
	# structure
	model_structure = [layer._return_self() for layer in model._structure]
	# description
	model_str_description = str(model).splitlines()
	# all info needed
	to_be_saved = {'structure':model_structure, 'params':model_params, 'model_str':model_str_description}
	with open(file_name, 'wb') as f:
		pickle.dump(to_be_saved, f)

	print(f'Model has been saved in path {path}.')
