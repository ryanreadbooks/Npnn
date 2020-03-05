from os.path import dirname
import pickle
from PIL import Image
import numpy as np


def show_cifar10(options, img2show=None):
	r"""
	Show picture in cifar10 randomly

	Args:
		options: 'random'(randomly pick a picture) or 'explicit'(numpy array and show)
		img2show: image to be shown
	"""

	assert options in ['random', 'explicit'], f'option {options} is not supported'

	if options == 'random':
		# pick a picture randomly
		batch = np.random.randint(1, 6)
		index = np.random.randint(10001)
		# file location
		filepath = dirname(__file__) + f'\\data\\cifar10\\data_batch_{batch}'
		metaflie = dirname(__file__) + '\\data\\cifar10\\batches.meta'

		with open(filepath, 'rb') as bin_file:
			# data of 3 channels
			data = pickle.load(bin_file, encoding='bytes')
			# convert to Image format
			img = data[b'data'][index, :].reshape(3, 32, 32).transpose(1,2,0)
			# label of the pic :0~9
			target = data[b'labels'][index]
			explicit_target = str(data[b'filenames'][index], 'utf-8')

		del data
		with open(metaflie, 'rb') as meta_file:
			metadata = pickle.load(meta_file, encoding='bytes')
			# find category of the picture
			category = str(metadata[b'label_names'][target], 'utf-8')
			
		print(f'Categiry: {category}')
		print(f'File name: {explicit_target}')
	else:
		img = img2show.astype(np.uint8)

	# show pic
	Image.fromarray(img).convert('RGB').show()


if __name__ == '__main__':
	show_cifar10('random', None);