import time


def timer(func):
	r"""
	A function timer
	"""

	def wrapper(*args, **kwargs):
		start_time = time.time()
		result = func(*args, **kwargs)
		print(f'Finishing {func.__name__} took time {time.time() - start_time}')
		return result

	return wrapper