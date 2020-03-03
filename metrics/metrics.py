from collections import Counter
import numpy as np


"""
Basic criterion functions

Functoins:
	outside:
		accuracy_score: calculate accuracy, multiclass is supported
		precision_score: calculate precision, multiclass is supported
		recall_score: calculate recall, multiclass is supported
		f1: calculate f1 score, multiclass is supported
		fbeta: calculate fbeta score, multiclass is supported
		confusion_matrix: calculate confusion matrix, multiclass is supported

	inside:
		_helper: helper function, calculate parameters for binary classification
		_binary_precision: precision for binary classification
		_binary_recall: recall for binary classification
		_binary_confusion_mat: confusion matrix for binary classification
		_average_checker: check legitimacy
"""


def accuracy_score(y_true, y_pred, normalize=True):
	"""
	Accuracy: the percentage of corrected labels in prediction

	Args:
		y_true: true label
		y_pred:	prediction label
		normalize: default=True, return percentage
	"""

	y_true = y_true.reshape((1, -1))
	y_pred = y_pred.reshape((1, -1))
	assert (y_true.shape == y_pred.shape), 'dimension do not match, make sure dimension matched'
	m = y_true.shape[1]
	temp = y_true - y_pred
	return ((m - np.count_nonzero(temp)) / m) if normalize else (m - np.count_nonzero(temp))


def _helper(y_true, y_pred, func):
	"""
	Calculate something

	Args:
		y_true: true label
		y_pred:	prediction label
		func:
	"""

	y_t = y_true.reshape((1, -1))
	y_p = y_pred.reshape((1, -1))
	tmp = y_t - y_p

	tp = np.count_nonzero(np.multiply(y_t, y_p))
	tn = np.where((y_t + y_p) == 0)[0].shape[0]

	if func == 'precision':
		tmp2 = np.copy(tmp)
		fp = np.where(tmp2 < 0)[0].shape[0]
		return tp, fp

	if func == 'recall':
		tmp1 = np.copy(tmp)
		tmp1 = np.where(tmp1 == 1, 1, 0)
		fn = np.count_nonzero(tmp1)
		return tp, fn

	if func == 'convert':
		# 将n分类转化成n个二分类
		# convert n-classification into n binary classifications
		class_dict = {}
		for i in list(set(y_true)):
			class_dict[i] = {'y_t': np.where(y_true == i, 1, 0), 'y_p': np.where(y_pred == i, 1, 0)}
		return class_dict

	if func == 'confusion':
		tmp2 = np.copy(tmp)
		fp = np.where(tmp2 < 0)[0].shape[0]
		tmp1 = np.copy(tmp)
		tmp1 = np.where(tmp1 == 1, 1, 0)
		fn = np.count_nonzero(tmp1)
		return tp, fn, fp, tn


def _average_checker(average):
	if not (not average) and not (average == 'macro') and not (average == 'micro'):
		raise TypeError('value of average {} is not supported'.format(average))


def _binary_precision(y_true, y_pred):
	"""
	Args:
		y_true: true label
		y_pred:	prediction label

	P = TP / (TP+FP)
	"""

	tp, fp = _helper(y_true, y_pred, func='precision')
	try:
		p = tp / (tp + fp)
		return p
	except ZeroDivisionError:
		return 0.0


def _binary_recall(y_true, y_pred):
	"""
	Args:
		y_true: true label
		y_pred:	prediction label


	R = TP / (TP+FN)
	"""

	tp, fn = _helper(y_true, y_pred, func='recall')
	try:
		r = tp / (tp + fn)
		return r
	except ZeroDivisionError:
		return 0.0


def _binary_confusion_mat(y_true, y_pred):
	"""
	Confusion matrix for binary classification(0 and 1)
	"""

	tp, fn, fp, tn = _helper(y_true, y_pred, 'confusion')
	return np.array([[tn, fp], [fn, tp]])


def precision_score(y_true, y_pred, average='macro'):
	"""
	Calculate precision

	Args:
		y_true: true label
		y_pred:	prediction label
		average: macro or micro
	"""

	n_class = np.max(y_true) + 1

	if n_class == 2:
		return _binary_precision(y_true, y_pred)
	if n_class > 2:
		marco_precision = []
		class_dict = _helper(y_true, y_pred, func='convert')
		for i in range(n_class):
			marco_precision.append(_binary_precision(class_dict[i]['y_t'], class_dict[i]['y_p']))
		if not average:
			print(marco_precision)
			return np.array(marco_precision)
		elif average == 'macro':
			p = np.sum(marco_precision) / n_class
			return p
		elif average == 'micro':
			bi_confusion_mat = np.zeros((2, 2))
			for cls in range(n_class):
				bi_confusion_mat += _binary_confusion_mat(class_dict[cls]['y_t'], class_dict[cls]['y_p'])
			bi_confusion_mat /= n_class
			tp = bi_confusion_mat[1, 1]
			fp = bi_confusion_mat[0, 1]
			p = tp / (tp + fp)
			return p


def recall_score(y_true, y_pred, average='macro'):
	"""
	Calculate recall

	Args:
		y_true: true label
		y_pred:	prediction label
		average: macro or micro
	"""

	_average_checker(average)
	# May be is not the best option to know n_class
	n_class = np.max(y_true) + 1
	if n_class == 2:
		return _binary_recall(y_true, y_pred)
	if n_class > 2:
		marco_recall = []
		class_dict = _helper(y_true, y_pred, func='convert')
		for i in range(n_class):
			marco_recall.append(_binary_recall(class_dict[i]['y_t'], class_dict[i]['y_p']))
		if not average:
			return np.array(marco_recall)
		elif average == 'macro':
			r = np.sum(marco_recall) / n_class
			return r
		elif average == 'micro':
			bi_confusion_mat = np.zeros((2, 2))
			for cls in range(n_class):
				bi_confusion_mat += _binary_confusion_mat(class_dict[cls]['y_t'], class_dict[cls]['y_p'])
			bi_confusion_mat /= n_class
			tp = bi_confusion_mat[1, 1]
			fn = bi_confusion_mat[1, 0]
			r = tp / (tp + fn)
			return r


def f1(y_true, y_pred, average='macro'):

	_average_checker(average)
	n_class = np.max(y_true) + 1
	p = precision_score(y_true, y_pred, None)
	r = recall_score(y_true, y_pred, None)
	if n_class == 2:
		if p + r:
			return (2 * p * r) / (p + r)
		else:
			return 0
	f1 = []
	for i, j in zip(p, r):
		if i + j:
			f_1 = (2 * i * j) / (i + j)
			f1.append(float('%.8f' % f_1))
		else:
			f1.append(0)
	if not average:
		return f1
	else:
		f1_score = np.sum(f1) / n_class
		return f1_score


def fbeta(y_true, y_pred, beta, average='macro'):

	_average_checker(average)
	n_class = np.max(y_true) + 1
	if beta < 0:
		raise ValueError('Invalid value for beta: {}'.format(beta))
	p = precision_score(y_true, y_pred, None)
	r = recall_score(y_true, y_pred, None)

	if n_class == 2:
		if (beta ** 2) * p + r:
			return ((1 + beta ** 2) * p * r) / ((beta ** 2) * p + r)
		else:
			return 0

	fbeta = []
	for i, j in zip(p, r):
		if (beta ** 2) * i + j:
			f_beta = ((1 + (beta ** 2)) * i * j) / ((beta ** 2) * i + j)
			fbeta.append(float('%.8f' % f_beta))
		else:
			fbeta.append(0)
	if not average:
		return fbeta
	else:
		fbeta_score = np.sum(fbeta) / n_class
		return fbeta_score


def confusion_matrix(y_true, y_pred):

	n_class = np.max(y_true) + 1

	if n_class == 2:
		return _binary_confusion_mat(y_true, y_pred)
	if n_class > 2:
		marco_confusion = []
		class_dict = _helper(y_true, y_pred, func='convert')
		confusion_mat = np.zeros((n_class, n_class))
		for cls in range(n_class):
			index = np.where(y_true == cls)
			tmp = y_pred[index]
			cls_counter = Counter(tmp)
			for key, value in cls_counter.items():
				confusion_mat[cls][key] += value

	return confusion_mat
