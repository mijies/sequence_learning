from common.util import np

def sigmoid(x):
	return 1 / (1 + np.exp(-x))


def relu(x):
	return np.maximum(0, x)


def softmax(x):
	if x.ndim == 2: # for batch
		x = x - x.max(axis=1, keepdims=True)
		x = np.exp(x)
		x /= x.sum(axis=1, keepdims=True)

	elif x.ndim == 1:
		x = x - np.max(x) # scale down
		x = np.exp(x) / np.sum(np.exp(x))

	return x


def cross_entropy_error(y, t):
	if y.ndim == 1: # if flattend
		t = t.reshape(1, t.size)
		y = y.reshape(1, y.size)

	if t.size == y.size: # which means t is one-hot-vector
		t = t.argmax(axis=1) # t stores the index number of 1

	batch_size = y.shape[0]
	y = y[np.arange(batch_size), t] + 1e-7 # y only stores the values matched up with t's index
	return -np.sum(np.log(y)) / batch_size