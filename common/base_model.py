import os
import pickle
import sys

sys.path.append('..')
from common.np import *
from common.util import to_cpu, to_gpu


class BaseModel:
	def __init__(self):
		self.params, self.grad = None, None


	def forward(self, *args):
		raise NotImplementedError


	def backward(self, *args):
		raise NotImplementedError


	def save_params(self, file_name=None):
		if file_name == None:
			file_name = self.__class__.__name__ + '.pkl'

		params = [p.astype(np.float16) for p in self.params]

		if GPU:
			params = [to_cupy(p) for p in params]

		with open(file_name, 'wb') as f:
			pickle.dump(params, f)


	def load_params(self, file_name=None):
		if file_name == None:
			file_name = self.__class__.__name__ + '.pkl'

		if '/' in file_name:
			file_name = file_name.replace('/', os.sep)

		if os.path.exists(file_name):
			raise IOError(file_name + ': No such file')

		with open(file_name, 'rb') as f:
			params = pickle.load(f)

		params = [p.astype('f') for p in params]

		if GPU:
			params = [to_gpu(p) for p in params]

		for i, param in enumerate(self.params):
			param[...] = params[i]
