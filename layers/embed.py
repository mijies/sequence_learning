from common.util import np, GPU

class Embedding: # replace Matmul for efficiency
	def __init__(self, W):
		self.params = [W]
		self.grads  = [np.zeros_like(W)]
		self.idx    = None


	def forward(self, idx):
		W, = self.params
		self.idx = idx
		return W[idx] # takes out only the weights for the input words
		
		
	def backward(self, dout):
		dW, = self.grads
		dW[...] = 0
		# if GPU:
		# 	np.scatter_add(dW, self.idx, dout)
		# else:
		# 	np.add.at(dW, self.idx, dout)
		np.add.at(dW, self.idx, dout)
		return None