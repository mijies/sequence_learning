from common.util import np

class Embedding:
    def __init__(self, W):
		self.params = [W]
		self.grads  = [np.zeros_like(W)]
		self.idx    = None

	def forward(self, idx):
		W, = self.params
		self.idx = idx
		return W[idx]
		
	def backward(self, dout):
		dW, = self.grads
		dW[...] = 0
		np.add.at(dW, self.idx, dout)
		return None