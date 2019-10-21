import sys
sys.path.append('..')
from common.base_model import BaseModel
from common.time_layers import *

# DONE
class RnnLSTM(BaseModel):
	def __init__(self, vocab_size=10000, wordvec_size=100, hidden_size=100):
		V, D, H = vocab_size, wordvec_size, hidden_size
		rn = np.random

		# initialize weights and biases
		# np.sqrt(D) is Xavier initialization
		embed_W  = (rn(V, D) / 100).astype('f')
		lstm_Wx  = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
		lstm_Wh  = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
		lstm_b   = np.zeros(4 * H).astype('f')
		affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
		affine_b = np.zeros(V).astype('f')

		# setup layers
		self.layers = [
			TimeEmbedding(embed_W),
			TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True),
			TimeAffine(affine_W, affine_b)
		]
		self.lstm_layer = self.layers[1] # for state resetting
		self.loss_layer = TimeSoftmaxWithLoss()

		# pack all the params and grads in
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads


	def predict(self, xs): # called in prediction
		for layer in self.layers:
			xs = layer.forward(xs)
		return xs


	def forward(self, xs, ts): # called in training
		score = self.predict(xs)
		loss  = self.loss_layer.forward(score, ts)
		return loss


	def backward(self, dout=1):
		dx = self.loss_layer.backward(dout)
		for layer in reversed(self.layers):
			dx = layer.backward(dx)
		return None # TimeEmbedding.backward() returns None


	def reset_state(self):
		self.lstm_layer.reset_state()
