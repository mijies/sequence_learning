import sys
sys.path.append('..')
from common.time_layers import *
from models.base_model import BaseModel

# DONE
class Seq2seq(BaseModel):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        self.encoder = Encoder(V, D, H)
        self.decoder = Dncoder(V, D, H)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads


    def forward(self, xs):
        docoder_xs, decoder_ts = xs[:, :-1], xs[:, 1:]
        hs   = self.encoder.forward(xs)
        xs   = self.decoder.forward(xs, hs)
        loss = self.softmax.forward(xs, decoder_ts)
        return loss
    

    def backward(self, dout=1):
        dx = self.softmax.backward(dout)
        dh = self.decoder.backward(dx)
        _  = self.encoder.backward(dh)
        return None # Encoder.backward() returns None


    def generate(self, xs, start_id, sample_size):
        hs = self.encoder.forward(xs)
        samples = self.decoder.generate(hs, start_id, sample_size)
        return samples
    

    def evaluate(self, inputs, correct, id_to_char, verbose=False, is_reverse=False)
        correct  = correct.flatten()
        start_id = correct[0] # separator
        correct  = correct[1:]

        samples = self.generate(inputs, start_id, len(correct))

        # vec2words
        inputs  = ''.join([id_to_char[int(char)] for char in inputs.flatten()])
        correct = ''.join([id_to_char[int(char)] for char in correct])
        samples = ''.join([id_to_char[int(char)] for char in samples])

        if verbose:
            if is_reverse:
                inputs = inputs[::-1]
            
            print('Input :', inputs)
            print('Output:', samples)
            print('Answer:', correct)
            print('-'*10)

        return 1 if samples == correct else 0


# DONE
class Encoder():
    def __init__(self, , vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f') # all weights for LSTM
        lstm_Wh = (rn(D, 4 * H) / np.sqrt(H)).astype('f') # all weights for LSTM
        lstm_b  = np.zeros(4 * H).astype('f')              # all weights for LSTM

        self.embed = TimeEmbedding(embed_W)
        self.lstm  = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=False)

        # pack all the params and grads in
        self.params = self.embed.params + self.lstm.params
        self.grads  = self.embed.grads + self.lstm.grads
        self.hs = None


    def forward(self, xs):
        xs = self.embed.forward(xs)
        hs = self.lstm.forward(xs)
        self.hs = hs
        return hs[:, -1, :] # returns only the last hidden state in time sequence
    

    def backward(self, dh):
        dhs = np.zeros_like(self.hs)
        dhs[:, -1, :] = dh
        dx = self.lstm.backward(dhs)
        _ = self.embed.backward(dx)
        return None # TimeEmbedding.backward() returns None


# DONE
class Decoder():
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

		# np.sqrt(D) is Xavier initialization
        embed_W  = (rn(V, D) / 100).astype('f')
        lstm_Wx  = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh  = (rn(D, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b   = np.zeros(4 * H).astype('f')
        affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')

		self.layers = [
			TimeEmbedding(embed_W),
			TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True),
			TimeAffine(affine_W, affine_b)
		]
        self.lstm_layer = self.layers[1] # used in backward()

		# pack all the params and grads in
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    
    def forward(self, xs, h):
        self.set_state(h)
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    
    def backward(self, dx):
        for layer in reversed(self.layers):
            dx = layer.backward(dx) # TimeEmbedding.backward() returns None
        return self.lstm_layer.dh


    def generate(self, h, start_id, sample_size):
        samples = []
        sample_id = start_id
        self.lstm_layer.set_state(h)

        for _ in range(sample_size):
            xs = np.array(sample_id).reshape((1, 1))
            for layer in self.layers:
                xs = layer.forward(xs)
            
            sample_id = np.argmax(xs.flatten())
            samples.append(sample_id)
        
        return samples
