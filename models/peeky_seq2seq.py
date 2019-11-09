from models.seq2seq import np, Seq2seq, Encoder
from layers.time_embed import TimeEmbedding
from layers.time_lstm  import TimeLSTM
from layers.time_affine import TimeAffine
from layers.time_softmax_with_loss import TimeSoftmaxWithLoss


class PeekySeq2seq(Seq2seq):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size

        self.encoder = Encoder(V, D, H)
        self.decoder = PeekyDecoder(V, D, H) # replaced with Decoder
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads  = self.encoder.grads  + self.decoder.grads



class PeekyDecoder(): # put encoder hs into all xs time sequences for LSTM and Affine
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn
        self.cache = None # keep H for backward()

        embed_W  = (rn(V, D) / 100).astype('f')
        lstm_Wx  = (rn(H + D, 4 * H) / np.sqrt(H + D)).astype('f') # np.concatenate((out, hs), axis=2)
        lstm_Wh  = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b   = np.zeros(4 * H).astype('f')

        affine_W = (rn(H + H, V) / np.sqrt(H + H)).astype('f') # np.concatenate((out, hs), axis=2)
        affine_b = np.zeros(V).astype('f')

        self.embed  = TimeEmbedding(embed_W)
        self.lstm   = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.affine = TimeAffine(affine_W, affine_b)

        self.params, self.grads = [], []
        for layer in (self.embed, self.lstm, self.affine):
            self.params += layer.params
            self.grads  += layer.grads


    
    def forward(self, xs, h):
        N, T = xs.shape
        N, H = h.shape # Encoder returns hs[:, -1, :] -> (N, H)
        self.cache = H

        self.lstm.set_state(h) # TimeLSTM.h.shape (N, H)

        out = self.embed.forward(xs) # out.shape (N, T, D)
        hs  = np.repeat(h, T, axis=0).reshape(N, T, H)
        out = np.concatenate((hs, out), axis=2) # (N, T, H + D)
        
        out = self.lstm.forward(out)
        out = np.concatenate((hs, out), axis=2) # (N, T, H + H)

        score = self.affine.forward(out)
        return score

    
    def backward(self, dscore):
        H = self.cache

        dx = self.affine.backward(dscore)
        dx, dh0 = dx[:, :, H:], dx[:, :, :H]

        dx = self.lstm.backward(dx)
        dx, dh1 = dx[:, :, H:], dx[:, :, :H]
        self.embed.backward(dx)

        dhs = dh0 + dh1
        return self.lstm.dh + np.sum(dhs, axis=1) # (np.repeat(h, T, axis=0))'


    def generate(self, hs, start_id, sample_size):
        samples = []
        sample_id = start_id
        self.lstm.set_state(hs)

        H  = hs[1] # (N, H)
        hs = hs.reshape(1, 1, H) # T is 1 as used in time loop
        for _ in range(sample_size):
            xs  = np.array(sample_id).reshape((1, 1))
            out = self.embed.forward(xs)

            out = np.concatenate((hs, out), axis=2)
            out = self.lstm.forward(out)

            out = np.concatenate((hs, out), axis=2)
            out = self.affine.forward(out)
            
            sample_id = np.argmax(out.flatten())
            samples.append(int(sample_id))
        
        return samples
