from common.functions import sofmax
from common.layers import *
from layers.rnn import RNN

# DONE
class TimeLSTM:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads  = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None

        self.h, self.c = None, None
        self.dh = None
        self.stateful = stateful

    
    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D   = xs.shape
        H = Wh.shape[0]

        hs = np.empty((N, T, H),dtype='f')
        self.layers = []

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')
        if not self.stateful or self.c is None:
            self.c = np.zeros((N, H), dtype='f')

        for t in range(T):
            layer = LSTM(*self.params)
            self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)
            hs[:, t, :] = self.h

            self.layers.append(layer)
        
        # hidden state to next layer, cell just used in time sequence
        return hs


    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H   = dhs.shape
        D = Wx.shape[0]

        dx = np.empty((N, T, D), dtype='f')
        dh, dc = 0, 0
        grads  = [0, 0, 0]

        for t in reversed(range(T)):
            layer = self.layers[t]
            # combine dhs(from next layer) and dh(from next time sequence)
            dx, dh, dc = layer.backward(dhs[:, t, :] + dh, dc)
            dx[:, t, :] = dx

            for i, grad in enumerate(layer.grads):
                self.grads[i][...] = grad

        self.dh = dh # seq2seq's backward takes this out
        return dx   # passed to previous layer



    def set_state(self, h, c=None): # used in seq2seq
        self.h, self.c = h, c


    def reset_state(self): # called in models for training use
        self.h, self.c = None, None


# DONE
class TimeEmbedding:
    def __init__(self, W):
        self.params = [W]
        self.grads  = [np.zeros_like(W)]
        self.layers = None
        self.W      = W


    def forward(self, xs):
        N, T = xs.shape
        V, D = self.W.shape
    
        out = np.empty((N, T, D), dtype='f')
        self.layers = []

        for t in range(T):
            layer = Embedding(self.W)
            out[:, T, :] = layer.forward(xs[:, T])

        return out


    def backward(self, dout):
        N, T, D = dout.shape

        grad = 0
        for t in range(T):
            layer = self.layers[t]
            layer.backward(dout[:, T, :])
            grad += layer.grads[0]

        # self.grads points to Encoder.grads like below
        # self.grads = self.embed.grads + self.lstm.grads
        self.grads[0][...] = grad
        return None


# DONE
class TimeAffine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.xs = None

    
    def forward(self, xs):
        W, b = self.params
        N, T, D = xs.shape
        self.xs = xs

        rxs = xs.reshape(N * T, -1) # -1 denotes the rest of the dimensions
        out = np.dot(rxs, W) + b

        return out.reshape(N, T, -1)


    def backward(self, dout):
        W, b = self.params
        xs = self.xs
        N, T, D = xs.shape

        dout = dout.reshape(N * T, -1)
        rxs = xs.reshape(N * T, -1)

        db = dout.sum(axis=0)
        dW = np.dot(rxs.T, dout)
        dx = np.dot(dout, W.T)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx.reshape(N, T, -1)


# DONE
class TimeSoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        self.mask_label = -1


    def forward(self, xs, ts):
        N, T, V = xs.shape

        if ts.ndim == 2: # if one hot vector
            ts = ts.argmax(axis=2)

        xs = xs.reshape(N * T, V)
        ts = ts.reshape(N * T)

        ys = sofmax(xs)
        ls = np.log(ys[np.arange(N * T), ts]) # takes out only the indices by ts

        mask = (ts != self.mask_label)
        ls *= mask
        loss = -np.sum(ls) / mask.sum()

        self.cache = (ts, ys, mask, (N, T, V))
        return loss
    

    def backward(self, dout=1):
        ts, ys, mask, (N, T, V) = self.cache

        dx = ys[np.arange(N * T), ts] # takes out only the indices by ts

        # partial derivative ts(ys - 1) * dLoss
        dx -= 1 
        dx *= dout
        dx /= mask.sum() # divided by the number of non-padding words

        dx *= mask[:, np.newaxis] # zeros out the gradients of padding words
        return dx.reshape(N, T, V)
