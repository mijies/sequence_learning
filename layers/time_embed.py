from layers.embed import np, Embedding

class TimeEmbedding:
    def __init__(self, W):
        self.params = [W] # into a list because concatenated in Encoder and Decoder
        self.grads  = [np.zeros_like(W)] # into a list as it will be concatenated
        self.layers = None
        self.W      = W


    def forward(self, xs):
        N, T = xs.shape
        V, D = self.W.shape
    
        out = np.empty((N, T, D), dtype='f')
        self.layers = []

        for t in range(T):
            layer = Embedding(self.W)
            out[:, t, :] = layer.forward(xs[:, t])
            self.layers.append(layer)

        return out


    def backward(self, dout):
        N, T, D = dout.shape

        grad = 0
        for t in range(T):
            layer = self.layers[t]
            layer.backward(dout[:, t, :])
            grad += layer.grads[0]

        # self.grads points to Encoder.grads like below
        # self.grads = self.embed.grads + self.lstm.grads
        self.grads[0][...] = grad
        return None
