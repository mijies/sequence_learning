from common.util import np

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