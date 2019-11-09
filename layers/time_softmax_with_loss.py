from common.functions import np, softmax


class TimeSoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        self.mask_label = -1


    def forward(self, xs, ts):
        N, T, V = xs.shape
        if ts.ndim == 3: # if one hot vector
            ts = ts.argmax(axis=2)

        xs = xs.reshape(N * T, V)
        ts = ts.reshape(N * T)

        ys = softmax(xs)   
        ls = np.log(ys[np.arange(N * T), ts]) # takes out only the indices by ts

        mask = (ts != self.mask_label) # paddings "-1" get False
        ls *= mask
        loss = -np.sum(ls) / mask.sum() # divided by the number of non-padding words

        self.cache = (ts, ys, mask, (N, T, V))
        return loss
    

    def backward(self, dout):
        ts, ys, mask, (N, T, V) = self.cache
        dx = ys

        # partial derivative ts(ys - 1) * dLoss
        dx[np.arange(N * T), ts] -= 1 # only the indixed by ts
        dx *= dout
        dx /= mask.sum() # divided by the number of non-padding words

        dx *= mask[:, np.newaxis] # zeros out the gradients of padding words
        return dx.reshape((N, T, V))
