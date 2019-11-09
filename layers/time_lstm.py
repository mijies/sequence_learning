from layers.lstm import np, LSTM

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
        N, T, D   = xs.shape # if peeky, D is (H + D)
        H = Wh.shape[0]

        hs = np.empty((N, T, H),dtype='f')
        self.layers = []

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')
        if not self.stateful or self.c is None:
            self.c = np.zeros((N, H), dtype='f')

        for t in range(T):
            layer = LSTM(Wx, Wh, b)
            self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)
            hs[:, t, :] = self.h

            self.layers.append(layer)
        
        return hs # hidden state to next layer, cell just used in time sequence


    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H   = dhs.shape
        D = Wx.shape[0]

        dxs = np.empty((N, T, D), dtype='f')
        dh, dc = 0, 0
        grads  = [0, 0, 0]

        for t in reversed(range(T)):
            layer = self.layers[t]
            # combine dhs(from next layer) and dh(from next time sequence)
            dx, dh, dc = layer.backward(dhs[:, t, :] + dh, dc)
            dxs[:, t, :] = dx

            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad

        self.dh = dh # seq2seq's backward takes this out
        return dxs   # passed to previous layer


    def set_state(self, h, c=None): # used in seq2seq forward and generate
        self.h, self.c = h, c


    def reset_state(self): # called in models for training use
        self.h, self.c = None, None
