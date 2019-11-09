from common.functions import np, sigmoid

class LSTM:
    def __init__(self, Wx, Wh, b):
        # pack in the weights and biases for the following 4
        # 3 gates(forget, input, output)
        # tanh vector used with input gate to update cell
        self.params = [Wx, Wh, b]
        self.grads  = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache  = None
    

    def forward(self, x, h_prev, c_prev): # LSTM has cell
        Wx, Wh, b = self.params
        N, H      = h_prev.shape

        ALL = np.dot(x, Wx) + np.dot(h_prev, Wh) + b # if peeky, x.shape (N, H + D)

        f  = ALL[:, :H]
        tv = ALL[:H, H:H*2] # used with input gate to update cell
        i  = ALL[:, H*2:H*3]
        o  = ALL[:, H*3:]

        f  = sigmoid(f)
        tv = np.tanh(tv)
        i  = sigmoid(i)
        o  = sigmoid(o)

        c_next = f * c_prev + tv * i
        h_next = o * np.tanh(c_next)

        self.cache = (x, h_prev, c_prev, i, f, tv, o, c_next)
        return h_next, c_next
    

    def backward(self, dh_next, dc_next):
        Wx, Wh, b = self.params
        x, h_prev, c_prev, i, f, tv, o, c_next = self.cache

        # h_next = o * np.tanh(c_next)
        tc_next  = np.tanh(c_next)
        do = dh_next * tc_next
        dc_next2 = (dh_next * o) * (1 - tc_next ** 2) # tanh' => 1 - y**2

        # c_next = f * c_prev + tv * i
        dc_next += dc_next2 # combine the split c_next computational graph
        dc_prev  = dc_next * f
        # df, dtv, di = [c_prev, i, tv] * dc_next
        #    => this syntax doesn't work with cupy
        df  = c_prev * dc_next
        dtv = i  * dc_next
        di  = tv * dc_next

        # activation functions
        df  *= f * (1 - f)
        di  *= i * (1 - i)
        do  *= o * (1 - o)
        dtv *= (1 - tv ** 2)

        dALL = np.hstack((df, dtv, di, do)) # horizontally concatnate

        dWx = np.dot(x.T, dALL)
        dWh = np.dot(h_prev.T, dALL)
        db  = dALL.sum(axis=0)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db
        
        dx = np.dot(dALL, Wx.T)
        dh_prev = np.dot(dALL, Wh.T)

        return dx, dh_prev, dc_prev