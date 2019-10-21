from common.util import np

# DONE
class RNN:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads  = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache  = None
    

    def forward(self, x, h_prev):
        Wx, Wh, b = self.params
        t = np.dot(x, Wx) + np.dot(h_prev, Wh) + b
        h_next = np.tanh(t)

        self.cache = (x, h_prev, h_next)
        return h_next


    def backward(self, dh_next):
        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache

        dt  = dh_next * (1 - h_next ** 2) # tanh'
        db  = np.sum(dt, axis=0)
        dWh = np.dot(h_prev.T, dt)
        dWx = np.dot(x.T, dt)
        dh_prev = np.dot(dt, Wh.T)
        x   = np.dot(dt, Wx.T)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx, dh_prev