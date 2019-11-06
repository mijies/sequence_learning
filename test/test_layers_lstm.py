import unittest
import sys
sys.path.append('..')
from layers.lstm import np, LSTM

class TestLayersLSTM(unittest.TestCase):
    """test class for layers/lstm.py
    """

    def test_forward_backward(self):
        # D, H = wordvec_size, hidden_size
        D, H = 3, 4 # H must be at least 4 because of set of 4 params
        N, T = 2, 2
        xs   = np.array([1.]*N*T*D).reshape(N, T, D)

        h = np.zeros((N, H), dtype='f')
        c = np.zeros((N, H), dtype='f')

        # np.dot(x, Wx) + np.dot(h_prev, Wh) + b
        Wx = np.array([[1.]*4*H]*D)
        Wh = np.array([[1.]*4*H]*H)
        b  = np.zeros(4*H).astype('f')

        self.layers = []
        for t in range(T):
            layer = LSTM(Wx, Wh, b)
            h, c = layer.forward(xs[:, t, :], h, c)
            (x, h_prev, c_prev, i, f, tv, o, c_next) = layer.cache

            self.assertTrue(all(f[0]==i[1]))
            self.assertTrue(all(f[1]==o[0]))

            # TODO: more specific tests
            # print(lstm.cache)
            # print(h)
            # print(c)

            self.layers.append(layer)

        tc_next = np.tanh(c_next)
        dh_next = np.ones_like(tc_next) # shape determined by :do = dh_next * tc_next
        dc_next = np.ones_like(f)       # shape determined by :dc_prev = dc_next * f

        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh, dc = layer.backward(dh_next, dc_next)
            # TODO: more specific tests
            # print(dx)
            # print(dh)
            # print(dc)

if __name__ == '__main__':
    unittest.main(verbosity=2)