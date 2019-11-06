import unittest
import sys
sys.path.append('..')
from layers.time_lstm import np, TimeLSTM

class TestTimeLSTM(unittest.TestCase):
    """test class for TimeLSTM in layers/time_lstm.py
    """

    def test_forward_backward(self):

        D, H = 4, 3
        Wx = np.arange(D*H*4).reshape(D, 4*H).astype('f') # (D, 4 * H)
        Wh = np.arange(H*H*4).reshape(H, 4*H).astype('f') # (H, 4 * H)
        b  = np.zeros(4 * H).astype('f')

        N, T, V = 2, 3, 4
        xs = np.array([1.]*24).reshape(N, T, V) # (N, T, V)

        layer = TimeLSTM(Wx, Wh, b, stateful=False)
        hs = layer.forward(xs)
        self.assertEqual(hs.shape, (N, T, H)) # hs = np.empty((N, T, H),dtype='f')
        self.assertEqual(len(layer.layers), T)

        self.assertEqual(layer.h.shape, (N, H))
        self.assertEqual(layer.c.shape, (N, H))
        
        dhs = np.ones_like(hs)
        dxs = layer.backward(dhs)
        self.assertEqual(dxs.shape, (N, T, V )) # np.empty((N, T, D), dtype='f')

        # TODO: more tests
        # print(layer.grads)


if __name__ == '__main__':
    unittest.main(verbosity=2)