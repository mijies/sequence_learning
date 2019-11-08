import unittest
import sys
sys.path.append('..')
from layers.time_affine import np, TimeAffine

class TestTimeAffine(unittest.TestCase):
    """test class for TimeAffine in layers/time_affine.py
    """

    def test_forward_backward(self):

        H, V = 3, 12
        W = np.arange(H*V).reshape(H, V).astype('f') # (rn(H, V) / np.sqrt(H))
        b = np.zeros(V).astype('f') # np.zeros(V)

        N, T = 2, 1
        hs = np.array([[[1.]*H]*T]*N) # np.empty((N, T, H),dtype='f')

        layer = TimeAffine(W, b)
        out = layer.forward(hs)
        self.assertEqual(out.shape, (N, T, V))

        expected = np.dot(hs, W) + b
        for x, y in zip(out, expected):
            for x, y in zip(x, y):
                self.assertTrue(all(x==y))


        dout = np.array([[[1.]*V]*T]*N) # (N, T, V)
        dhs = layer.backward(dout)

        expected = np.dot(dout, W.T)
        for x, y in zip(dhs, expected):
            for x, y in zip(x, y):
                self.assertTrue(all(x==y))
        
        expected = dout.sum(axis=0)[0]
        db = layer.grads[1]
        self.assertTrue(all(db==expected))

        rxs = hs.reshape(N * T, -1)
        expected = np.dot(rxs.T, dout.reshape(N * T, -1))
        dW = layer.grads[0]
        for x, y in zip(dW, expected):
            self.assertTrue(all(x==y))



if __name__ == '__main__':
    unittest.main(verbosity=2)