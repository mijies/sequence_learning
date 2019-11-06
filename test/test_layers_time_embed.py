import unittest
import sys
sys.path.append('..')
from layers.time_embed import np, TimeEmbedding

class TestTimeEmbedding(unittest.TestCase):
    """test class for TimeEmbedding in layers/time_embed.py
    """

    def test_forward_backward(self):

        V, D = 4, 4 # V, D = vocab_size, wordvec_size
        W = np.arange(V*D).reshape(V, D).astype('f') # W = np.array([[1]*D]*V)

        # N, T = idx.shape
        idx = np.array([
            [1, 3, 1] , [2, 0, 1] # shape is (2, 3)
        ])

        layer = TimeEmbedding(W)
        xs = layer.forward(idx)
        self.assertEqual(xs.shape, (2, 3, 4)) # N, T, V
        self.assertTrue(all(xs[0][0]==W[1]))
        self.assertTrue(all(xs[0][1]==W[3]))
        self.assertTrue(all(xs[1][2]==W[1]))

        for grad in layer.grads[0]:
            self.assertTrue(all(grad==0))

        dout = np.zeros_like(xs) + 10
        dout = layer.backward(dout)
        self.assertEqual(dout, None)

        self.assertTrue(all(layer.grads[0][1]==np.array([10.]*4)*3))
        self.assertTrue(all(layer.grads[0][0]==np.array([10.]*4)*1))


if __name__ == '__main__':
    unittest.main(verbosity=2)