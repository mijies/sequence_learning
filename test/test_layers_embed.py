import unittest
import sys
sys.path.append('..')
from layers.embed import np, Embedding

class TestLayersEmbed(unittest.TestCase):
    """test class for layers/embed.py
    """

    def test_forward_backward(self):
        W = np.arange(5) * 10
        embed = Embedding(W)

        idx = np.array([3, 1, 3])
        out = embed.forward(idx)
        self.assertTrue(all(idx*10==out))

        dout = out * 2
        dout = embed.backward(dout)
        self.assertEqual(dout, None)

        expected = np.zeros_like(W)
        np.add.at(expected, idx, out*2)
        self.assertTrue(all(expected==embed.grads[0]))


if __name__ == '__main__':
    unittest.main(verbosity=2)