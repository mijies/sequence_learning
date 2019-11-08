import unittest
import sys
sys.path.append('..')
from layers.time_softmax_with_loss import np, TimeSoftmaxWithLoss

class TestTimeSoftmaxWithLoss(unittest.TestCase):
    """test class for TimeSoftmaxWithLoss in layers/time_softmax_with_loss.py
    """

    def test_forward_backward(self):

        N, T, V = 3, 2, 6
        xs = np.array([[[1.]*V]*T]*N) # (N, T, V)
        ts = np.arange(N*T) -1  # (N, T)

        layer = TimeSoftmaxWithLoss()
        loss  = layer.forward(xs, ts)

        ts, ys, mask, (N, T, V) = layer.cache
        self.assertTrue(all(mask==(ts!=-1)))

        # if one-hot-vector
        ts = np.zeros_like(xs)
        for i in range(N):
            for j in range(T):
                ts[i][j][(i+j)*8%6] = 1

        layer = TimeSoftmaxWithLoss()
        loss  = layer.forward(xs, ts)

        ts, ys, mask, (N, T, V) = layer.cache
        self.assertTrue(all(mask==(ts!=-1)))

        dout = 1
        dx = layer.backward(dout)
        self.assertEqual(dx.shape, (N, T, V))


if __name__ == '__main__':
    unittest.main(verbosity=2)