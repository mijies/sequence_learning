import unittest
import numpy as np
import sys
sys.path.append('..')
from common.functions import sigmoid, relu, softmax, cross_entropy_error

class TestCommonFunctions(unittest.TestCase):
    """test class for common/functions.py
    """

    def test_sigmoid(self):
        x = np.array([1, 2])
        x = sigmoid(x)
        y = np.array([1, 2])
        y = 1 / (1 + np.exp(-y))
        self.assertTrue(all(x==y))


    def test_relu(self):
        x = np.array([1, -1])
        x = relu(x)
        y = np.array([1, 0])
        self.assertTrue(all(x==y))


    def test_softmax(self):
        x = np.array([0, 0])
        x = softmax(x)
        y = np.array([0.5, 0.5])
        self.assertTrue(all(x==y))
        self.assertEqual(np.sum(x), 1.)

        # test ndim == 2
        x = np.array([[1., 1.], [1., 1.]])
        x = softmax(x)
        y = np.array([[0.5, 0.5], [0.5, 0.5]])
        for x, y in zip(x, y):
            self.assertTrue(all(x==y))
            self.assertEqual(sum(x), 1.)


    def test_cross_entropy_error(self):
        y = np.array([[0., 1.5, 1.],[1., 1.5, 0.]])
        t = np.array([[2, 0]])
        loss = cross_entropy_error(y, t)
        expected = np.array([1., 1.])+1e-7
        expected = -np.sum(np.log(expected)) / 2
        self.assertEqual(loss, expected)

        # if flattend
        y = np.array([0., 1.5, 1.])
        t = np.array([2])
        loss = cross_entropy_error(y, t)
        self.assertEqual(loss, expected)

        # if t is one-hot-vector
        y = np.array([[0., 1.5, 1.],[1., 1.5, 0.]])
        t = np.array([[0, 0, 1], [1, 0, 0]])
        loss = cross_entropy_error(y, t)
        self.assertEqual(loss, expected)


if __name__ == '__main__':
    unittest.main(verbosity=2)