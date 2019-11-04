import unittest
import numpy
import sys
sys.path.append('..')
from common.util import GPU, np, to_cpu

class TestCommonUtil(unittest.TestCase):
    """test class for common/util.py
    """

    def test_np_mode(self):
        if GPU:
            self.assertEqual(np.__name__, 'cupy')
        else:
            self.assertEqual(np.__name__, 'numpy')


    def test_to_cpu(self):
        x = np.array([1,2,3])
        self.assertEqual(type(to_cpu(x)), numpy.ndarray)


if __name__ == '__main__':
    unittest.main(verbosity=2)