import unittest
import copy
import sys
sys.path.append('..')
from common.trainer import np, Trainer, concat_duplicate, clip_grads

class TestCommonTrainer(unittest.TestCase):
    """test class for common/trainer.py
    """

    def test_concat_duplicate(self):
        self.params = np.

    def test_clip_grads(self):
        max_norm = 2.
        grads = np.array([1., np.sqrt(3)]) # pythagorean theorem
        self.grads = copy.deepcopy(grads)

        clip_grads(grads, max_norm)
        self.assertTrue(all(grads < self.grads)) # because 1e-7 is added

        grads = np.array([1., 1.5]) # pythagorean theorem
        self.grads = copy.deepcopy(grads)
        self.assertTrue(all(grads==self.grads)) # not clipped


if __name__ == '__main__':
    unittest.main(verbosity=2)