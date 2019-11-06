import unittest
import sys
sys.path.append('..')
from common.optimizer import np, Adam

class TestCommonOptimizer(unittest.TestCase):
    """test class for common/optimizer.py
    """

    def test_update_fixed_grads(self):

        optimizer = Adam()
        self.params, params = np.array([[1.,-1.]]), np.array([[1.,-1.]])
        grads = np.array([[1.,1.]]) # grads fixed

        for _ in range(5):
            optimizer.update(params, grads)
            self.assertTrue(all(self.params[0] > params[0])) # simply reduced
            self.params[...] = params


    def test_update_descending_grads(self):

        optimizer = Adam()
        self.params, params = np.array([[1.,-1.]]), np.array([[1.,-1.]])
        grads = np.array([[1.,1.]])

        for _ in range(5):
            grads *= 0.9 # grads descend
            optimizer.update(params, grads)
            self.assertTrue(all(self.params[0] > params[0])) # simply reduced
            self.params[...] = params


    def test_update_exponential_decay(self):

        beta1=0.9
        optimizer = Adam(lr=0.5, beta1=beta1)
        self.params, params = np.array([[1.]]), np.array([[1.]])
        grads = np.array([[100.]])

        optimizer.update(params, grads)
        self.params[...] = params

        grads *= (-1 * beta1) # update grads by the same degree as beta1
        optimizer.update(params, grads)
        self.assertEqual(self.params[0], params[0]) # because m is also updated by beta1
        self.params[...] = params

        grads *= 0.5 # update but no sign change
        optimizer.update(params, grads)
        self.assertLess(self.params, params[0]) # updated in the positive direction
        self.params[...] = params

        # momentum works
        grads *= (-0.99 * beta1) # flip grads by shy of beta1
        optimizer.update(params, grads)
        self.assertLess(self.params, params[0]) # still updated in the positive direction
        self.params[...] = params

        # momentum works
        optimizer.update(params, grads) # with the same grads
        self.assertGreater(self.params, params[0]) # this time updated in the negative direction


if __name__ == '__main__':
    unittest.main(verbosity=2)