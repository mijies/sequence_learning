import unittest
import numpy
import os
import sys
sys.path.append('..')
from models.base_model import BaseModel

class TestModelsBaseModel(unittest.TestCase):
    """test class for common/base_model.py
    """
    def setUp(self):
        self.err = None

    def test_not_implemented_forward(self):
        class SubBaseModel(BaseModel):
            def backward(self):pass
        try:
            subcls = SubBaseModel()
        except TypeError as e:
            self.err = e
        self.assertEqual(type(self.err), type(TypeError()))


    def test_not_implemented_backward(self):
        class SubBaseModel(BaseModel):
            def forward(self):pass
        try:
            subcls = SubBaseModel()
        except TypeError as e:
            self.err = e
        self.assertEqual(type(self.err), type(TypeError()))


    def test_save_load_params(self):
        class BaseModelTest(BaseModel):
            def forward(self):pass
            def backward(self):pass

        subcls = BaseModelTest()
        x = numpy.array([1., -2.])
        subcls.params = x
        subcls.save_params()
        subcls.params = None
        subcls.load_params()
        self.assertTrue(all(subcls.params==x))

        file_name = "basemodel_save_param_test.pkl"
        x = numpy.array([10.5, 20.25])
        subcls.params = x
        subcls.save_params(file_name)
        subcls.params = None
        subcls.load_params(file_name)
        self.assertTrue(all(subcls.params==x))


    def tearDown(self):
        file_name = "BaseModelTest.pkl"
        if os.path.exists(file_name):
            os.remove(file_name)
            print('\n --- %s is deleted' % file_name)

        file_name = "basemodel_save_param_test.pkl"
        if os.path.exists(file_name):
            os.remove(file_name)
            print(' --- %s is deleted' % file_name)


if __name__ == '__main__':
    unittest.main(verbosity=2)