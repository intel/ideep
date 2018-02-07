import numpy
import ideep4py
import testing
import unittest


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32],
    'shape': [(3, 16, 2, 4), (2, 7, 1, 1),
              (2, 7, 3, 2), (2, 2, 2, 2), (3, 4), (1, 1)],
}))
class TestMemcpy(unittest.TestCase):
    def setUp(self):
        self.check_options = {'atol': 1e-5, 'rtol': 1e-4}

    def test_memcpy(self):
        x1 = numpy.ndarray(shape=self.shape, dtype=self.dtype, order='C')
        x = ideep4py.mdarray(x1)
        x2 = numpy.array(x)
        print("x = ", x1)
        print("x2 = ", x2)
        numpy.testing.assert_allclose(x, x2, **self.check_options)
