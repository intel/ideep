import numpy
import ideep4py
import testing
import unittest


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32],
    'shape': [(3, 16, 2, 4), (2, 7, 1, 1),
              (2, 7, 3, 2), (2, 2, 2, 2), (3, 4), (1, 1)],
}))
class TestMdarrayReshape(unittest.TestCase):
    def setUp(self):
        self.x = numpy.ndarray(shape=self.shape, dtype=self.dtype, order='C')
        # self.x.fill(2.)
        self.check_options = {'atol': 1e-5, 'rtol': 1e-4}

    def test_list(self):
        x1 = numpy.ndarray(shape=(2, 2, 2, 2), dtype=numpy.float32, order='C')
        x = ideep4py.mdarray(x1)
        y1 = x1.reshape([4, 4])
        y = x.reshape([4, 4])
        numpy.testing.assert_allclose(y, y1, **self.check_options)

    def test_single_number(self):
        x1 = numpy.ndarray(shape=(2, 2, 2, 2), dtype=numpy.float32, order='C')
        x = ideep4py.mdarray(x1)
        x1.reshape(16)
        x.reshape(16)
        numpy.testing.assert_allclose(x, x1, 1e-5, 1e-4)

    def test_value_change(self):
        # value change
        x1 = numpy.ndarray(shape=(2, 2, 2, 2), dtype=numpy.float32, order='C')
        x = ideep4py.mdarray(x1)
        y = x.reshape(len(x), -1)
        x[0, 0, 0, 0] = 3.333
        self.assertEqual(x[0, 0, 0, 0], y[0, 0])
        x[0, 0, 0, 0] = 4.4444
        self.assertEqual(x[0, 0, 0, 0], y[0, 0])

    def test_minusOne(self):
        x1 = numpy.ndarray(shape=(2, 2, 2, 2), dtype=numpy.float32, order='C')
        x = ideep4py.mdarray(x1)
        y = x.reshape((2, 2, -1))
        y1 = x1.reshape((2, 2, -1))
        numpy.testing.assert_allclose(y, y1, 1e-5, 1e-4)
