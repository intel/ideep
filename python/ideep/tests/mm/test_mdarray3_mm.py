import numpy
import ideep4py
import testing
import unittest


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32],
    'shape': [(3, 16, 2, 4), (2, 7, 1, 1),
              (2, 7, 3, 2), (2, 2, 2, 2), (3, 4), (1, 1)],
}))
class TestMdarray3(unittest.TestCase):
    def setUp(self):
        self.x = numpy.ndarray(shape=self.shape, dtype=self.dtype, order='C')
        self.x.fill(2.)
        self.check_options = {'atol': 1e-5, 'rtol': 1e-4}

    @unittest.skip("should be catch in the future")
    def test_noneInput(self):
        x1 = None
        x2 = numpy.ndarray(x1)
        x = ideep4py.mdarray(x1)
        print(x, x2)

    def test_basicOp(self):
        x1 = self.x
        x = ideep4py.mdarray(x1)
        numpy.testing.assert_allclose(1 / x1, 1 / x, **self.check_options)
        numpy.testing.assert_allclose(2 * x1, 2 * x, **self.check_options)
        numpy.testing.assert_allclose(1 - x1, 1 - x, **self.check_options)
        numpy.testing.assert_allclose(1 + x1, 1 + x, **self.check_options)
        x1 /= 3
        x /= 3
        numpy.testing.assert_allclose(x1, x, **self.check_options)

        x1 *= 2
        x *= 2
        numpy.testing.assert_allclose(x1, x, **self.check_options)

        x1 += 3
        x += 3
        numpy.testing.assert_allclose(x1, x, **self.check_options)

        x1 -= 5
        x -= 5
        numpy.testing.assert_allclose(x1, x, **self.check_options)
