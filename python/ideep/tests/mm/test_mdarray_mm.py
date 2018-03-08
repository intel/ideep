import numpy
import ideep4py
import testing
import unittest


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32],
    'shape': [(3, 16, 2, 4), (2, 7, 1, 1),
              (2, 7, 3, 2), (2, 2, 2, 2), (3, 4), (1, 1)],
}))
class TestMdarray(unittest.TestCase):
    def setUp(self):
        self.x1 = numpy.ndarray(shape=self.shape, dtype=self.dtype, order='C')
        self.xOne = numpy.ones(shape=self.shape, dtype=self.dtype, order='C')
        self.check_options = {'atol': 1e-5, 'rtol': 1e-4}

    def test_addOne(self):
        x = ideep4py.mdarray(self.x1)
        x = x + 1
        numpy.testing.assert_allclose(self.x1 + 1, x, **self.check_options)
        # testing.assert_allclose(x1 + 1, x)

    def test_add(self):
        x = ideep4py.mdarray(self.x1)
        y = self.x1
        y += x
        x += x
        x2 = numpy.array(x)
        numpy.testing.assert_allclose(x2, y, **self.check_options)
        # testing.assert_allclose(x1, x2)

    def test_oneAdd(self):
        x = ideep4py.mdarray(self.xOne)
        y = x + self.xOne
        y2 = numpy.array(y)
        numpy.testing.assert_allclose(
            y2, self.xOne + self.xOne, **self.check_options)

    def test_mul(self):
        x = ideep4py.mdarray(self.xOne)
        y = x * self.xOne
        y2 = numpy.array(y)
        numpy.testing.assert_allclose(
            y2, self.xOne * self.xOne, **self.check_options)
        # testing.assert_allclose(y2, self.xOne * self.xOne)

    def test_md(self):
        x1 = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        x = ideep4py.mdarray(x1)
        z1 = (x1 > 0).astype(x1.dtype)
        z = (x > 0).astype(x1.dtype)
        numpy.testing.assert_allclose(z, z1, **self.check_options)
        # testing.assert_allclose(z, z1)

    # @unittest.skip("demonstrating skipping")
    def test_attriMdarray(self):
        x = ideep4py.mdarray(self.x1)
        self.assertEqual(x.ndim, self.x1.ndim)
        self.assertEqual(x.shape, self.x1.shape)
        self.assertEqual(x.size, self.x1.size)
        self.assertEqual(x.dtype, self.x1.dtype)
        self.assertTrue(x.is_mdarray)
        # self.assertTrue(0)
