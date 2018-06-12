import ideep4py  # NOQA
import numpy
import testing
from ideep4py import mdarray
import unittest


class TestMdarraySet(unittest.TestCase):
    def setUp(self):
        self.check_options = {'atol': 1e-5, 'rtol': 1e-4}

    def test_set1(self):
        x = numpy.array([1, 1, 1], dtype=numpy.float32)
        mx = mdarray(x)
        numpy.testing.assert_allclose(
            mx, x, **self.check_options)
        x = numpy.array([1, 2, 1], dtype=numpy.float32)
        mx.set(x)
        numpy.testing.assert_allclose(
            mx, x, **self.check_options)

    def test_set2(self):
        x = numpy.arange(24, dtype=numpy.float32)
        mx = mdarray(x)
        numpy.testing.assert_allclose(
            mx, x, **self.check_options)
        x.fill(1)
        mx.set(x)
        numpy.testing.assert_allclose(
            mx, x, **self.check_options)

    def test_set3(self):
        x = numpy.random.rand(10, 10, 10, 10)
        x = x.astype(numpy.float32)
        mx = mdarray(x)
        numpy.testing.assert_allclose(
            mx, x, **self.check_options)
        x = numpy.random.rand(10, 10, 10, 10)
        x = x.astype(numpy.float32)
        mx.set(x)
        numpy.testing.assert_allclose(
            mx, x, **self.check_options)

    def test_set4(self):
        x = numpy.array([0, 0, 0, 0], dtype=numpy.float32)
        mx1 = mdarray(x)
        mx2 = mdarray(x)
        mx1.fill(0)
        mx2.set(mx1)
        numpy.testing.assert_allclose(
            mx1, mx2, **self.check_options)


testing.run_module(__name__, __file__)
