import ideep4py  # NOQA
import numpy
import testing
from ideep4py import relu, mdarray
import unittest


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32],
    'shape': [(3, 16, 2, 4), (2, 7, 1, 1),
              (2, 7, 3, 2), (2, 2, 2, 2), (3, 4), (1, 1)],
}))
class TestMdarraySum(unittest.TestCase):
    def setUp(self):
        self.check_options = {'atol': 1e-5, 'rtol': 1e-4}

    def test_sum1(self):
        x = numpy.arange(24, dtype=numpy.float32)
        x = x.reshape((2, 3, 2, 2))
        mx = mdarray(x)
        numpy.testing.assert_allclose(
            mx.sum((0, 2, 3)), x.sum((0, 2, 3)), **self.check_options)

    def test_sum2(self):
        x = numpy.arange(24, dtype=numpy.float32)
        x = x.reshape((2, 3, 2, 2))
        mx = mdarray(x)
        numpy.testing.assert_allclose(
            mx.sum((1, 2)), x.sum((1, 2)), **self.check_options)

    def test_sum3(self):
        x = numpy.arange(24, dtype=numpy.float32)
        x = x.reshape((2, 3, 2, 2))
        mx = mdarray(x)
        numpy.testing.assert_allclose(
            mx.sum((0, 2)), x.sum((0, 2)), **self.check_options)
        numpy.testing.assert_allclose(
            mx.sum((1, 3)), x.sum((1, 3)), **self.check_options)
        numpy.testing.assert_allclose(
            mx.sum((0)), x.sum((0)), **self.check_options)
        numpy.testing.assert_allclose(
            mx.sum((1, 2, 3)), x.sum((1, 2, 3)), **self.check_options)
        numpy.testing.assert_allclose(
            mx.sum((0, 1, 2)), x.sum((0, 1, 2)), **self.check_options)
        numpy.testing.assert_allclose(
            mx.sum((3)), x.sum((3)), **self.check_options)
        numpy.testing.assert_allclose(
            mx.sum((1)), x.sum((1)), **self.check_options)

    def test_sum4(self):
        x = numpy.random.rand(256, 384, 13, 13)
        x = x.astype(numpy.float32)
        y = numpy.maximum(x, 0, dtype=numpy.float32)
        mx = mdarray(x)
        my = relu.Forward(mx)
        numpy.testing.assert_allclose(
            my.sum((0, 2, 3)), y.sum((0, 2, 3)), **self.check_options)
        numpy.testing.assert_allclose(
            my.sum((1, 2, 3)), y.sum((1, 2, 3)), **self.check_options)
        numpy.testing.assert_allclose(
            my.sum((0, 1, 2)), y.sum((0, 1, 2)), **self.check_options)
        numpy.testing.assert_allclose(
            my.sum((0, 2)), y.sum((0, 2)), **self.check_options)
        numpy.testing.assert_allclose(
            my.sum((1, 3)), y.sum((1, 3)), **self.check_options)
        numpy.testing.assert_allclose(
            my.sum((1, 2)), y.sum((1, 2)), **self.check_options)
        numpy.testing.assert_allclose(
            my.sum((0)), y.sum((0)), **self.check_options)
        numpy.testing.assert_allclose(
            my.sum((3)), y.sum((3)), **self.check_options)
        numpy.testing.assert_allclose(
            my.sum((2)), y.sum((2)), **self.check_options)
        numpy.testing.assert_allclose(
            my.sum((0, 2, 3)), y.sum((0, 2, 3)), **self.check_options)
        numpy.testing.assert_allclose(
            my.sum((0, 2, 3), keepdims=True),
            y.sum((0, 2, 3), keepdims=True), **self.check_options)

    def test_sum5(self):
        x = numpy.random.rand(256, 385, 13, 13)
        x = x.astype(numpy.float32)
        mx = mdarray(x)
        ms = mx.sum((0, 2, 3))
        ns = x.sum((0, 2, 3))
        numpy.testing.assert_allclose(ms, ns, **self.check_options)

    def test_sum6(self):
        x = numpy.random.rand(256, 1000)
        x = x.astype(numpy.float32)
        mx = mdarray(x)
        ms = mx.sum((0))
        ns = x.sum((0))
        numpy.testing.assert_allclose(ms, ns, **self.check_options)

    def test_sum7(self):
        x = numpy.random.rand(256, 1000)
        x = x.astype(numpy.float32)
        mx = mdarray(x)
        ms = mx.sum((1))
        ns = x.sum((1))
        numpy.testing.assert_allclose(ms, ns, **self.check_options)


testing.run_module(__name__, __file__)
