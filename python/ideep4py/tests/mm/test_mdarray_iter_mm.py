import numpy
# import ideep4py
import six
import testing
import unittest
from ideep4py import relu, mdarray


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32],
    'shape': [(3, 16, 2, 4), (2, 7, 1, 1),
              (2, 7, 3, 2), (2, 2, 2, 2), (3, 4), (256, 512, 13, 13), (1, 1)],
}))
class TestMdarrayIter(unittest.TestCase):
    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.mx = mdarray(self.x)
        self.x1 = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.x2 = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.check_options = {'atol': 1e-5, 'rtol': 1e-4}

    def test_enumerate(self):
        a = []
        b = []
        for p, xi in enumerate(self.x):
            a.append(xi)
        for p, mxi in enumerate(self.mx):
            b.append(mxi)
        numpy.testing.assert_allclose(numpy.asarray(a), numpy.asarray(b))

    def test_zip(self):
        mx1 = mdarray(self.x1)
        mx2 = mdarray(self.x2)
        a1 = []
        a2 = []
        b1 = []
        b2 = []
        for x, y in six.moves.zip(self.x1, self.x2):
            a1.append(x)
            a2.append(y)
        for mx, my in six.moves.zip(mx1, mx2):
            b1.append(mx)
            b2.append(my)
        numpy.testing.assert_allclose(numpy.asarray(a1), numpy.asarray(b1))
        numpy.testing.assert_allclose(numpy.asarray(a2), numpy.asarray(b2))

    def test_mkldnn_format(self):
        y = numpy.maximum(self.x, 0, dtype=self.x.dtype)
        my = relu.Forward(self.mx)
        numpy.testing.assert_allclose(y, my)
        a = []
        b = []
        for p, xi in enumerate(y):
            a.append(xi)
        for p, mxi in enumerate(my):
            b.append(mxi)
        numpy.testing.assert_allclose(numpy.asarray(a), numpy.asarray(b))
