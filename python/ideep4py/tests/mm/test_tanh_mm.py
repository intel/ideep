import numpy
import ideep4py
import testing
import unittest


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32],
    'shape': [(3, 16, 2, 4), (2, 7, 1, 1), (2, 7, 3, 2),
              (1, 32, 224, 224), (2, 2, 2, 2), (3, 4), (1, 1)],
}))
class TestTanh(unittest.TestCase):
    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.y = numpy.tanh(self.x)
        self.check_options = {'atol': 1e-5, 'rtol': 1e-4}

    def test_tanh(self):
        mx = ideep4py.mdarray(self.x)
        x2 = numpy.array(mx)
        numpy.testing.assert_allclose(self.x, x2)

    def test_tanhForward(self):
        mx = ideep4py.mdarray(self.x)
        my = ideep4py._ideep4py.tanh.Forward(mx)
        y2 = numpy.array(my)
        numpy.testing.assert_allclose(self.y, y2, **self.check_options)

    def test_tanhBackward(self):
        x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        gy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        gx = gy * (1 - numpy.tanh(x) ** 2)

        mx = ideep4py.mdarray(x)
        mgy = ideep4py.mdarray(gy)
        mgx = ideep4py._ideep4py.tanh.Backward(mx, mgy)

        gx1 = numpy.array(mgx)
        numpy.testing.assert_allclose(gx1, gx, **self.check_options)
