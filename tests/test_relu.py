import unittest

import mock
import numpy

import chainer
import chainer.functions as F
import example.functions as E

from chainer import gradient_check
from chainer import testing
from chainer.testing import condition



@testing.parameterize(*testing.product({
    'shape': [(3, 2)],
    'dtype': [numpy.float32],
}))
class TestReLU(unittest.TestCase):

    def setUp(self):
        # Avoid unstability of numerical grad
        # fanout.clear()
        x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        for i in numpy.ndindex(self.shape):
            if -0.1 < x[i] < 0.1:
                x[i] = 0.5

        self.x = x
        self.gy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.check_backward_options = {}

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = E.relu(x)
        self.assertEqual(y.data.dtype, self.dtype)

        y_expect = F.relu(x)
        testing.assert_allclose(y_expect.data, y.data)

#    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    def check_backward(self, x_data, y_grad):
        def f(*args):
            E.relu(*args)

        gradient_check.check_backward(
                f, x_data, y_grad,
                **self.check_backward_options)

#    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

testing.run_module(__name__, __file__)
