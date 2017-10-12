import numpy
import chainer
import unittest
import chainer.functions as F
import example.functions as E

from chainer import gradient_check
from chainer import testing
from chainer.testing import condition


def _as_mat(x):
    if x.ndim == 2:
        return x
    return x.reshape(len(x), -1)


@testing.parameterize(*testing.product({
    'in_shape': [(4, 3), (4, 3, 1, 1)],
    'out_shape': [(4, 2)],
    'x_dtype': [numpy.float32],
    'W_dtype': [numpy.float32],
}))
class TestNonparameterizedLinear(unittest.TestCase):

    def setUp(self):
        self.W = numpy.random.uniform(
                -1, 1, (self.out_shape[1],) + self.in_shape[1:]).astype(self.W_dtype)
        self.b = numpy.random.uniform(
            -1, 1, 2).astype(self.x_dtype)

        self.x = numpy.random.uniform(-1, 1, self.in_shape).astype(self.x_dtype)
        self.gy = numpy.random.uniform(-1, 1, self.out_shape).astype(self.x_dtype)
        self.y = _as_mat(self.x).dot(_as_mat(self.W).T) + self.b
        self.check_forward_options = {}
        self.check_backward_options = {}
        if self.x_dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-3, 'rtol': 1e-2}
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 5e-4, 'rtol': 5e-3}
        elif self.W_dtype == numpy.float16:
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 5e-4, 'rtol': 5e-3}

    def check_forward(self, x_data, W_data, b_data):
        x = chainer.Variable(x_data)
        W = chainer.Variable(W_data)
        if b_data is None:
            y = E.linear(x, W)
            y_expect = F.linear(x, W)
        else:
            b = chainer.Variable(b_data)
            y = E.linear(x, W, b)
            y_expect = F.linear(x, W, b)
        self.assertEqual(y.data.dtype, self.x_dtype)
        testing.assert_allclose(
            y_expect, y.data, **self.check_forward_options)

    @condition.retry(3)
    def test_forward_cpu(self):
        print("test forward cpu")
        self.check_forward(self.x, self.W, self.b)
        print("finish")

    @condition.retry(3)
    def test_forward_cpu_nobias(self):
        print("test forward cpu nobias")
        self.check_forward(self.x, self.W, None)
        print("finish")

    def check_backward(self, x_data, W_data, b_data, y_grad):
        def f(*args):
            E.linear(*args)

        args = (x_data, W_data)
        if b_data is not None:
            args = args + (b_data,)

        gradient_check.check_backward(
            f, args, y_grad,
            eps=1e-2, **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        print("test backward cpu")
        self.check_backward(self.x, self.W, self.b, self.gy)
        print("finish")

    @condition.retry(3)
    def test_backward_cpu_nobias(self):
        print("test backward cpu nobias")
        self.check_backward(self.x, self.W, None, self.gy)
        print("finish")

testing.run_module(__name__, __file__)
