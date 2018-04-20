import sys
import unittest

import numpy
import six
import ideep4py
from ideep4py import batchNormalization

try:
    import testing
    from testing import condition
except Exception as ex:
    print('*** testing directory is missing: %s' % ex)
    sys.exit(-1)


def _x_hat(x, mean, inv_std):
    x_mu = x - mean
    x_mu *= inv_std
    return x_mu


def _batch_normalization(expander, gamma, beta, x, mean, var):
    mean = mean[expander]
    std = numpy.sqrt(var)[expander]
    y_expect = (gamma[expander] * (x - mean) / std + beta[expander])
    return y_expect


@testing.parameterize(*(testing.product({
    'param_shape': [(3, ), ],
    'ndim': [2, ],
    'dtype': [numpy.float32],
})))
class TestBatchNormalizationF32(unittest.TestCase):

    def setUp(self):
        self.eps = 2e-5
        self.expander = (None, Ellipsis) + (None,) * self.ndim
        self.gamma = numpy.random.uniform(.5, 1,
                                          self.param_shape).astype(self.dtype)
        self.beta = numpy.random.uniform(-1, 1,
                                         self.param_shape).astype(self.dtype)
        self.head_ndim = self.gamma.ndim + 1
        shape = (5,) + self.param_shape + (2,) * self.ndim
        self.x = numpy.random.uniform(-1, 1, shape).astype(self.dtype)
        self.gy = numpy.random.uniform(-1, 1, shape).astype(self.dtype)

        self.args = [self.x, self.gamma, self.beta]
        self.aggr_axes = (0,) + tuple(
            six.moves.range(self.head_ndim, self.x.ndim))
        self.mean = self.x.mean(axis=self.aggr_axes)
        self.var = self.x.var(axis=self.aggr_axes) + self.eps
        self.check_forward_options = {'atol': 1e-4, 'rtol': 1e-3}
        self.check_backward_options = {'atol': 1e-4, 'rtol': 1e-3}

    def check_forward(self, args):
        x, gamma, beta = args
        expander = (None, Ellipsis) + (None,) * (x.ndim - self.head_ndim)
        self.expander = expander
        self.axis = (0,) + tuple(range(self.head_ndim, x.ndim))
        expand_dim = False
        if x.ndim == 2:
            expand_dim = True
            x = x[:, :, None, None]

        gamma = gamma[expander]
        beta = beta[expander]

        y_act, self.mean, self.var, inv_std = batchNormalization.Forward(
            ideep4py.mdarray(x),
            ideep4py.mdarray(gamma),
            ideep4py.mdarray(beta),
            None,
            None,
            self.eps
        )

        if expand_dim:
            y_act = numpy.squeeze(y_act, axis=(2, 3))
        y_act = numpy.array(y_act, dtype=self.dtype)

        y_expect = _batch_normalization(
            self.expander, self.gamma, self.beta, self.x, self.mean, self.var)

        numpy.testing.assert_allclose(
            y_expect, y_act, **self.check_forward_options)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.args)

    def check_backward(self, args, y_grad):
        x, gamma, beta = args
        gy = y_grad
        expander = self.expander
        inv_m = gamma.dtype.type(1. / (x.size // gamma.size))

        expand_dim = False
        if x.ndim == 2:
            expand_dim = True
            x = x[:, :, None, None]
            gy = gy[:, :, None, None]

        gamma = gamma[self.expander]

        gx_act, gW = batchNormalization.Backward(
            ideep4py.mdarray(x),
            ideep4py.mdarray(gy),
            ideep4py.mdarray(self.mean),
            ideep4py.mdarray(self.var),
            ideep4py.mdarray(gamma),
            self.eps
        )
        if expand_dim:
            gx_act = numpy.squeeze(gx_act, axis=(2, 3))
        gx_act = numpy.array(gx_act, dtype=self.dtype)

        self.inv_std = self.var ** (-0.5)

        gbeta = y_grad.sum(axis=self.aggr_axes)
        x_hat = _x_hat(x, self.mean[expander], self.inv_std[expander])
        ggamma = (y_grad * x_hat).sum(axis=self.aggr_axes)
        gx_expect = (self.gamma * self.inv_std)[expander] * (
            y_grad - (x_hat * ggamma[expander] + gbeta[expander]) * inv_m)

        numpy.testing.assert_allclose(
            gx_expect, gx_act, **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.args, self.gy)


testing.run_module(__name__, __file__)
