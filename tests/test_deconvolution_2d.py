import mock
import unittest

import numpy

import chainer
from chainer import cuda
import chainer.functions as F
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer.testing import parameterize
from chainer.utils import conv


import example.functions as E


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x


@parameterize(*(testing.product({
    'c_contiguous': [True],
    'test_outsize': [True, False],
    'nobias': [True],
    'stride': [1, 2],
    'x_dtype': [numpy.float32],
    'W_dtype': [numpy.float32],
}) + testing.product({
    'c_contiguous': [False],
    'test_outsize': [True],
    'nobias': [False],
    'stride': [1, 2],
    'x_dtype': [numpy.float32],
    'W_dtype': [numpy.float32],
})))
class TestDeconvolution2DFunction(unittest.TestCase):

    in_channels = 3
    out_channels = 2
    ksize = 3
    pad = 1

    def setUp(self):
        kh, kw = _pair(self.ksize)
        sh, sw = _pair(self.stride)
        ph, pw = _pair(self.pad)
        self.W = numpy.random.normal(
            0, numpy.sqrt(1. / (kh * kw * self.in_channels)),
            (self.in_channels, self.out_channels, kh, kw)
        ).astype(self.W_dtype)
        self.b = None if self.nobias else numpy.random.uniform(
            -1, 1, self.out_channels).astype(self.x_dtype)

        N = 2
        inh, inw = 4, 3
        outh = conv.get_deconv_outsize(inh, kh, sh, ph)
        outw = conv.get_deconv_outsize(inw, kw, sw, pw)
        self.outsize = (outh, outw) if self.test_outsize else None
        self.x = numpy.random.uniform(
            -1, 1, (N, self.in_channels, inh, inw)).astype(self.x_dtype)
        self.gy = numpy.random.uniform(
            -1, 1, (N, self.out_channels, outh, outw)).astype(self.x_dtype)

        self.ggx = numpy.random.uniform(-1, 1, self.x.shape).astype(
            self.x_dtype)
        self.ggW = numpy.random.uniform(-1, 1, self.W.shape).astype(
            self.W_dtype)
        self.ggb = None if self.nobias else numpy.random.uniform(
            -1, 1, self.b.shape).astype(self.x_dtype)

        self.test_forward_options = {}
        self.check_backward_options = {'dtype': numpy.float64}
        self.check_double_backward_options = {'dtype': numpy.float64}
        if self.x_dtype == numpy.float16:
            self.test_forward_options.update(atol=5e-3, rtol=5e-2)
            self.check_backward_options.update(atol=5e-4, rtol=5e-3)
            self.check_double_backward_options.update(atol=5e-3, rtol=5e-2)
        elif self.W_dtype == numpy.float16:
            self.check_backward_options.update(atol=5e-4, rtol=5e-3)
            self.check_double_backward_options.update(atol=5e-3, rtol=5e-2)

    def test_forward_consistency(self):
        x_cpu = chainer.Variable(self.x)
        W_cpu = chainer.Variable(self.W)
        b_cpu = None if self.nobias else chainer.Variable(self.b)
        y_cpu = F.deconvolution_2d(
            x_cpu, W_cpu, b_cpu, stride=self.stride, pad=self.pad,
            outsize=self.outsize)

        x_mkl = chainer.Variable(self.x)
        W_mkl = chainer.Variable(self.W)
        b_mkl = None if self.nobias else chainer.Variable(self.b)
        y_mkl = E.deconvolution_2d(
            x_mkl, W_mkl, b_mkl, stride=self.stride, pad=self.pad,
            outsize=self.outsize)

        self.assertEqual(y_cpu.data.dtype, self.x_dtype)
        self.assertEqual(y_mkl.data.dtype, self.x_dtype)
        testing.assert_allclose(
            y_cpu.data, numpy.array(y_mkl.data, copy=False),
            **self.test_forward_options)

    def check_backward(self, x_data, W_data, b_data, y_grad):
        xp = cuda.get_array_module(x_data)

        if not self.c_contiguous:
            x_data = xp.asfortranarray(x_data)
            W_data = xp.asfortranarray(W_data)
            y_grad = xp.asfortranarray(y_grad)
            self.assertFalse(x_data.flags.c_contiguous)
            self.assertFalse(W_data.flags.c_contiguous)
            self.assertFalse(y_grad.flags.c_contiguous)
            if b_data is not None:
                b = xp.empty((len(b_data) * 2,), dtype=self.b.dtype)
                b[::2] = b_data
                b_data = b[::2]
                self.assertFalse(b_data.flags.c_contiguous)

        args = (x_data, W_data)
        if b_data is not None:
            args = args + (b_data,)

        def f(*args):
            return E.deconvolution_2d(
                *args, stride=self.stride, pad=self.pad, outsize=self.outsize)

        gradient_check.check_backward(
            f, args, y_grad, **self.check_backward_options)

    @condition.retry(10)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.W, self.b, self.gy)

    def check_double_backward(self, x_data, W_data, b_data, y_grad,
                              x_grad_grad, W_grad_grad, b_grad_grad):
        xp = cuda.get_array_module(x_data)

        if not self.c_contiguous:
            x_data = xp.asfortranarray(x_data)
            W_data = xp.asfortranarray(W_data)
            y_grad = xp.asfortranarray(y_grad)
            x_grad_grad = xp.asfortranarray(x_grad_grad)
            W_grad_grad = xp.asfortranarray(W_grad_grad)
            self.assertFalse(x_data.flags.c_contiguous)
            self.assertFalse(W_data.flags.c_contiguous)
            self.assertFalse(y_grad.flags.c_contiguous)
            self.assertFalse(x_grad_grad.flags.c_contiguous)
            self.assertFalse(W_grad_grad.flags.c_contiguous)
            if b_data is not None:
                b = xp.empty((len(b_data) * 2,), dtype=self.b.dtype)
                b[::2] = b_data
                b_data = b[::2]
                self.assertFalse(b_data.flags.c_contiguous)

                ggb = xp.empty((len(b_data) * 2,), dtype=self.b.dtype)
                ggb[::2] = b_grad_grad
                b_grad_grad = ggb[::2]
                self.assertFalse(b_grad_grad.flags.c_contiguous)

        args = (x_data, W_data)
        grad_grads = (x_grad_grad, W_grad_grad)
        if b_data is not None:
            args = args + (b_data,)
            grad_grads = grad_grads + (b_grad_grad,)

        def f(*args):
            y = E.deconvolution_2d(
                *args, stride=self.stride, pad=self.pad, outsize=self.outsize)
            return y * y  # make the function nonlinear

        gradient_check.check_double_backward(
            f, args, y_grad, grad_grads,
            **self.check_double_backward_options)

    @condition.retry(10)
    def test_double_backward_cpu(self):
        self.check_double_backward(self.x, self.W, self.b, self.gy,
                                   self.ggx, self.ggW, self.ggb)


testing.run_module(__name__, __file__)
