# import mock
import unittest

import numpy

import chainer
from chainer import cuda
import chainer.functions as F
from chainer import gradient_check
from chainer import testing
from chainer.testing import condition
from chainer.utils import conv

import example.functions as E


@testing.parameterize(*(
    testing.product({
        'in_shape': [(2, 3, 4, 3)],
        'kernel_geo': [(2, 3, 3, 2, 1)],
        'c_contiguous': [True],
        'cover_all': [True, False],
        'x_dtype': [numpy.float32],
        'W_dtype': [numpy.float32], }) +
    testing.product({
        'in_shape': [(1, 3, 9, 9)],
        'kernel_geo': [(8, 3, 3, 1, 0)],
        'c_contiguous': [True],
        'cover_all': [True, False],
        'x_dtype': [numpy.float32],
        'W_dtype': [numpy.float32], }) +
    testing.product({
        'in_shape': [(8, 3, 15, 15)],
        'kernel_geo': [(3, 11, 11, 4, 0)],
        'c_contiguous': [True],
        'cover_all': [True],
        'x_dtype': [numpy.float32],
        'W_dtype': [numpy.float32]})
    ))
class TestConvolution2DFunction(unittest.TestCase):

    def setUp(self):
        n, c, h, w = self.in_shape
        out_c = self.kernel_geo[0]
        kh, kw = (self.kernel_geo[1], self.kernel_geo[2])
        self.stride = self.kernel_geo[3]
        self.pad = self.kernel_geo[4]
        self.W = numpy.random.normal(
            0, numpy.sqrt(1. / (kh * kw * c)),
            (out_c, c, kh, kw)).astype(self.W_dtype)

        self.b = numpy.random.uniform(
            -1, 1, out_c).astype(self.x_dtype)

        self.x = numpy.random.uniform(
            -1, 1, self.in_shape).astype(self.x_dtype)

        out_h = conv.get_conv_outsize(
            h, kh, self.stride, self.pad, cover_all=self.cover_all)

        out_w = conv.get_conv_outsize(
            w, kw,
            self.stride, self.pad, cover_all=self.cover_all)

        self.gy = numpy.random.uniform(
            -1, 1,
            (n, out_c, out_h, out_w)).astype(self.x_dtype)

        self.ggx = numpy.random.uniform(-1, 1, self.x.shape).astype(
            self.x_dtype)
        self.ggW = numpy.random.uniform(-1, 1, self.W.shape).astype(
            self.W_dtype)
        self.ggb = numpy.random.uniform(-1, 1, self.b.shape).astype(
            self.x_dtype)

        self.check_forward_options = {}
        self.check_backward_options = {
            'dtype': numpy.float32, 'atol': 5e-4, 'rtol': 5e-3}
        self.check_double_backward_options = {'dtype': numpy.float32}
        if self.x_dtype == numpy.float16 or self.W_dtype == numpy.float16:
            self.check_forward_options = {'atol': 5e-4, 'rtol': 5e-3}
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 5e-4, 'rtol': 5e-3}

    def test_forward_consistency(self, nobias=False):
        x_cpu = chainer.Variable(self.x)
        W_cpu = chainer.Variable(self.W)
        b_cpu = None if nobias else chainer.Variable(self.b)
        y_cpu = F.convolution_2d(
                x_cpu, W_cpu, b_cpu, stride=self.stride, pad=self.pad,
                cover_all=self.cover_all)

        x_mkl = chainer.Variable(self.x)
        W_mkl = chainer.Variable(self.W)
        b_mkl = None if nobias else chainer.Variable(self.b)
        y_mkl = E.convolution_2d(
            x_mkl, W_mkl, b_mkl, stride=self.stride, pad=self.pad,
            cover_all=self.cover_all)

        testing.assert_allclose(
            y_cpu.data,
            numpy.array(y_mkl.data, copy=False),
            **self.check_forward_options)

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
            return E.convolution_2d(*args, stride=self.stride, pad=self.pad,
                                    cover_all=self.cover_all)

        gradient_check.check_backward(
            f, args, y_grad, **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.W, self.b, self.gy)

    @condition.retry(3)
    def test_backward_cpu_nobias(self):
        self.check_backward(self.x, self.W, None, self.gy)

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
            y = E.convolution_2d(*args, stride=self.stride, pad=self.pad,
                                 cover_all=self.cover_all)
            return y * y  # make the function nonlinear

        gradient_check.check_double_backward(f, args, y_grad, grad_grads,
           **self.check_double_backward_options)

    @condition.retry(3)
    def test_double_backward_cpu(self):
        self.check_double_backward(self.x, self.W, self.b, self.gy,
                                   self.ggx, self.ggW, self.ggb)

    @condition.retry(3)
    def test_double_backward_cpu_nobias(self):
        self.check_double_backward(self.x, self.W, None, self.gy,
                                   self.ggx, self.ggW, None)


testing.run_module(__name__, __file__)
