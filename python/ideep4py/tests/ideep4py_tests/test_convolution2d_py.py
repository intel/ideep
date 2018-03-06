import os
import sys
import unittest
import numpy
import ideep4py
from ideep4py import convolution2DParam
from ideep4py import convolution2D

try:
    import testing
    from testing import condition
    from testing.conv import im2col_cpu, col2im_cpu, get_conv_outsize
except Exception as ex:
    print('*** testing directory is missing: %s' % ex)
    sys.exit(-1)


def _set_cover_all(self, x, W):
    in_h, in_w = x.shape[2:]
    kh, kw = W.shape[2:]
    self.cover_all = (
        in_h != get_conv_outsize(self.outh, kh, self.sy,
                                 self.ph, d=self.dy) or
        in_w != get_conv_outsize(self.outw, kw, self.sx,
                                 self.pw, d=self.dx))


if bool(int(os.environ.get('ENALE_TRAVIS_TEST', '0'))):
    bs_list = [1, 2, 4, 5, 8, 10, 16, 32, 64, ]
else:
    bs_list = [1, 2, 4, 5, 8, 10, 16, 32, 64, 96, 128, 192, 256, 512, ]
print('bs_list: ', bs_list)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, ],
    'cover_all': [False, True],
    'channel': [1, 2, 4, 8, 10, ],
    'bs': bs_list,
    'with_bias': [True, ],
}))
@testing.fix_random()
class TestConvolution2DPyF32(unittest.TestCase):

    def setUp(self):
        self.x_shape = (self.bs, self.channel, 224, 224)
        self.w_shape = (self.channel, self.channel, 3, 3)
        self.b_shape = self.channel

        self.x = numpy.random.uniform(-1, 1, self.x_shape).astype(self.dtype)
        self.x = ideep4py.mdarray(self.x)
        self.w = numpy.random.uniform(-1, 1, self.w_shape).astype(self.dtype)
        self.w = ideep4py.mdarray(self.w)
        self.b = numpy.random.uniform(-1, 1, self.b_shape).astype(self.dtype)
        self.b = ideep4py.mdarray(self.b)

        self.cp = convolution2DParam(self.x_shape,
                                     1, 1,
                                     1, 1,
                                     1, 1,
                                     1, 1)

        stride = 1
        pad = 1
        dilate = 1
        self.sy, self.sx = stride, stride
        self.ph, self.pw = pad, pad
        self.n = self.x_shape[0]
        self.outc = self.w_shape[0]
        self.outh = self.x_shape[2]
        self.outw = self.x_shape[3]
        self.cover_all = self.cover_all
        self.dy, self.dx = dilate, dilate

        self.gy = numpy.random.uniform(
            -1, 1,
            (self.n, self.outc, self.outh, self.outw)).astype(self.dtype)
        self.gy = ideep4py.mdarray(self.gy)

        self.check_forward_options = {'atol': 1e-3, 'rtol': 1e-2}
        self.check_backward_options = {'atol': 1e-3, 'rtol': 1e-2}

    def check_forward(self, x, w, b, cp):
        if self.with_bias:
            y_act = convolution2D.Forward(x, w, b, cp)
        else:
            y_act = convolution2D.Forward(x, w, None, cp)
        y_act = numpy.array(y_act, dtype=self.dtype)

        x = numpy.array(x, dtype=self.dtype)
        w = numpy.array(w, dtype=self.dtype)
        b = numpy.array(b, dtype=self.dtype)
        kh, kw = w.shape[2:]
        col = im2col_cpu(
            x, kh, kw, self.sy, self.sx, self.ph, self.pw,
            cover_all=self.cover_all, dy=self.dy, dx=self.dx)
        y = numpy.tensordot(
            col, w, ((1, 2, 3), (1, 2, 3))).astype(x.dtype, copy=False)
        if b is not None:
            y += b
        y_expect = numpy.rollaxis(y, 3, 1)
        numpy.testing.assert_allclose(
            y_act, y_expect, **self.check_forward_options)

    def test_forward_cpu(self):
        self.check_forward(self.x, self.w, self.b, self.cp)

    def check_backward_weights(self, x, w, b, cp, gy):
        gW_act, gB_act = convolution2D.BackwardWeightsBias(x, gy, cp)
        gW_act = numpy.array(gW_act, dtype=self.dtype)

        x = numpy.array(x, dtype=self.dtype)
        w = numpy.array(w, dtype=self.dtype)
        b = numpy.array(b, dtype=self.dtype)
        gy = numpy.array(gy, dtype=self.dtype)
        kh, kw = w.shape[2:]
        col = im2col_cpu(
            x, kh, kw, self.sy, self.sx, self.ph, self.pw,
            cover_all=self.cover_all, dy=self.dy, dx=self.dx)

        gW_expect = numpy.tensordot(
            gy, col, ((0, 2, 3), (0, 4, 5))).astype(self.dtype, copy=False)
        numpy.testing.assert_allclose(
            gW_act, gW_expect, **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu_weights(self):
        print("test_backward_cpu_weights")
        cp = convolution2DParam(self.w_shape,
                                1, 1,
                                1, 1,
                                1, 1,
                                1, 1)

        self.check_backward_weights(self.x, self.w, self.b, cp, self.gy)

    def check_backward_data(self, x, w, b, cp):
        out_c, in_c, kh, kw = w.shape
        n, out_c, in_h, in_w = x.shape
        self.pd = self.sy * (in_h - 1) + (
            kh + (kh - 1) * (self.dy - 1)) - self.outh - self.ph
        self.pr = self.sx * (in_w - 1) + (
            kw + (kw - 1) * (self.dx - 1)) - self.outw - self.pw

        _set_cover_all(self, x, w)
        # create conv parameter
        # for IA specific
        param = convolution2DParam(x.shape,
                                   self.dy, self.dx,
                                   self.sy, self.sx,
                                   self.ph, self.pw,
                                   self.pd, self.pr)
        y_act = convolution2D.BackwardData(w, x, param)
        if b is not None:
            y_act += b.reshape(1, b.size, 1, 1)
        y_act = numpy.array(y_act, dtype=self.dtype)

        x = numpy.array(x, dtype=self.dtype)
        w = numpy.array(w, dtype=self.dtype)

        gcol = numpy.tensordot(w, x, (0, 1)).astype(x.dtype, copy=False)
        # - k, m, n: shape of out_channel
        # - b: number of inputs
        # - h, w: height and width of kernels
        # k, m, n, b, h, w -> b, k, m, n, h, w
        gcol = numpy.rollaxis(gcol, 3)
        y_expect = col2im_cpu(
            gcol, self.sy, self.sx, self.ph, self.pw, self.outh, self.outw,
            dy=self.dy, dx=self.dx)
        # b, k, h, w
        if b is not None:
            y_expect += b.reshape(1, b.size, 1, 1)

        numpy.testing.assert_allclose(
            y_act, y_expect, **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu_data(self):
        print("test_backward_cpu_data")
        self.check_backward_data(self.x, self.w, self.b, self.cp)


testing.run_module(__name__, __file__)
