import sys
import unittest

import numpy

from ideep4py import array
from ideep4py import pooling2DParam
from ideep4py import convolution2DParam
from ideep4py import pooling2D
from ideep4py import convolution2D
from ideep4py import linear

try:
    import testing
    from testing import condition
    from testing.conv import col2im_cpu
    from testing.conv import im2col_cpu
except Exception as ex:
    print('*** testing directory is missing: %s' % ex)
    sys.exit(-1)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32],
    'bs': [256],
    'ic': [256],
    'oc': [256],
}))
class TestPooling2DPyF32(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(
            -1, 1, (self.bs, self.ic, 13, 13)).astype(self.dtype)
        self.W = numpy.random.uniform(
            -1, 1, (self.oc, self.ic, 3, 3)).astype(self.dtype)

        self.linear_gy = numpy.random.uniform(
            -1, 1, (self.bs, 4096)).astype(self.dtype)
        self.linear_W = \
            numpy.random.uniform(-1, 1, (4096, 9216)).astype(self.dtype)

        self.cp = convolution2DParam(
            (self.bs, self.oc, 13, 13), 1, 1, 1, 1, 1, 1, 1, 1)

        self.pp_fwd = pooling2DParam(
            (self.bs, self.oc, 6, 6), 3, 3, 2, 2, 0, 0, 0, 0,
            pooling2DParam.pooling_max)
        self.pp_bwd = pooling2DParam(
            self.x.shape, 3, 3, 2, 2, 0, 0, 0, 0,
            pooling2DParam.pooling_max)

        self.check_forward_options = {'atol': 1e-5, 'rtol': 1e-4}
        self.check_backward_options = {'atol': 1e-5, 'rtol': 1e-4}

    def check_forward(self, x, W, cp, pp):
        x_in = convolution2D.Forward(array(x), array(W), None, cp)
        y_act, self.indexes_act = pooling2D.Forward(x_in, pp)

        x_in_npy = numpy.array(x_in, dtype=self.dtype)
        y_act_npy = numpy.array(y_act, dtype=self.dtype)
        indexes_act_npy = numpy.array(self.indexes_act, dtype=self.dtype)

        col = im2col_cpu(
            x_in_npy, 3, 3, 2, 2, 0, 0,
            pval=-float('inf'), cover_all=True)
        n, c, kh, kw, out_h, out_w = col.shape
        col = col.reshape(n, c, kh * kw, out_h, out_w)
        self.indexes_ref = col.argmax(axis=2)
        y_ref = col.max(axis=2)

        numpy.testing.assert_allclose(
            y_act_npy, y_ref, **self.check_forward_options)
        numpy.testing.assert_allclose(
            indexes_act_npy, self.indexes_ref, **self.check_forward_options)

    def check_backward(self, gy, W, pp):
        gy_in = linear.BackwardData(array(W), array(gy))
        gy_in = gy_in.reshape(self.bs, self.oc, 6, 6)
        gx_act = pooling2D.Backward(gy_in, self.indexes_act, pp)

        gy_in_npy = numpy.array(gy_in, dtype=self.dtype)
        gx_act_npy = numpy.array(gx_act, dtype=self.dtype)

        n, c, out_h, out_w = gy_in_npy.shape
        h = 13
        w = 13
        kh = 3
        kw = 3
        gcol = numpy.zeros(
            (n * c * out_h * out_w * 3 * 3), dtype=self.dtype)
        indexes = self.indexes_ref.flatten()
        indexes += numpy.arange(0, indexes.size * kh * kw, kh * kw)
        gcol[indexes] = gy_in.ravel()
        gcol = gcol.reshape(n, c, out_h, out_w, kh, kw)
        gcol = numpy.swapaxes(gcol, 2, 4)
        gcol = numpy.swapaxes(gcol, 3, 5)
        gx_ref = col2im_cpu(gcol, 2, 2, 0, 0, h, w)

        numpy.testing.assert_allclose(
            gx_act_npy, gx_ref, **self.check_backward_options)

    @condition.retry(3)
    def test_alexnet_max_pooling_3_cpu(self):
        self.check_forward(self.x, self.W, self.cp, self.pp_fwd)
        self.check_backward(self.linear_gy, self.linear_W, self.pp_bwd)


testing.run_module(__name__, __file__)
