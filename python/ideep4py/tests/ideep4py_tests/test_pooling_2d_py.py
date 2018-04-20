import os
import sys
import unittest

import numpy
import six

import ideep4py
from ideep4py import pooling2DParam
from ideep4py import pooling2D

try:
    import testing
    from testing import condition
    from testing.conv import col2im_cpu
except Exception as ex:
    print('*** testing directory is missing: %s' % ex)
    sys.exit(-1)

if bool(int(os.environ.get('ENALE_TRAVIS_TEST', '0'))):
    bs_list = [1, 2, 4, 5, 6, 8, 10, 16, 24, 32, 64, ]
else:
    bs_list = [1, 2, 4, 5, 6, 8, 10, 16, 24, 32, 64, 96, 128, 196, 256, ]
print('bs_list: ', bs_list)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32],
    'channel': [1, 2, 4, 8, 10, 16, 24, 32, 64, ],
    'bs': bs_list,
    'stride': [2, ],
}))
class TestPooling2DPyF32(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(
            -1, 1, (self.bs, self.channel, 4, 3)).astype(self.dtype)
        self.gy = numpy.random.uniform(
            -1, 1, (self.bs, self.channel, 2, 2)).astype(self.dtype)

        self.pp_fwd = pooling2DParam(
            self.gy.shape, 3, 3, self.stride, self.stride, 1, 1,
            1, 1, pooling2DParam.pooling_avg_include_padding)
        self.pp_bwd = pooling2DParam(
            (self.bs, self.channel, 4, 3), 3, 3, self.stride, self.stride,
            1, 1, 1, 1, pooling2DParam.pooling_avg_include_padding)

        self.check_forward_options = {'atol': 1e-5, 'rtol': 1e-4}
        self.check_backward_options = {'atol': 1e-5, 'rtol': 1e-4}

    def check_forward(self, x, pp):
        x_mdarray = ideep4py.mdarray(x)
        (y_act,) = pooling2D.Forward(x_mdarray, pp)
        y_act = numpy.array(y_act, dtype=self.dtype)

        for k in six.moves.range(self.bs):
            for c in six.moves.range(self.channel):
                x = self.x[k, c]
                expect = numpy.array([
                    [x[0:2, 0:2].sum(), x[0:2, 1:3].sum()],
                    [x[1:4, 0:2].sum(), x[1:4, 1:3].sum()]]) / 9
                numpy.testing.assert_allclose(
                    expect, y_act[k, c], **self.check_forward_options)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x, self.pp_fwd)

    def check_backward(self, x, gy, pp):
        # self.shape[2:]
        h, w = 4, 3
        gcol = numpy.tile(gy[:, :, None, None],
                          (1, 1, 3, 3, 1, 1))
        gx_expect = col2im_cpu(gcol, 2, 2, 1, 1, h, w)
        gx_expect /= 3 * 3
        gy_mdarray = ideep4py.mdarray(gy)
        x_mdarray = ideep4py.mdarray(x)
        gx_act = pooling2D.Backward(x_mdarray, gy_mdarray, None, pp)
        gx_act = numpy.array(gx_act, dtype=self.dtype)

        numpy.testing.assert_allclose(
            gx_expect, gx_act, **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy, self.pp_bwd)


testing.run_module(__name__, __file__)
