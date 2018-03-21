import sys
import unittest

import numpy
import ideep4py
from ideep4py import dropout

try:
    import testing
except Exception as ex:
    print('*** testing directory is missing: %s' % ex)
    sys.exit(-1)


def _dropout(x, creator):
    return x * creator.mask


@testing.parameterize(*testing.product({
    'dropout_ratio': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'dtype': [numpy.float32, ],
}))
@testing.fix_random()
class TestDropoutF32(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.rand(128, 3, 224, 224).astype(self.dtype)
        self.x_md = ideep4py.mdarray(self.x)
        self.gy = numpy.random.rand(128, 3, 224, 224).astype(self.dtype)

    def check_forward(self, x, x_md):
        mask, y = dropout.Forward(x_md, self.dropout_ratio)
        y = numpy.array(y, dtype=self.dtype)
        y_expect = x * mask
        numpy.testing.assert_allclose(y, y_expect)

    def check_backward(self, x_md, gy):
        mask, y = dropout.Forward(x_md, self.dropout_ratio)
        gy_md = ideep4py.mdarray(gy)
        gx = dropout.Backward(mask, gy_md)
        gx = numpy.array(gx, dtype=self.dtype)
        gx_expect = gy * mask
        numpy.testing.assert_allclose(gx, gx_expect)

    def test_forward_cpu(self):
        self.check_forward(self.x, self.x_md)

    def test_backward_cpu(self):
        self.check_backward(self.x_md, self.gy)


testing.run_module(__name__, __file__)
