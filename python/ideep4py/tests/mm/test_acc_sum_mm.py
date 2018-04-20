import sys
import unittest
import numpy
import ideep4py
try:
    import testing
except Exception as ex:
    print('*** testing directory is missing: %s' % ex)
    sys.exit(-1)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32],
    'shape': [(3, 16, 2, 4), (2, 7, 1, 1), (2, 7, 3, 2), ],
}))
class TestAccSum(unittest.TestCase):
    def setUp(self):
        self.x1 = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.x2 = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.x3 = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.x4 = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.check_options = {'atol': 1e-5, 'rtol': 1e-4}

    def test_sum(self):
        mx1 = ideep4py.mdarray(self.x1)
        mx2 = ideep4py.mdarray(self.x2)
        mx3 = ideep4py.mdarray(self.x3)
        mx4 = ideep4py.mdarray(self.x4)
        x = self.x1 + self.x2 + self.x3 + self.x4
        mx = ideep4py.basic_acc_sum((mx1, mx2, mx3, mx4))
        # mx = numpy.asarray(mx)
        numpy.testing.assert_allclose(mx, x, **self.check_options)

    def test_multi_add(self):
        mx1 = ideep4py.mdarray(self.x1)
        mx2 = ideep4py.mdarray(self.x2)
        x = self.x1 + self.x2 + self.x3 + self.x4
        mx = ideep4py.multi_add((mx1, mx2, self.x3, self.x4))
        numpy.testing.assert_allclose(mx, x, **self.check_options)


testing.run_module(__name__, __file__)
