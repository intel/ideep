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
class TestCopyto(unittest.TestCase):
    def setUp(self):
        self.x1 = numpy.ndarray(shape=self.shape, dtype=self.dtype, order='C')
        self.x2 = numpy.ndarray(shape=self.shape, dtype=self.dtype, order='C')
        self.check_options = {'atol': 1e-5, 'rtol': 1e-4}

    def test_copyto1(self):
        mx1 = ideep4py.mdarray(self.x1)
        mx2 = ideep4py.mdarray(self.x2)
        numpy.copyto(self.x2, self.x1)
        ideep4py.basic_copyto(mx2, mx1)
        t = numpy.asarray(mx2)
        numpy.testing.assert_allclose(t, self.x2, **self.check_options)
        # numpy.allclose(t, x2, 1e-5, 1e-4, True)

    def test_copytoOne(self):
        mx2 = ideep4py.mdarray(self.x2)
        numpy.copyto(self.x2, self.x1)
        ideep4py.basic_copyto(mx2, self.x1)
        t = numpy.asarray(mx2)
        numpy.testing.assert_allclose(t, self.x2, **self.check_options)
        # numpy.allclose(t, x2, 1e-5, 1e-4, True)
