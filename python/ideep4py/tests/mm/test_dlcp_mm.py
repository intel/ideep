import sys
import unittest
import numpy
import ideep4py
from ideep4py import dlCompression
try:
    import testing
except Exception as ex:
    print('*** testing directory is missing: %s' % ex)
    sys.exit(-1)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32],
    'shape': [(3, 16, 2, 4), (2, 7, 1, 1), (2, 7, 3, 2), ],
}))
class TestDlcp(unittest.TestCase):
    def setUp(self):
        self.a = numpy.arange(9, dtype=self.dtype)
        self.a = self.a.reshape((3, 3))
        self.check_options = {'atol': 1e-5, 'rtol': 1e-4}

    @unittest.skip("demonstrating skipping, not support yes")
    def test_dlcp(self):
        am = ideep4py.array(self.a)
        ret = dlCompression.Compress(
            am, am, None, 4, dlCompression.dl_comp_dfp)
        assert(ret == dlCompression.dl_comp_ok)

        ret = dlCompression.Decompress(am, am)
        assert(ret == dlCompression.dl_comp_o)
        _a = numpy.array(am)
        numpy.testing.assert_allclose(am, _a, **self.check_options)
