import ideep4py
from ideep4py import dlCompression

import numpy

a = numpy.arange(9, dtype=numpy.float32)
a = a.reshape((3, 3))
am = ideep4py.array(a)

ret = dlCompression.Compress(am, am, None, 4, dlCompression.dl_comp_dfp)
assert(ret == dlCompression.dl_comp_ok)

ret = dlCompression.Decompress(am, am)
assert(ret == dlCompression.dl_comp_ok)

_a = numpy.array(am)

numpy.testing.assert_allclose(a, _a, atol=0.1, rtol=0.01, verbose=True)
