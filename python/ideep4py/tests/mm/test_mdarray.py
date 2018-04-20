import numpy
from chainer import testing
from chainer import utils  # NOQA
import ideep4py

# from memory_profiler import profile


# @profile
def run():
    print('[1]')
    x = numpy.ndarray(shape=(256, 3, 224, 224), dtype=numpy.float32, order='C')
    print('[2]')
    x = ideep4py.mdarray(x)
    print('[3]')
    x = ideep4py.mdarray(x)
    print('[4]')
    x = ideep4py.mdarray(x)
    print('[5]')
    x = ideep4py.mdarray(x)
    print('[6]')
    x = ideep4py.mdarray(x)
    print('[7]')
    x = ideep4py.mdarray(x)

    x1 = numpy.ndarray(shape=(2, 2, 2, 2), dtype=numpy.float32, order='C')
    x = ideep4py.mdarray(x1)
    x = x + 1
    testing.assert_allclose(x1 + 1, x)

    x = ideep4py.mdarray(x1)

    print(x)
    print("ndims=", x.ndim)
    print("shape=", x.shape)
    print("size=", x.size)
    print("dtype=", x.dtype)
    print("is_mdarry=", x.is_mdarray)

    x1 += x
    x += x
    x2 = numpy.array(x)
    testing.assert_allclose(x1, x2)

    x1 = numpy.ones(shape=(2, 2, 2, 2), dtype=numpy.float32, order='C')
    x = ideep4py.mdarray(x1)
    y = x + x1
    y2 = numpy.array(y)
    testing.assert_allclose(y2, x1 + x1)

    y = x * x1
    y2 = numpy.array(y)
    testing.assert_allclose(y2, x1 * x1)

    x1 = numpy.random.uniform(-1, 1, (3, 4)).astype(numpy.float32)
    x = ideep4py.mdarray(x1)
    z1 = (x1 > 0).astype(x1.dtype)
    z = (x > 0).astype(x1.dtype)
    testing.assert_allclose(z, z1)


run()
