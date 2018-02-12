import numpy
from chainer import testing  # NOQA
from chainer import utils  # NOQA
import ideep4py

x1 = numpy.ndarray(shape=(2, 2), dtype=numpy.float32, order='C')
x = ideep4py.mdarray(x1)
print(x1)
y = x1 > 0
print(y)
x *= y


# test devide
x1 = numpy.ndarray(shape=(2, 2), dtype=numpy.float32, order='C')
x1.fill(2.)
x = ideep4py.mdarray(x1)
testing.assert_allclose(1 / x1, 1 / x)
testing.assert_allclose(2 * x1, 2 * x)
testing.assert_allclose(1 - x1, 1 - x)
testing.assert_allclose(1 + x1, 1 + x)

x1 /= 3
x /= 3
testing.assert_allclose(x1, x)

x1 *= 2
x *= 2
testing.assert_allclose(x1, x)

x1 += 3
x += 3
testing.assert_allclose(x1, x)

x1 -= 5
x -= 5
testing.assert_allclose(x1, x)
