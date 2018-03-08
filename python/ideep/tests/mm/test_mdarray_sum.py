import ideep4py  # NOQA
import numpy
from chainer import testing
from ideep4py import relu, mdarray

print('mdarray sum [larg shape routine]')
print('shape (256, 384, 13, 13) along (0, 2, 3)')
x = numpy.ndarray((256, 384, 13, 13), dtype=numpy.float32)
y = numpy.maximum(x, 0, dtype=x.dtype)

mx = mdarray(x)
my = relu.Forward(mx)

testing.assert_allclose(my.sum((0, 2, 3)), y.sum((0, 2, 3)))
print('pass ...\n')


print('mdarray sum [small shape routine]')
print('shape (39, 32, 13, 13) along (0, 2, 3)')
x = numpy.ndarray((39, 32, 13, 13), dtype=numpy.float32)
y = numpy.maximum(x, 0, dtype=x.dtype)

mx = mdarray(x)
my = relu.Forward(mx)

testing.assert_allclose(my.sum((0, 2, 3)), y.sum((0, 2, 3)))
print('pass ...\n')


print('mdarray sum [mkldnn format keepdims routine]')
print('shape (39, 32, 13, 13) along (0, 2, 3)')
x = numpy.ndarray((39, 32, 13, 13), dtype=numpy.float32)
y = numpy.maximum(x, 0, dtype=x.dtype)

mx = mdarray(x)
my = relu.Forward(mx)

testing.assert_allclose(my.sum((0, 2, 3), keepdims=True),
                        y.sum((0, 2, 3), keepdims=True))
print('pass ...\n')


print('mdarray sum [common format small shape routine]')
print('shape (2, 2, 3, 3) along (0, 2, 3)')
x = numpy.ndarray((2, 2, 3, 3), dtype=numpy.float32)

x.fill(2.3232)
x[0].fill(3.1212)
mx = mdarray(x)

testing.assert_allclose(mx.sum((0, 2, 3)), x.sum((0, 2, 3)))
print('pass ...\n')


print('mdarray sum [common format small shape routine]')
print('shape (2, 2, 3, 3) along (1, 3)')
x = numpy.ndarray((2, 2, 3, 3), dtype=numpy.float32)

x.fill(2.3232)
x[0].fill(3.1212)
mx = mdarray(x)

testing.assert_allclose(mx.sum((1, 3)), x.sum((1, 3)))
print('pass ...\n')


print('mdarray sum [common format routine keepdims]')
print('shape (2, 2, 3, 3) along (0, 2, 3)')
x = numpy.ndarray((2, 2, 3, 3), dtype=numpy.float32)

x.fill(2.3232)
x[0].fill(3.1212)
mx = mdarray(x)

ms = mx.sum((0, 2, 3), keepdims=True)
ns = x.sum((0, 2, 3), keepdims=True)
testing.assert_allclose(ms, ns)
print('pass ...\n')


print('mdarray sum [common format routine]')
print('shape (2, 15, 3, 3) along (0, 2, 3)')
x = numpy.ndarray((2, 15, 3, 3), dtype=numpy.float32)

x.fill(1)
x[0].fill(3.1212)
mx = mdarray(x)

ms = mx.sum((0, 2, 3))
ns = x.sum((0, 2, 3))
testing.assert_allclose(ms, ns)
print('pass ...\n')


print('mdarray sum [common format big shape routine]')
print('shape (256, 385, 13, 13) along (0, 2, 3)')
x = numpy.ndarray((256, 385, 13, 13), dtype=numpy.float32)

x.fill(1)
x[0].fill(3.1212)
mx = mdarray(x)

ms = mx.sum((0, 2, 3))
ns = x.sum((0, 2, 3))
testing.assert_allclose(ms, ns)
print('pass ...\n')


print('mdarray sum [common format big shape routine]')
print('shape (256, 1000) along (0)')
x = numpy.ndarray((256, 1000), dtype=numpy.float32)

x.fill(1)
x[0].fill(3.1212)
mx = mdarray(x)

ms = mx.sum((0))
ns = x.sum((0))
testing.assert_allclose(ms, ns)
print('pass ...\n')

print('mdarray sum [common format big shape routine]')
print('shape (256, 1000) along (1)')
x = numpy.ndarray((256, 1000), dtype=numpy.float32)

x.fill(1)
x[0].fill(3.1212)
mx = mdarray(x)

ms = mx.sum((1))
ns = x.sum((1))
testing.assert_allclose(ms, ns)
print('pass ...\n')
