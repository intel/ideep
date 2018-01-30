import ideep4py  # NOQA
import numpy
from chainer import testing
from ideep4py import relu, mdarray


print('mdarray sum [common cases 1]')
print('shape (2, 3, 2, 2) along (0, 2, 3)')
x = numpy.arange(24, dtype=numpy.float32)
x = x.reshape((2, 3, 2, 2))
mx = mdarray(x)
testing.assert_allclose(mx.sum((0, 2, 3)), x.sum((0, 2, 3)))
print('pass ...\n')


print('mdarray sum [common cases 2]')
print('shape (2, 3, 2, 2) along (1, 2)')
x = numpy.arange(24, dtype=numpy.float32)
x = x.reshape((2, 3, 2, 2))
mx = mdarray(x)
testing.assert_allclose(mx.sum((1, 2)), x.sum((1, 2)))
print('pass ...\n')


print('mdarray sum [common cases 3]')
print('shape (2, 3, 2, 2) along (0, 2)')
x = numpy.arange(24, dtype=numpy.float32)
x = x.reshape((2, 3, 2, 2))
mx = mdarray(x)
testing.assert_allclose(mx.sum((0, 2)), x.sum((0, 2)))
print('pass ...\n')


print('mdarray sum [common cases 4]')
print('shape (2, 3, 2, 2) along (1, 3)')
x = numpy.arange(24, dtype=numpy.float32)
x = x.reshape((2, 3, 2, 2))
mx = mdarray(x)
testing.assert_allclose(mx.sum((1, 3)), x.sum((1, 3)))
print('pass ...\n')


print('mdarray sum [common cases 5]')
print('shape (2, 3, 2, 2) along (0)')
x = numpy.arange(24, dtype=numpy.float32)
x = x.reshape((2, 3, 2, 2))
mx = mdarray(x)
testing.assert_allclose(mx.sum((0)), x.sum((0)))
print('pass ...\n')


print('mdarray sum [common cases 6]')
print('shape (2, 3, 2, 2) along (1, 2, 3)')
x = numpy.arange(24, dtype=numpy.float32)
x = x.reshape((2, 3, 2, 2))
mx = mdarray(x)
testing.assert_allclose(mx.sum((1, 2, 3)), x.sum((1, 2, 3)))
print('pass ...\n')


print('mdarray sum [common cases 7]')
print('shape (2, 3, 2, 2) along (0, 1, 2)')
x = numpy.arange(24, dtype=numpy.float32)
x = x.reshape((2, 3, 2, 2))
mx = mdarray(x)
testing.assert_allclose(mx.sum((0, 1, 2)), x.sum((0, 1, 2)))
print('pass ...\n')


print('mdarray sum [common cases 8]')
print('shape (2, 3, 2, 2) along (3)')
x = numpy.arange(24, dtype=numpy.float32)
x = x.reshape((2, 3, 2, 2))
mx = mdarray(x)
testing.assert_allclose(mx.sum((3)), x.sum((3)))
print('pass ...\n')


print('mdarray sum [common cases 9]')
print('shape (2, 3, 2, 2) along (1)')
x = numpy.arange(24, dtype=numpy.float32)
x = x.reshape((2, 3, 2, 2))
mx = mdarray(x)
testing.assert_allclose(mx.sum((1)), x.sum((1)))
print('pass ...\n')


print('mdarray sum [internal cases 1]')
print('shape (256, 384, 13, 13) along (0, 2, 3)')
x = numpy.random.rand(256, 384, 13, 13)
x = x.astype(numpy.float32)
y = numpy.maximum(x, 0, dtype=numpy.float32)
mx = mdarray(x)
my = relu.Forward(mx)
testing.assert_allclose(my.sum((0, 2, 3)), y.sum((0, 2, 3)))
print('pass ...\n')


print('mdarray sum [internal cases 2]')
print('shape (256, 384, 13, 13) along (1, 2, 3)')
x = numpy.random.rand(256, 384, 13, 13)
x = x.astype(numpy.float32)
y = numpy.maximum(x, 0, dtype=numpy.float32)
mx = mdarray(x)
my = relu.Forward(mx)
testing.assert_allclose(my.sum((1, 2, 3)), y.sum((1, 2, 3)))
print('pass ...\n')


print('mdarray sum [internal cases 3]')
print('shape (256, 384, 13, 13) along (0, 1, 2)')
x = numpy.random.rand(256, 384, 13, 13)
x = x.astype(numpy.float32)
y = numpy.maximum(x, 0, dtype=numpy.float32)
mx = mdarray(x)
my = relu.Forward(mx)
testing.assert_allclose(my.sum((0, 1, 2)), y.sum((0, 1, 2)))
print('pass ...\n')


print('mdarray sum [internal cases 4]')
print('shape (256, 384, 13, 13) along (0, 2)')
x = numpy.random.rand(256, 384, 13, 13)
x = x.astype(numpy.float32)
y = numpy.maximum(x, 0, dtype=numpy.float32)
mx = mdarray(x)
my = relu.Forward(mx)
testing.assert_allclose(my.sum((0, 2)), y.sum((0, 2)))
print('pass ...\n')


print('mdarray sum [internal cases 5]')
print('shape (256, 384, 13, 13) along (1, 3)')
x = numpy.random.rand(256, 384, 13, 13)
x = x.astype(numpy.float32)
y = numpy.maximum(x, 0, dtype=numpy.float32)
mx = mdarray(x)
my = relu.Forward(mx)
testing.assert_allclose(my.sum((1, 3)), y.sum((1, 3)))
print('pass ...\n')


print('mdarray sum [internal cases 6]')
print('shape (256, 384, 13, 13) along (1, 2)')
x = numpy.random.rand(256, 384, 13, 13)
x = x.astype(numpy.float32)
y = numpy.maximum(x, 0, dtype=numpy.float32)
mx = mdarray(x)
my = relu.Forward(mx)
testing.assert_allclose(my.sum((1, 2)), y.sum((1, 2)))
print('pass ...\n')


print('mdarray sum [internal cases 7]')
print('shape (256, 384, 13, 13) along (0)')
x = numpy.random.rand(256, 384, 13, 13)
x = x.astype(numpy.float32)
y = numpy.maximum(x, 0, dtype=numpy.float32)
mx = mdarray(x)
my = relu.Forward(mx)
testing.assert_allclose(my.sum((0)), y.sum((0)))
print('pass ...\n')


print('mdarray sum [internal cases 8]')
print('shape (256, 384, 13, 13) along (3)')
x = numpy.random.rand(256, 384, 13, 13)
x = x.astype(numpy.float32)
y = numpy.maximum(x, 0, dtype=numpy.float32)
mx = mdarray(x)
my = relu.Forward(mx)
testing.assert_allclose(my.sum((3)), y.sum((3)))
print('pass ...\n')


print('mdarray sum [internal cases 9]')
print('shape (256, 384, 13, 13) along (2)')
x = numpy.random.rand(256, 384, 13, 13)
x = x.astype(numpy.float32)
y = numpy.maximum(x, 0, dtype=numpy.float32)
mx = mdarray(x)
my = relu.Forward(mx)
testing.assert_allclose(my.sum((2)), y.sum((2)))
print('pass ...\n')


print('mdarray sum [small shape routine]')
print('shape (39, 32, 13, 13) along (0, 2, 3)')
x = numpy.random.rand(39, 32, 13, 13)
x = x.astype(numpy.float32)
y = numpy.maximum(x, 0, dtype=numpy.float32)
mx = mdarray(x)
my = relu.Forward(mx)
testing.assert_allclose(my.sum((0, 2, 3)), y.sum((0, 2, 3)))
print('pass ...\n')


print('mdarray sum [mkldnn format keepdims routine]')
print('shape (39, 32, 13, 13) along (0, 2, 3)')
x = numpy.random.rand(39, 32, 13, 13)
x = x.astype(numpy.float32)
y = numpy.maximum(x, 0, dtype=numpy.float32)
mx = mdarray(x)
my = relu.Forward(mx)
testing.assert_allclose(my.sum((0, 2, 3), keepdims=True),
                        y.sum((0, 2, 3), keepdims=True))
print('pass ...\n')


print('mdarray sum [common format big shape routine]')
print('shape (256, 385, 13, 13) along (0, 2, 3)')
x = numpy.random.rand(256, 385, 13, 13)
x = x.astype(numpy.float32)
mx = mdarray(x)
ms = mx.sum((0, 2, 3))
ns = x.sum((0, 2, 3))
testing.assert_allclose(ms, ns)
print('pass ...\n')


print('mdarray sum [common format big shape routine]')
print('shape (256, 1000) along (0)')
x = numpy.random.rand(256, 1000)
x = x.astype(numpy.float32)
mx = mdarray(x)
ms = mx.sum((0))
ns = x.sum((0))
testing.assert_allclose(ms, ns)
print('pass ...\n')


print('mdarray sum [common format big shape routine]')
print('shape (256, 1000) along (1)')
x = numpy.random.rand(256, 1000)
x = x.astype(numpy.float32)
mx = mdarray(x)
ms = mx.sum((1))
ns = x.sum((1))
testing.assert_allclose(ms, ns)
print('pass ...\n')
