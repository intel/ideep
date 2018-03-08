import ideep4py  # NOQA
import numpy
import six
from chainer import testing
from ideep4py import relu, mdarray

# enumerate test
x = numpy.random.uniform(-1, 1, (256, 512, 13, 13)).astype(numpy.float32)
mx = mdarray(x)

a = []
b = []
for p, xi in enumerate(x):
    a.append(xi)
for p, mxi in enumerate(mx):
    b.append(mxi)

testing.assert_allclose(numpy.asarray(a), numpy.asarray(b))


# zip test
x1 = numpy.random.uniform(-1, 1, (256, 512, 13, 13)).astype(numpy.float32)
x2 = numpy.random.uniform(-1, 1, (256, 512, 13, 13)).astype(numpy.float32)

mx1 = mdarray(x1)
mx2 = mdarray(x2)

a1 = []
a2 = []
b1 = []
b2 = []

for x, y in six.moves.zip(x1, x2):
    a1.append(x)
    a2.append(y)

for mx, my in six.moves.zip(mx1, mx2):
    b1.append(mx)
    b2.append(my)

testing.assert_allclose(numpy.asarray(a1), numpy.asarray(b1))
testing.assert_allclose(numpy.asarray(a2), numpy.asarray(b2))


# mkl-dnn format test
x = numpy.random.uniform(-1, 1, (256, 512, 13, 13)).astype(numpy.float32)
y = numpy.maximum(x, 0, dtype=x.dtype)
mx = mdarray(x)
my = relu.Forward(mx)
testing.assert_allclose(y, my)

a = []
b = []
for p, xi in enumerate(y):
    a.append(xi)
for p, mxi in enumerate(my):
    b.append(mxi)

testing.assert_allclose(numpy.asarray(a), numpy.asarray(b))
