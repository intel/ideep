import numpy
from chainer import testing
import ideep4py
from ideep4py import relu

# x = numpy.ndarray(shape=(1,32,224,224), dtype=numpy.float32, order='C')
x = numpy.random.uniform(-1, 1, (1, 32, 224, 224)).astype(numpy.float32)
y = numpy.maximum(x, 0, dtype=x.dtype)

mx = ideep4py.mdarray(x)
x2 = numpy.array(mx)
testing.assert_allclose(x, x2)

print("Relu fwd")
my = relu.Forward(mx)
y2 = numpy.array(my)
testing.assert_allclose(y, y2)
my = relu.Forward(my)
y2 = numpy.array(my)
testing.assert_allclose(y, y2)


# Test backward
print("Relu bwd")
x = numpy.random.uniform(-1, 1, (1, 32, 224, 224)).astype(numpy.float32)
gy = numpy.random.uniform(-1, 1, (1, 32, 224, 224)).astype(numpy.float32)
gx = (x > 0) * gy


mx = ideep4py.mdarray(x)
mgy = ideep4py.mdarray(gy)
mgx = relu.Backward(mx, mgy)


gx1 = numpy.array(mgx)
testing.assert_allclose(gx1, gx)
