import numpy
from chainer import testing
import ideep4py

# x = numpy.ndarray(shape=(1,32,224,224), dtype=numpy.float32, order='C')
x = numpy.random.uniform(-1, 1, (1, 32, 2, 224)).astype(numpy.float32)
y = numpy.tanh(x)

mx = ideep4py.mdarray(x)
x2 = numpy.array(mx)
testing.assert_allclose(x, x2)

print("tanh fwd")
my = ideep4py._ideep4py.tanh.Forward(mx)
y2 = numpy.array(my)
testing.assert_allclose(y, y2)

# Test backward
print("tanh bwd")
x = numpy.random.uniform(-1, 1, (1, 32, 224, 224)).astype(numpy.float32)
gy = numpy.random.uniform(-1, 1, (1, 32, 224, 224)).astype(numpy.float32)
gx = gy * (1 - numpy.tanh(x) ** 2)


mx = ideep4py.mdarray(x)
mgy = ideep4py.mdarray(gy)
mgx = ideep4py._ideep4py.tanh.Backward(mx, mgy)

gx1 = numpy.array(mgx)
testing.assert_allclose(gx1, gx)
