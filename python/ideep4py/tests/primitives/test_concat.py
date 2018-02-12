import numpy
import ideep4py

# from dnn._dnn import convolution2DParam, conv_test
from ideep4py import intVector, mdarrayVector, concat

x1 = numpy.ndarray(shape=(1, 16, 224, 224), dtype=numpy.float32, order='C')
x2 = numpy.ndarray(shape=(1, 32, 224, 224), dtype=numpy.float32, order='C')
x3 = numpy.ndarray(shape=(1, 64, 224, 224), dtype=numpy.float32, order='C')
inputs = (x1, x2, x3)
sizes = numpy.array(
    [v.shape[1] for v in inputs[:-1]]
).cumsum()
print("sizes=", sizes)
print("type=", type(sizes))

x1 = ideep4py.mdarray(x1)
x2 = ideep4py.mdarray(x2)
x3 = ideep4py.mdarray(x3)

xs = mdarrayVector()
xs.push_back(x1)
xs.push_back(x2)
xs.push_back(x3)

print("fwd")
y = concat.Forward(xs, 1)
print("==============")
y = concat.Forward(xs, 1)
print("y.shape=", y.shape)

print("backward")

int_sizes = intVector()

for i in sizes:
    print("i=", i)
    int_sizes.push_back(i)

gxs = concat.Backward(y, int_sizes, 1)

for gx in gxs:
    print("gx.type=", type(gx))
    print("gx.shape=", gx.shape)
print("after backward")
