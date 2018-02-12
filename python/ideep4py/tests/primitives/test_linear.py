import numpy
import ideep4py
# from ideep4py import linearParam, linear_test
from ideep4py import linear

x = numpy.ndarray(shape=(1, 32), dtype=numpy.float32, order='C')
x = ideep4py.mdarray(x)

w = numpy.ndarray(shape=(32, 32), dtype=numpy.float32, order='C')
print("ndarray w", w.shape)
w = ideep4py.mdarray(w)
print("w.dim", w.shape)
b = numpy.ndarray(shape=(32,), dtype=numpy.float32, order='C')
b = ideep4py.mdarray(b)

print("===============2 dims============")

print("fwd")
y = linear.Forward(x, w, b)
print("================")
y = linear.Forward(x, w, b)
print("================")
y = linear.Forward(x, w, b)

print("bwd data")
x = linear.BackwardData(w, y)
print("================")
x = linear.BackwardData(w, y)
print("================")
x = linear.BackwardData(w, y)
print("================")

print("bwd weight bias")
weights = linear.BackwardWeightsBias(x, y)
print("weights= ", type(weights))
print("len", len(weights))
print("gw.shape", weights[0].shape)
print("gb.shape = ", weights[1].shape)
print("================")

x = numpy.ndarray(shape=(1, 32), dtype=numpy.float32, order='C')
x = ideep4py.mdarray(x)
weights = linear.BackwardWeightsBias(x, y)
print("weights= ", type(weights))
print("len", len(weights))
print("gw.shape", weights[0].shape)
print("gb.shape = ", weights[1].shape)
print("================")

x = numpy.ndarray(shape=(1, 32), dtype=numpy.float32, order='C')
x = ideep4py.mdarray(x)
weights = linear.BackwardWeightsBias(x, y)
print("weights= ", type(weights))
print("len", len(weights))
print("gw.shape", weights[0].shape)
print("gb.shape = ", weights[1].shape)
print("================")

print("bwd weight")
weights = linear.BackwardWeights(x, y)
print("weights= ", type(weights))
print("gw.shape", weights.shape)
print("================")

x = numpy.ndarray(shape=(1, 32), dtype=numpy.float32, order='C')
x = ideep4py.mdarray(x)
weights = linear.BackwardWeights(x, y)
print("weights= ", type(weights))
print("gw.shape", weights.shape)
print("================")

x = numpy.ndarray(shape=(1, 32), dtype=numpy.float32, order='C')
x = ideep4py.mdarray(x)
weights = linear.BackwardWeights(x, y)
print("weights= ", type(weights))
print("gw.shape", weights.shape)
print("================")

# print("==========4 dims=================")
#
# x = numpy.ndarray(shape=(1, 32, 224, 224), dtype=numpy.float32, order='C')
# x = ideep4py.mdarray(x)
#
# w = numpy.ndarray(shape=(32, 32, 224, 224), dtype=numpy.float32, order='C')
# print("ndarray w", w.shape)
# w = ideep4py.mdarray(w)
# print("w.dim", w.shape)
# b = numpy.ndarray(shape=(32,), dtype=numpy.float32, order='C')
# b = ideep4py.mdarray(b)
#
# print("fwd")
# y = linear.Forward(x, w, b)
# print("================")
# y = linear.Forward(x, w, b)
# print("================")
# y = linear.Forward(x, w, b)
#
# print("================")
# print("bwd data")
# x = linear.BackwardData(w, y)
