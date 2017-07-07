import numpy
import ideep4py

# list case
x1 = numpy.ndarray(shape=(2, 2, 2, 2), dtype=numpy.float32, order='C')
x = ideep4py.mdarray(x1)
y1 = x1.reshape([4, 4])
y = x.reshape([4, 4])
res = numpy.allclose(y, y1, 1e-5, 1e-4)
if res is not True:
    print("error!!!!")

# singal number case
x1 = numpy.ndarray(shape=(2, 2, 2, 2), dtype=numpy.float32, order='C')
x = ideep4py.mdarray(x1)
x1.reshape(16)
x.reshape(16)
res = numpy.allclose(y, y1, 1e-5, 1e-4)
if res is not True:
    print("error!!!!")

# value change
x1 = numpy.ndarray(shape=(2, 2, 2, 2), dtype=numpy.float32, order='C')
x = ideep4py.mdarray(x1)
print(type(x))
y = x.reshape(len(x), -1)
x[0, 0, 0, 0] = 3.333
assert(x[0, 0, 0, 0] == y[0, 0])

y = x.reshape((len(x), -1))
x[0, 0, 0, 0] = 4.4444
assert(x[0, 0, 0, 0] == y[0, 0])

# -1 case
x1 = numpy.ndarray(shape=(2, 2, 2, 2), dtype=numpy.float32, order='C')
x = ideep4py.mdarray(x1)
y = x.reshape((2, 2, -1))
y1 = x1.reshape((2, 2, -1))
res = numpy.allclose(y, y1, 1e-5, 1e-4)
if res is not True:
    print("error!!!!")
y = x.reshape(2, 2, -1)
numpy.allclose(y, y1, 1e-5, 1e-4)
if res is not True:
    print("error!!!!")
