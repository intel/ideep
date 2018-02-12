import numpy
import ideep4py
x1 = numpy.ndarray(shape=(1, 2, 3, 4), dtype=numpy.float32, order='C')
x = ideep4py.mdarray(x1)
x2 = numpy.array(x)
print("x = ", x1)
print("x2 = ", x2)
