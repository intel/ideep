import numpy
import ideep4py

x1 = numpy.random.uniform(-1, 1, (3, 16, 2, 4)).astype(numpy.float32)
x2 = numpy.random.uniform(-1, 1, (3, 16, 2, 4)).astype(numpy.float32)
x3 = numpy.random.uniform(-1, 1, (3, 16, 2, 4)).astype(numpy.float32)
x4 = numpy.random.uniform(-1, 1, (3, 16, 2, 4)).astype(numpy.float32)
mx1 = ideep4py.mdarray(x1)
mx2 = ideep4py.mdarray(x2)
mx3 = ideep4py.mdarray(x3)
mx4 = ideep4py.mdarray(x4)

x = x1 + x2 + x3 + x4
mx = ideep4py.basic_acc_sum((mx1, mx2, mx3, mx4))
# mx = numpy.asarray(mx)
res = numpy.allclose(mx, x, 1e-5, 1e-4, True)
if res is not True:
    print("error!!!!")
