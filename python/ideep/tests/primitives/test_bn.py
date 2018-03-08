import numpy
import ideep4py

from ideep4py import batchNormalization


def run():
    src = numpy.arange(3 * 2 * 2 * 2, dtype=numpy.float32)
    src = src.reshape((3, 2, 2, 2))
    src = ideep4py.mdarray(src)

    gamma = numpy.ones(2, dtype=numpy.float32)
    beta = numpy.zeros(2, dtype=numpy.float32)
    w = numpy.concatenate((gamma, beta), axis=0).reshape((2, -1))
    w = ideep4py.mdarray(w)

    eps = 2e-5

    print("FWD *****************************")
    y = batchNormalization.Forward(src, w, None, None, eps)
    print(y)
    print(-y[0])
    print(-y[1])
    print(-y[2])
    print("==============")
    y = batchNormalization.Forward(src, w, None, None, eps)
    print(y)
    print(-y[0])
    print(-y[1])
    print(-y[2])
    print("==============")
    mean = y[1]
    var = y[2]
    y = batchNormalization.Forward(src, w, mean, var, eps)
    print(y)
    print(-y[0])
    print("==============")

    print("BWD *****************************")
    diff_dst = numpy.ones(src.shape, dtype=numpy.float32)
    diff_dst = ideep4py.mdarray(diff_dst)
    y = batchNormalization.Backward(src, diff_dst, mean, var, w, eps)
    print(y)
    print(-y[0])
    print(-y[1])
    print("==============")
    y = batchNormalization.Backward(src, diff_dst, mean, var, w, eps)
    print(y)
    print(-y[0])
    print(-y[1])
    print("==============")
    src = numpy.arange(3 * 2 * 3 * 3, dtype=numpy.float32)
    src = src.reshape((3, 2, 3, 3))
    src = ideep4py.mdarray(src)
    diff_dst = numpy.ones(src.shape, dtype=numpy.float32)
    diff_dst = ideep4py.mdarray(diff_dst)
    y = batchNormalization.Backward(src, diff_dst, mean, var, w, eps)
    print(y)
    print(-y[0])
    print(-y[1])
    print("==============")
    y = batchNormalization.Backward(src, diff_dst, mean, var, None, eps)
    print(y)
    print(-y[0])
    print("==============")


run()
