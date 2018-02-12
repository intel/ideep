import numpy
import ideep4py

from ideep4py import pooling2DParam
from ideep4py import pooling2D

x = numpy.ndarray(shape=(1, 32, 224, 224), dtype=numpy.float32, order='C')
x = ideep4py.mdarray(x)

pp = pooling2DParam()
pp.src_d1 = 1
pp.src_d2 = 32
pp.src_d3 = 224
pp.src_d4 = 224
pp.dst_d1 = 1
pp.dst_d2 = 32
pp.dst_d3 = 224
pp.dst_d4 = 224
pp.kh = pp.kw = 3
pp.sy = pp.sx = 1
pp.pad_lh = pp.pad_lw = pp.pad_rh = pp.pad_rw = 1
pp.algo_kind = ideep4py.pooling2DParam.pooling_avg

print("fwd")
y = pooling2D.Forward(x, pp)
print("==============")
y = pooling2D.Forward(x, pp)
print("==============")

pp.algo_kind = ideep4py.pooling2DParam.pooling_max
(y, ws) = pooling2D.Forward(x, pp)
print("==============")
(y, ws) = pooling2D.Forward(x, pp)

print("y.shape=", y.shape)
print("ws.shape=", ws.shape)
print("ws.dtype=", ws.dtype)

print("==============")
print("bwd")
x = pooling2D.Backward(y, ws, pp)
print("==============")
x = pooling2D.Backward(y, ws, pp)
print("===== Finish max pooling backward=========")

pp.algo_kind = ideep4py.pooling2DParam.pooling_avg
x = pooling2D.Backward(y, ws, pp)
print("==============")
x = pooling2D.Backward(y, ws, pp)
print("==============")
x = numpy.ndarray(shape=(1, 32, 224, 224), dtype=numpy.float32, order='C')
x = ideep4py.mdarray(x)
x = pooling2D.Backward(x, ws, pp)
print("===== Finsh avg pooing backward =========")
print("x.shape=", x.shape)
print("==============")
