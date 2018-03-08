import numpy
from chainer import ideepy


dropout_ratio = 0.8

# Forward
x = numpy.random.rand(128, 3, 224, 224).astype(numpy.float32)
x_md, = ideepy.to_mdarray((x, ))
mask, y = ideepy.dropout.Forward(x_md, dropout_ratio)
y = numpy.array(y, dtype=numpy.float32)
y_expect = x * mask
numpy.testing.assert_allclose(y, y_expect)

# Backward
gy = numpy.random.rand(128, 3, 224, 224).astype(numpy.float32)
gy_md, = ideepy.to_mdarray((gy, ))
gx = ideepy.dropout.Backward(mask, gy_md)
gx = numpy.array(gx, dtype=numpy.float32)
gx_expect = gy * mask
numpy.testing.assert_allclose(gx, gx_expect)
