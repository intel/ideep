import numpy
import chainer

from chainer import function_node
from chainer import utils
from chainer.utils import type_check

from ideep import xnn


def _heaviside(x):
    return (x > 0).astype(x.dtype)

class ReLU(function_node.FunctionNode):

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype.kind == 'f',
        )

    def forward_cpu(self, x):
        self.retain_inputs((0,))
        self.retain_outputs((0,))

        cc = xnn.ReLUForward(x)
        self.hint = cc.hint
        y, = cc.execute_on()

        return y,

    def backward(self, indexes, gy):
        x = self.get_retained_inputs()[0]
        y = self.get_retained_outputs()[0]

        return ReLUGrad(x, y, self.hint).apply((gy[0],))


class ReLUGrad(function_node.FunctionNode):

    def __init__(self, x, y, hint):
        super(ReLUGrad, self).__init__()

        self.x = x.data
        self.y = y.data
        self.hint = hint

    def forward_cpu(self, inputs):

        cc = xnn.ReLUBackward((self.x,), inputs, self.hint)
        gx, = cc.execute_on()

        return gx,

    def backward(self, indexes, gy):
        return gy[0] * _heaviside(self.y),


def relu(x):
    y, = ReLU().apply((x,))
    return y
