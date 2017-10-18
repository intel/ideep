import numpy
import chainer
import chainer.functions

from chainer import function_node
from chainer.utils import type_check
from ideep import xnn


class LinearFunction(function_node.FunctionNode):

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(2 <= n_in, n_in <= 3)
        x_type, w_type = in_types[:2]

        type_check.expect(
            x_type.dtype.kind == 'f',
            w_type.dtype.kind == 'f',
            x_type.ndim >= 2,
            w_type.ndim >= 2,
            type_check.prod(x_type.shape[1:]) == \
                    type_check.prod(w_type.shape[1:]),
        )
        if type_check.eval(n_in) == 3:
            b_type = in_types[2]
            type_check.expect(
                b_type.dtype == x_type.dtype,
                b_type.ndim == 1,
                b_type.shape[0] == w_type.shape[0],
            )

    def forward(self, inputs):
        cc = xnn.LinearForward(inputs)
        self.hint = cc.hint
        self.W = cc.W

        y, = cc.execute_on()
        y.reset_buf_order()

        if len(inputs) == 3:
            self.retain_inputs((0, 1, 2))
        else:
            self.retain_inputs((0, 1))
        return y,

    def backward(self, indexes, gy):
        inputs = self.get_retained_inputs()
        inputs = tuple([input.data for input in self.inputs])

        ret = []
        if 0 in indexes:
            gx = LinearGradD(inputs, self.hint, self.W).apply(gy)
            ret.append(gx[0])
        if 1 in indexes or 2 in indexes:
            gW_b = LinearGradW(inputs, self.hint).apply(gy)
            if 1 in indexes:
                ret.append(gW_b[0])
            if 2 in indexes:
                ret.append(gW_b[1])

        return ret


class LinearGradD(function_node.FunctionNode):

    def __init__(self, inputs, hint, ccW):
        super(LinearGradD, self).__init__()

        self.inputs = inputs
        self.W = ccW
        self.hint = hint

    def forward_cpu(self, inputs):
        cc = xnn.LinearBackwardData(self.inputs, inputs, self.hint, self.W)

        gx = cc.execute_on()
        gx[0].reset_buf_order()

        return gx

    def backward(self, indexes, gy):
        x = self.inputs[0]
        W = self.inputs[1]

        ret = []
        if 0 in indexes:
            gx = linear(gy, W)
            ret.append(gx)
        if 1 in indexes:
            gW = linear(gy, x)
            ret.append(gW)
        if 2 in indexes:
            gb = chainer.functions.sum(gy, axis=0)
            ret.append(gb)

        return ret


class LinearGradW(function_node.FunctionNode):

    def __init__(self, inputs, hint):
        super(LinearGradW, self).__init__()

        self.inputs = inputs
        self.hint = hint

    def forward_cpu(self, inputs):
        cc = xnn.LinearBackwardWeighs(self.inputs, inputs, self.hint)

        gW_b = cc.execute_on()
        gW_b[0].reset_buf_order()

        return gW_b

    def backward(self, indexes, gy):
        x = self.inputs[0]
        W = self.inputs[1]

        ret = []
        if 0 in indexes:
            gx = linear(gy, W)
            ret.append(gx)
        if 1 in indexes:
            gW = linear(gy, x)
            ret.append(gW)
        if 2 in indexes:
            gb = chainer.functions.sum(gy, axis=0)
            ret.append(gb)

        return ret


def linear(x, W, b=None):
    if b is None:
        args = x, W
    else:
        args = x, W, b

    y, = LinearFunction().apply(args)
    return y
