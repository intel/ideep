import numpy
import unittest

import chainer

from chainer import function_node
from ideep.xnn import convolution_2d

class Convolution2DFunctionMKLDNN(function_node.FunctionNode):

    def __init__(self, stride=1, pad=0, cover_all=False):
        self.sy, self.sx = _pair(stride)
        self.ph, self.pw = _pair(pad)
        self.cover_all = cover_all

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(2 <= n_in, n_in <= 3)

        x_type = in_types[0]
        w_type = in_types[1]
        type_check.expect(
            x_type.dtype.kind == 'f',
            w_type.dtype.kind == 'f',
            x_type.ndim == 4,
            w_type.ndim == 4,
            x_type.shape[1] == w_type.shape[1],
        )

        if type_check.eval(n_in) == 3:
            b_type = in_types[2]
            type_check.expect(
                b_type.dtype == x_type.dtype,
                b_type.ndim == 1,
                b_type.shape[0] == w_type.shape[0],
            )

    def forward_cpu(self, inputs):
        if len(inputs) == 3:
            self.retain_inputs((0, 1, 2))
        else:
            self.retain_inputs((0, 1))

        cc = convolution_2d.ConvolutionForward(
            inputs, stride=(self.sy, self.sx),
            pad=(self.ph, self.pw), cover_all=self.cover_all,
            pos=(self.rank, self.fanout))

        self.hint = cc.hint
        self.W = cc.W
        weight_optimization_trigger(self)

        y, = cc.execute_on()
        y.reset_buf_order()

        return y,

    def backward_cpu(self, indexes, grad_outputs):
        inputs = self.get_retained_inputs()

        ret = []

        if 0 in indexes:
            cc_data = convolution_2d.ConvolutionBackwardData(
                inputs, grad_outputs,
                self.hint, self.W,
                stride=(self.sy, self.sx), pad=(self.ph, self.pw),
                cover_all=self.cover_all, pos=(self.rank, self.fanout))
            gx = cc_data.execute_on()
            gx[0].reset_buf_order()
            ret.append(gx)

        if 1 in indexes:
            cc_weight = convolution_2d.ConvolutionBackwardWeighs(
                inputs, grad_outputs, self.hint,
                stride=(self.sy, self.sx), pad=(self.ph, self.pw),
                cover_all=self.cover_all, pos=(self.rank, self.fanout))

            gW_b = cc_weight.execute_on()
            gW_b[0].reset_buf_order()

            ret.append(gW_b[0])

            if 2 in indexes:
                ret.append(gW_b[1])

        return ret
