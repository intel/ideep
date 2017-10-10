import numpy

import chainer
from chainer import function_node

import example.functions
import ideep.xnn as xnn
from example.functions.convolution_2d import Convolution2DGradW


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x

class Deconvolution2DFucntion(function_node.FunctionNode):

    cover_all = None

    def __init__(self, stride=1, pad=0, outsize=None):
        self.sy, self.sx = _pair(stride)
        self.ph, self.pw = _pair(pad)
        self.outh, self.outw = (None, None) if outsize is None else outsize

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(2 <= n_in, n_in <= 3)
        x_type, w_type = in_types[:2]

        type_check.expect(
            x_type.dtype.kind == 'f',
            w_type.dtype.kind == 'f',
            x_type.ndim == 4,
            w_type.ndim == 4,
            x_type.shape[1] == w_type.shape[0]
        )

        if self.outh is not None:
            lower_bound = conv.get_conv_outsize(
                self.outh, w_type.shape[2], self.sy, self.ph)
            upper_bound = conv.get_conv_outsize(
                self.outh, w_type.shape[2], self.sy, self.ph, cover_all=True)
            type_check.expect(
                lower_bound <= x_type.shape[2],
                x_type.shape[2] <= upper_bound)
        if self.outw is not None:
            lower_bound = conv.get_conv_outsize(
                self.outw, w_type.shape[3], self.sx, self.pw)
            upper_bound = conv.get_conv_outsize(
                self.outw, w_type.shape[3], self.sx, self.pw, cover_all=True)
            type_check.expect(
                lower_bound <= x_type.shape[3],
                x_type.shape[3] <= upper_bound)

        if type_check.eval(n_in) == 3:
            b_type = in_types[2]
            type_check.expect(
                b_type.dtype == x_type.dtype,
                b_type.ndim == 1,
                b_type.shape[0] == w_type.shape[1]
            )

    def forward_cpu(self, inputs):
        cc_data = xnn.ConvolutionBackwardData(
            inputs, xnn.dummy_hint, stride=(self.sy, self.sx),
            pad=(self.ph, self.pw), cover_all=self.cover_all,
            pos=(0, 0))

        gx = cc_data.execute_on()
        return gx

    def backward(self, indexes, grad_outputs):
        x, W = self.get_retained_inputs()
        gy, = grad_outputs

        ret = []
        if 0 in indexes:
            if self.cover_all is None:
                self._set_cover_all(x, W)
            gx = example.functions.convolution_2d(
                    gy, W, stride=(self.sy, self.sx),
                pad=(self.ph, self.pw), cover_all=self.cover_all)
            ret.apped(gx)
        if 1 in indexes:
            if self.cover_all is None:
                self._set_cover_all(x, W)
            gW, = Convolution2DGradW(self).apply((gx, x))
            ret.append(gW[0])

            if 2 in indexes:
                ret.append(gW[1])
        return ret

    def _set_cover_all(self, x, W):
        in_h, in_w = x.shape[2:]
        kh, kw = W.shape[2:]
        self.cover_all = (
            in_h != conv.get_conv_outsize(self.outh, kh, self.sy, self.ph) or
            in_w != conv.get_conv_outsize(self.outw, kw, self.sx, self.pw))

def deconvolution_2d(x, W, b=None, stride=1, pad=0, outsize=None, **kwargs):
    argument.check_unexpected_kwargs(
        kwargs, deterministic="deterministic argument is not "
        "supported anymore. "
        "Use chainer.using_config('cudnn_deterministic', value) "
        "context where value is either `True` or `False`.")
    argument.assert_kwargs_empty(kwargs)

    func = Deconvolution2DFunction(stride, pad, outsize)
    if b is None:
        args = x, W
    else:
        args = x, W, b
    y, = func.apply(args)
    return y
