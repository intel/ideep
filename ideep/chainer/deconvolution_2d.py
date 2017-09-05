import numpy as np

from chainer import function
from chainer.utils import conv
from chainer.utils import type_check

from mkldnn.chainer.runtime import Engine
from mkldnn.compute_complex import ComputeComplex, array, reuse_buffer

from mkldnn.api.support import forward, convolution_direct, zero

import mkldnn.api.memory as m

import mkldnn.api.convolution_forward as conv_forward
import mkldnn.api.convolution_backward_data as conv_backdata
import mkldnn.api.convolution_backward_weights as conv_backweights
from mkldnn.mdarray import mdarray

deconv_f_op = conv_backdata.conv_bd_op  # deconv fwd --> conv bwd data
deconv_bd_op = conv_forward.conv_f_op  # deconv bwd data --> conv fwd
deconv_bw_op = conv_backweights.conv_bw_op  # deconv bwd weights --> conv bwd weights


class deconv_geometry(object):
    def __init__(self, x_shape, W_shape, stride, pad, outsize):
        assert isinstance(x_shape, tuple), 'X shape must be tuple'
        assert isinstance(W_shape, tuple), 'W shape must be tuple'

        sy, sx = _pair(stride)
        p_upper, p_left = _pair(pad)

        # deconv's weight is inited as iohw
        in_c, out_c, kh, kw = W_shape
        n, c, h, w = x_shape

        out_h, out_w = outsize

        if out_h is None:
            out_h = conv.get_deconv_outsize(h, kh, sy, p_upper)
            assert out_h > 0, 'Height in the output should be positive.'
        if out_w is None:
            out_w = conv.get_deconv_outsize(w, kw, sx, p_left)
            assert out_w > 0, 'Width in the output should be positive.'

        p_down = p_upper
        p_right = p_left

        self.p_upper = p_upper
        self.p_lef = p_left
        self.p_down = p_down
        self.p_right = p_right
        self.out_h = out_h
        self.out_w = out_w

        """
        output channel should be weight's input channle
        """
        self._out_shape = (n, out_c, out_h, out_w)
        self._geometry = (_pair(stride)), (p_upper, p_left), (p_down, p_right)

    @property
    def out_shape(self):
        return self._out_shape

    @property
    def geometry(self):
        return self._geometry


def create_forward_desc(d_creator, y, inputs, geometry):
    inputs_d = [m.desc(v.shape, m.memory.f32, m.memory.any)
                for v in inputs if v is not None]

    strides = geometry[0]
    padding_ul = geometry[1]
    padding_dr = geometry[2]
    x_desc = inputs_d[0]
    w_desc = inputs_d[1]
    """
       deconv forward --> convolution backward data
       src desc <--> dst desc
    """
    return d_creator(convolution_direct, y, w_desc, x_desc, strides, padding_ul, padding_dr, zero)


def create_backward_data_desc(d_creator, y, inputs, geometry):
    inputs_d = [m.desc(v.shape, m.memory.f32, m.memory.any)
                for v in inputs if v is not None]

    strides = geometry[0]
    padding_ul = geometry[1]
    padding_dr = geometry[2]
    x_desc = inputs_d[0]
    w_desc = inputs_d[1]

    """
        deconv backward data --> convolution forward
        src desc <--> dst desc
    """
    return d_creator(forward, convolution_direct, y, w_desc, x_desc,
                     strides, padding_ul, padding_dr, zero)


def create_backward_weights_desc(d_creator, y, inputs, geometry):
    inputs_d = [m.desc(v.shape, m.memory.f32, m.memory.any)
                for v in inputs if v is not None]

    strides = geometry[0]
    padding_ul = geometry[1]
    padding_dr = geometry[2]
    x_desc = inputs_d[0]
    w_desc = inputs_d[1]

    """
        deconv backward weights --> convolution backward weights
        src desc <--> dst desc
    """
    return d_creator(convolution_direct, y, w_desc, x_desc, strides, padding_ul, padding_dr, zero)


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x


class DeconvolutionForward(ComputeComplex):
    cc_type = 'f'

    def __init__(self, inputs, stride=1, pad=0, outsize=None,
                 pos=None, e=Engine()):
        x = inputs[0]
        W = inputs[1]

        if self.new:
            self._create_cc(x, W, stride, pad, outsize, e)
            self.num_inputs = len(inputs)
        else:
            self._reuse_cc(x, W)

    def _create_cc(self, x, W, stride, pad, outsize, e):
        super(DeconvolutionForward, self).__init__()

        g = deconv_geometry(x.shape, W.shape, stride, pad, outsize)

        y_d = m.desc(g.out_shape, m.memory.f32, m.memory.any)

        self.geometry = g.geometry

        # Transform inputs
        self.x = array(x, m.memory.nchw, e)
        self.W = array(W, m.memory.oihw, e)

        # When create conv bwd data, need conv forward as hint
        # decov_backward_data --> conv forward
        self.hint_cc_d = create_backward_data_desc(conv_forward.desc, y_d, (x, W), g.geometry)
        self.hint_cc_pd = conv_forward.primitive_desc(self.hint_cc_d, e)

        # Create deconv forward primitive based on conv bwd data
        # need conv fwd as hint
        cc_d = create_forward_desc(conv_backdata.desc, y_d, (x, W), g.geometry)
        cc_pd = conv_backdata.primitive_desc(cc_d, e, self.hint_cc_pd)

        y = deconv_f_op(cc_pd, self.x, self.W, self.dag_)
        self.outputs = y,

    def _reuse_cc(self, x, W):
        reuse_buffer(self.x, x)
        reuse_buffer(self.W, W)

    def match(self, inputs, stride=1, pad=0, outsize=None, **kwargs):
        x = inputs[0]
        W = inputs[1]
        if (self.x.shape != x.shape) or (self.W.shape != W.shape):
            return False
        if (isinstance(x, mdarray) and (x is not self.x)):
            return False
        g = deconv_geometry(x.shape, W.shape, stride, pad, outsize)
        return (self.geometry == g.geometry) and (self.num_inputs == len(inputs))


class DeconvolutionBackwardData(ComputeComplex):
    cc_type = 'bd'

    def __init__(self, inputs, grad_outputs, hint,
                 stride=1, pad=0, outsize=None, pos=None, e=Engine()):
        x = inputs[0]
        W = inputs[1]
        gy = grad_outputs[0]
        if self.new:
            self._create_cc(x, W, gy, hint, stride, pad, outsize, e)
        else:
            self._reuse_cc(W, gy)

    def _create_cc(self, x, W, gy, hint, stride, pad, outsize, e):
        super(DeconvolutionBackwardData, self).__init__()

        # Transform inputs
        self.gy = array(gy, m.memory.nchw, e)
        self.W = array(W, m.memory.oihw, e)

        # here hint is deconv backward data pd, use it directly
        gx = deconv_bd_op(hint, self.gy, self.W, self.dag_)

        self._hint = hint
        self.outputs = gx,

    def _reuse_cc(self, W, gy):
        reuse_buffer(self.W, W)
        reuse_buffer(self.gy, gy)

    def match(self, inputs, grad_outputs, hint, *args, **kwargs):
        return ((hint is not None) and (hint is self._hint))


class DeconvolutionBackwardWeights(ComputeComplex):
    cc_type = 'bw'

    def __init__(self, inputs, grad_outputs, hint,
                 stride=1, pad=0, outsize=None, pos=None, e=Engine()):
        x = inputs[0]
        W = inputs[1]
        gy = grad_outputs[0]

        if self.new:
            self._create_cc(x, W, gy, hint, stride, pad, outsize, e)
        else:
            self._reuse_cc(x, gy)

    def _create_cc(self, x, W, gy, hint, stride, pad, outsize, e):
        super(DeconvolutionBackwardWeights, self).__init__()
        g = deconv_geometry(x.shape, W.shape, stride, pad, outsize)

        gy_d = m.desc(gy.shape, m.memory.f32, m.memory.any)

        cc_d = create_backward_weights_desc(conv_backweights.desc, gy_d, (x, W), g.geometry)
        cc_pd = conv_backweights.primitive_desc(cc_d, e, hint)

        self.gy = array(gy, m.memory.nchw, e)
        self.x = array(x, m.memory.nchw, e)
        self._hint = hint

        # use conv bwd weight to implement deconv bwd weight
        # need to x <--> gy
        gW = deconv_bw_op(cc_pd, self.gy, self.x, self.dag_)

        self.outputs = gW,

    def _reuse_cc(self, x, gy):
        reuse_buffer(self.x, x)
        reuse_buffer(self.gy, gy)

    def match(self, inputs, grad_outputs, hint, *args, **kwargs):
        return ((hint is not None) and (hint is self._hint))


class Deconvolution2DFunctionMKLDNN(function.Function):

    def __init__(self, stride=1, pad=0, outsize=None, deterministic=False):
        self.sy, self.sx = _pair(stride)
        self.ph, self.pw = _pair(pad)
        self.outsize = (None, None) if outsize is None else outsize
        self.outh, self.outw = (None, None) if outsize is None else outsize
        self.deterministic = deterministic

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
            type_check.expect(
                x_type.shape[2] ==
                conv.get_conv_outsize(self.outh, w_type.shape[2],
                                      self.sy, self.ph),
            )
        if self.outw is not None:
            type_check.expect(
                x_type.shape[3] ==
                conv.get_conv_outsize(self.outw, w_type.shape[3],
                                      self.sx, self.pw),
            )

        if type_check.eval(n_in) == 3:
            b_type = in_types[2]
            type_check.expect(
                b_type.dtype == x_type.dtype,
                b_type.ndim == 1,
                b_type.shape[0] == w_type.shape[1]
            )

    def forward_cpu(self, inputs):
        cc = DeconvolutionForward(inputs, stride=(self.sy, self.sx),
                                  pad=(self.ph, self.pw), outsize=self.outsize,
                                  pos=(self.rank, self.fanout))

        self.hint = cc.hint_cc_pd  # conv fwd primitive desc as hint, which can be used as deconv bwd data

        y, = cc.execute_on()
        y.reset_buf_order()

        b = inputs[2] if len(inputs) == 3 else None
        if b is not None:
            b = b.reshape(1, b.size, 1, 1)
            # FIXME
            # inplace add will call MKLDNN sum primitive, which need shape is same
            # Workaround here to resize it
            tmp = np.ones((y.shape), dtype=np.float32)
            b = b * tmp
            y += b

        return y,

    def backward_cpu(self, inputs, grad_outputs):
        gy = grad_outputs[0]

        cc_weights = DeconvolutionBackwardWeights(inputs, grad_outputs, self.hint,
                                                  stride=(self.sy, self.sx), pad=(self.ph, self.pw),
                                                  outsize=self.outsize,
                                                  pos=(self.rank, self.fanout))
        gW = cc_weights.execute_on()
        gW[0].reset_buf_order()

        cc_data = DeconvolutionBackwardData(inputs, grad_outputs, self.hint,
                                            stride=(self.sy, self.sx), pad=(self.ph, self.pw),
                                            outsize=self.outsize,
                                            pos=(self.rank, self.fanout))
        gx = cc_data.execute_on()
        gx[0].reset_buf_order()

        b = inputs[2] if len(inputs) == 3 else None
        if b is not None:
            gb = gy.sum(axis=(0, 2, 3))
            return gx[0], gW[0], gb
        else:
            return gx + gW
