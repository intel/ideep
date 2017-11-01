from ideep.mdarray import mdarray
from ideep.cpu_engine import Engine
import ideep.compute_complex as CC

from ideep.array import array
from ideep.utils import conv
from ideep.api.support import forward, convolution_direct, zero

import ideep.api.memory as m
import ideep.api.convolution_forward as conv_forward
import ideep.api.convolution_backward_data as conv_backdata
import ideep.api.convolution_backward_weights as conv_backweights

from ideep.xnn.optimization import WeightReorderOptimization


conv_f_op = conv_forward.conv_f_op
conv_bd_op = conv_backdata.conv_bd_op

conv_bw_op = conv_backweights.conv_bw_op
conv_bwb_op = conv_backweights.conv_bwb_op


def create_dummy_hint():
    """ Create a dummy hint

    To create a convolution backward primitive, one needs a forward
    primitive as a hint. Though there is no use of it in actual
    implementations. A dummy hint can be a wordaround of this situation.
    There would be a interface requires no hint in the furture.

    """

    x_md = m.desc((128, 3, 227, 227), m.memory.f32, m.memory.any)
    W_md = m.desc((96, 3, 11, 11), m.memory.f32, m.memory.any)
    o_md = m.desc((128, 96, 55, 55), m.memory.f32, m.memory.any)

    dummy_d = conv_forward.desc(
        forward, convolution_direct, x_md, W_md, o_md,
        (4, 4), (0, 0), (0, 0), zero)
    return conv_forward.primitive_desc(dummy_d, Engine());


dummy_hint = create_dummy_hint()


def create_forward_desc(d_creator, o_expect, inputs, geometry):
    inputs_d = [m.desc(v.shape, m.memory.f32, m.memory.any)
                for v in inputs if v is not None]

    strides = geometry[0]
    padding_ul = geometry[1]
    padding_dr = geometry[2]
    x_desc = inputs_d[0]
    w_desc = inputs_d[1]
    if len(inputs_d) == 3:
        b_desc = inputs_d[2]
        return d_creator(forward, convolution_direct,
                         x_desc, w_desc, b_desc, o_expect,
                         strides, padding_ul, padding_dr, zero)
    else:
        return d_creator(forward, convolution_direct,
                         x_desc, w_desc, o_expect,
                         strides, padding_ul, padding_dr, zero)


class ConvolutionForward(CC.ComputeComplex):
    cc_type = 'f'

    def __init__(self, inputs, stride=1, pad=0, cover_all=False,
                 pos=(0, 0), e=Engine()):

        x = inputs[0]
        W = inputs[1]
        b = inputs[2] if len(inputs) == 3 else None

        if self.new:
            self._create_cc(x, W, b, stride, pad, cover_all, e)
            self.num_inputs = len(inputs)
        else:
            self._reuse_cc(x, W, b)

    def _create_cc(self, x, W, b, stride, pad, cover_all, e):
        super(ConvolutionForward, self).__init__()
        g = conv.conv_geometry(x.shape, W.shape, stride, pad, cover_all)

        y_d = m.desc(g.out_shape, m.memory.f32, m.memory.any)

        # Create primitive_desc from any
        cc_d = create_forward_desc(
            conv_forward.desc, y_d, (x, W, b), g.geometry)

        cc_pd = conv_forward.primitive_desc(cc_d, e)
        w_mpd = cc_pd.weights_primitive_desc()
        self.usr_w = array(W, m.memory.oihw, e)
        outputs = CC.reorder_if_must(self.usr_w, w_mpd, e, self.dag)
        if len(outputs) == 2:
            self.W, self.itm_arr = outputs[:2]
        else:
            self.W = outputs[0]

        # Record weight reorder primitive hint
        if self.usr_w is not self.W:
            wro = WeightReorderOptimization()
            wro.reorder = self.dag.size() - 1
            wro.optimized = False
            self.weight_reorder_opt = wro
        else:
            self.weight_reorder_opt = None

        self.x = array(x, m.memory.nchw, e)
        if b is not None:
            self.b = array(b, m.memory.x, e)

        if b is None:
            y = conv_f_op(cc_pd, self.x, self.W, self.dag)
        else:
            y = conv_f_op(cc_pd, self.x, self.W, self.b, self.dag)

        self._hint = cc_pd
        self.outputs = y,

#    def _reuse_cc(self, x, W, b):
#        reuse_buffer(self.x, x)
#        # Weight optimization starts from second iteration.
#        # check cc.W with W
#        if self.W is not W:
#            reuse_buffer(self.usr_w, W)
#        else:
#            if self.weight_reorder_opt is not None and \
#               self.weight_reorder_opt.optimized is False:
#                self.dag.erase(self.dag.begin() + \
#                self.weight_reorder_opt.reorder)
#                self.weight_reorder_opt.optimized = True
#
#        if b is not None:
#            reuse_buffer(self.b, b)
#
#    def match(self, inputs, stride=1, pad=0, cover_all=False, **kwargs):
#        x = inputs[0]
#        W = inputs[1]
#        if (self.x.shape != x.shape) or (self.W.shape != W.shape):
#            return False
#        if (isinstance(x, mdarray) and (x is not self.x)):
#            return False
#        g = conv.conv_geometry(x.shape, W.shape, stride, pad, cover_all)
#
#        return (self.geometry == g.geometry) and \
#                (self.num_inputs == len(inputs))


class ConvolutionBackwardData(CC.ComputeComplex):
    cc_type = 'bd'

    def __init__(
        self, inputs, stride=1, pad=0, outsize = None,
        cover_all=False, hint = dummy_hint, pos=(0, 0), e=Engine()):

        # Not support b yet
        gy, W = inputs[:2]

        if self.new:
            self._create_cc(gy, W, stride, pad,
                outsize, cover_all, hint, e)
        else:
            self._reuse_cc(gy, W)

    def _create_cc(
        self, gy, W, stride, pad, outsize, cover_all, hint, e):
        super(ConvolutionBackwardData, self).__init__()

        # Get information
        g = conv.deconv_geometry(
            gy.shape, W.shape, stride, pad, outsize)

        gy_d = m.desc(gy.shape, m.memory.f32, m.memory.any)
        W_d = m.desc(W.shape, m.memory.f32, m.memory.any)
        x_d = m.desc(g.in_shape, m.memory.f32, m.memory.any)

        cc_d = conv_backdata.desc(
            convolution_direct, x_d, W_d, gy_d, g.geometry[0]
            , g.geometry[1], g.geometry[2], zero)

        cc_pd = conv_backdata.primitive_desc(cc_d, e, hint)

        self.gy = array(gy, m.memory.nchw, e)
        self.W = array(W, m.memory.oihw, e)

        gx = conv_bd_op(cc_pd, self.gy, self.W, self.dag)

        self._hint = hint
        self.outputs = gx,

#    def _reuse_cc(self, W, gy):
#        reuse_buffer(self.W, W)
#        reuse_buffer(self.gy, gy)
#
#    def match(self, inputs, **kwargs):
#        return hint is self._hint


class ConvolutionBackwardWeights(CC.ComputeComplex):
    cc_type = 'bw'

    def __init__(
        self, inputs, stride=1, pad=0, outsize=None, cover_all=False,
        hint=dummy_hint, pos=(0, 0), e=Engine()):

        x, gy = inputs[:2]

        if self.new:
            self._create_cc(x, gy, stride, pad, outsize, cover_all, hint, e)
        else:
            self._reuse_cc(x, gy)

    def _create_cc(self, x, gy, stride, pad, outsize, cover_all, hint, e):
        super(ConvolutionBackwardWeights, self).__init__()

        o = gy.shape[1]
        i = x.shape[1]
        kh, kw = outsize

        g = conv.conv_geometry(
            x.shape, (o, i, kh, kw), stride, pad, cover_all)

        x_d = m.desc(x.shape, m.memory.f32, m.memory.any)
        gy_d = m.desc(gy.shape, m.memory.f32, m.memory.any)
        W_d = m.desc((o, i, kh, kw), m.memory.f32, m.memory.any)
        b_d = m.desc((o, ), m.memory.f32, m.memory.any)

        cc_d = conv_backweights.desc(
            convolution_direct, x_d, W_d, b_d, gy_d,
            g.geometry[0], g.geometry[1], g.geometry[2], zero)

        cc_pd = conv_backweights.primitive_desc(cc_d, e, hint)

        self.x = array(x, m.memory.nchw, e)
        self.gy = array(gy, m.memory.nchw, e)
        self._hint = hint

        gW = conv_bwb_op(cc_pd, self.x, self.gy, self.dag)
        gb = gW.extra

        self.outputs = gW, gb

#    def _reuse_cc(self, x, gy):
#        reuse_buffer(self.x, x)
#        reuse_buffer(self.gy, gy)
#
#    def match(self, inputs, grad_ouputs, hint, *args, **kwargs):
#        return (hint is self._hint)
