from mkldnn.chainer.runtime import Engine
from mkldnn.compute_complex import reorder_if_must, ComputeComplex, array, reuse_buffer

# Most important thing
from mkldnn.api.support import forward
import mkldnn.api.memory as m
import mkldnn.api.inner_product_forward as ip_forward
import mkldnn.api.inner_product_backward_data as ip_backdata
import mkldnn.api.inner_product_backward_weights as ip_backweights
from mkldnn.mdarray import mdarray

from mkldnn.api.inner_product_forward import linear_f_op
from mkldnn.api.inner_product_backward_data import linear_bd_op
from mkldnn.api.inner_product_backward_weights import linear_bw_op
from mkldnn.api.inner_product_backward_weights import linear_bwb_op


def _x_format(ndim):
    if ndim == 2:
        return m.memory.nc
    elif ndim == 4:
        return m.memory.nchw
    else:
        return NotImplemented


def _W_format(ndim):
    if ndim == 2:
        return m.memory.oi
    elif ndim == 4:
        return m.memory.oihw
    else:
        return NotImplemented


def create_forward_desc(d_creator, o_expect, *inputs):
    inputs_d = [m.desc(v.shape, m.memory.f32, m.memory.any)
                for v in inputs if v is not None]
    x_m = inputs_d[0]
    W_m = inputs_d[1]
    if len(inputs_d) == 3:
        b_m = inputs_d[2]
        return d_creator(forward, x_m, W_m, b_m, o_expect)
    else:
        return d_creator(forward, x_m, W_m, o_expect)


def create_backward_desc(d_creator, *inputs):
    inputs_d = [m.desc(v.shape, m.memory.f32, m.memory.any)
                for v in inputs if v is not None]

    return d_creator(*inputs_d)


class LinearForward(ComputeComplex):
    cc_type = 'f'

    def _create_cc(self, x, W, b, e=Engine()):
        y_d = m.desc((x.shape[0], W.shape[0]), m.memory.f32, m.memory.any)
        # Create primitive_desc from any
        cc_d = create_forward_desc(ip_forward.desc, y_d, x, W, b)
        cc_pd = ip_forward.primitive_desc(cc_d, e)

        # Transform inputs
        self.x = array(x, _x_format(x.ndim), e)
        w_mpd = cc_pd.weights_primitive_desc()
        self.usr_w = array(W, _W_format(W.ndim), e)
        outputs = reorder_if_must(self.usr_w, w_mpd, e, self.dag_)
        if len(outputs) == 2:
            self.W, self.itm_arr = outputs[:2]
        else:
            self.W = outputs[0]

        if b is not None:
            self.b = array(b, m.memory.x, e)
            y = linear_f_op(cc_pd, self.x, self.W, self.b, self.dag_)
        else:
            y = linear_f_op(cc_pd, self.x, self.W, self.dag_)

        # Prepare output
        # y = mdarray(cc_pd.dst_primitive_desc())

        # dag = self.dag_

        # # Reorder if must
        # x_m = reorder_if_must(self.x.memory,
        #         cc_pd.src_primitive_desc(), dag)
        # W_m = reorder_if_must(self.W.memory,
        #         cc_pd.weights_primitive_desc(), dag)

        # if b is None:
        #     dag.push_back(ip_forward.inner_product_forward(cc_pd,
        #         at(x_m), at(W_m), y.memory))
        # else:
        #     dag.push_back(ip_forward.inner_product_forward(cc_pd,
        #         at(x_m), at(W_m), at(self.b.memory), y.memory))

        # self.x_m = x_m
        # self.W_m = W_m
        self._hint = cc_pd
        self.outputs = y,

    def _reuse_cc(self, x, W, b, e=Engine()):
        reuse_buffer(self.x, x)
        reuse_buffer(self.W, W)
        if b is not None:
            reuse_buffer(self.b, b)

    def match(self, inputs):
        if len(inputs) != self.argc:
            return False
        x, W = inputs[:2]
        if (x.shape != self.x.shape) or (W.shape != self.W.shape):
            print('WARNING: LinearForard x or w shape mismatch', x.shape, self.x.shape, W.shape, self.W.shape)
            return False
        if(isinstance(x, mdarray) and (x is not self.x)):
            return False
        return True

    def __init__(self, inputs, pos=(0, 0), e=Engine()):
        super(LinearForward, self).__init__()
        x = inputs[0]
        W = inputs[1]
        b = inputs[2] if len(inputs) == 3 else None
        self.argc = len(inputs)

        if self.new:
            self._create_cc(x, W, b, e)
        else:
            self._reuse_cc(x, W, b, e)


class LinearBackwardData(ComputeComplex):
    cc_type = 'bd'

    def __init__(self, inputs, grad_outputs, hint, fwd_W, pos=(0, 0), e=Engine()):
        super(LinearBackwardData, self).__init__()
        W = inputs[1]
        gy = grad_outputs[0]
        self.argc = len(inputs)

        if self.new:
            x = inputs[0]
            self._create_cc(x, W, gy, hint, fwd_W, e)
        else:
            self._reuse_cc(W, gy)

    def match(self, inputs, grad_outputs, hint, *args):
        if len(inputs) != self.argc:
            return False

        gy = grad_outputs[0]
        if isinstance(gy, mdarray) and gy is not self.gy:
            return False

        return (hint is self._hint)

    def _create_cc(self, x, W, gy, hint, fwd_W, e=Engine()):
        # Create primitive descriptor
        cc_d = create_backward_desc(ip_backdata.desc, x, W, gy)
        cc_pd = ip_backdata.primitive_desc(cc_d, e, hint)

        # Transform inputs
        self.W = fwd_W
        self.gy = array(gy, m.memory.nc, e)

        gx = linear_bd_op(cc_pd, self.gy, self.W, self.dag_)

        # # Prepare output mdarray
        # gx = mdarray(cc_pd.diff_src_primitive_desc())

        # dag = self.dag_

        # # Reorder if must
        # gy_m = reorder_if_must(self.gy.memory, cc_pd.diff_dst_primitive_desc(), dag)
        # W_m = reorder_if_must(self.W.memory, cc_pd.weights_primitive_desc(), dag)

        # dag.push_back(ip_backdata.inner_product_backward_data(cc_pd,
        #     at(gy_m), at(W_m), gx.memory))

        # self.gy_m = gy_m
        # self.W_m = W_m
        self._hint = hint
        self.outputs = gx,

    def _reuse_cc(self, W, gy):
        reuse_buffer(self.W, W)
        reuse_buffer(self.gy, gy)


class LinearBackwardWeighs(ComputeComplex):
    cc_type = 'bw'

    def _create_cc(self, x, W, b, gy, hint, e):
        cc_d = create_backward_desc(ip_backweights.desc, x, W, b, gy)
        cc_pd = ip_backweights.primitive_desc(cc_d, e, hint)

        # Transfer inputs to mdarray
        self.x = array(x, _x_format(x.ndim), e)
        self.gy = array(gy, m.memory.nc, e)

        if b is None:
            gW = linear_bw_op(cc_pd, self.x, self.gy, self.dag_)
        else:
            gW = linear_bwb_op(cc_pd, self.x, self.gy, self.dag_)
            gb = gW.extra

        # Prepare outputs mdarray
        # gW = mdarray(cc_pd.diff_weights_primitive_desc())
        # if b is not None:
        #     gb = mdarray(cc_pd.diff_bias_primitive_desc())
        #     self.has_b = True
        # else:
        #     self.has_b = False

        # dag = self.dag_

        # # Reorder if must
        # gy_m = reorder_if_must(self.gy.memory, cc_pd.diff_dst_primitive_desc(), dag)
        # x_m = reorder_if_must(self.x.memory, cc_pd.src_primitive_desc(), dag)

        # if b is not None:
        #     dag.push_back(ip_backweights.inner_product_backward_weights(cc_pd,
        #         at(x_m), at(self.gy.memory), gW.memory, gb.memory))
        # else:
        #     dag.push_back(ip_backweights.inner_product_backward_weights(cc_pd,
        #         at(x_m), at(self.gy.memory), gW.memory))

        # self.x_m = x_m
        self._hint = hint
        if b is None:
            self.outputs = gW,
        else:
            self.outputs = gW, gb

    def _reuse_cc(self, x, gy):
        reuse_buffer(self.x, x)
        reuse_buffer(self.gy, gy)

    def match(self, inputs, grad_outputs, hint, *args):
        if len(inputs) != self.argc:
            return False

        gy = grad_outputs[0]
        if isinstance(gy, mdarray) and gy is not self.gy:
            return False

        return (hint is self._hint)

    def __init__(self, inputs, grad_outputs, hint, pos, e=Engine()):
        super(LinearBackwardWeighs, self).__init__()
        x = inputs[0]
        gy = grad_outputs[0]
        self.argc = len(inputs)

        if self.new:
            W = inputs[1]
            b = inputs[2] if self.argc == 3 else None

            self._create_cc(x, W, b, gy, hint, e)
        else:
            self._reuse_cc(x, gy)
