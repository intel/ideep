from chainer import function
from chainer.utils import type_check

from mkldnn.chainer.runtime import Engine
from mkldnn.compute_complex import ComputeComplex, array, reuse_buffer

# Most important thing
from mkldnn.api.support import forward_training, lrn_across_channels, at
import mkldnn.api.memory as m
import mkldnn.api.lrn_forward as lrn_forward
import mkldnn.api.lrn_backward as lrn_backward
import mkldnn.api.cosim_dump as cdump
from mkldnn.api.cosim_dump import *
from mkldnn.mdarray import mdarray


class LrnForward(ComputeComplex):
    cc_type = 'f'

    def __init__(self, inputs, n=5, k=2, alpha=1e-4, beta=.75,
                 pos=None, e=Engine()):
        super(LrnForward, self).__init__()

        x = inputs[0]
        if self.new:
            self._create_cc(x, n, k, alpha, beta, e)
        else:
            self._reuse(x)

    def _create_cc(self, x, n, k, alpha, beta, e):
        self.n = n
        self.k = k
        self.alpha = alpha
        self.beta = beta
        # TODO: check avx512?
        self.x = array(x, m.memory.nchw, e)
        x_md = self.x.memory.get_primitive_desc().desc()
        cc_d = lrn_forward.desc(forward_training, lrn_across_channels, x_md,
                                n, alpha, beta, k)
        cc_pd = lrn_forward.primitive_desc(cc_d, e)
        y = mdarray(cc_pd.dst_primitive_desc())
        ws = mdarray(cc_pd.workspace_primitive_desc())

        self.dag_.push_back(lrn_forward.lrn_forward(cc_pd, at(self.x.memory), ws.memory, y.memory))
        self._hint = cc_pd
        self.outputs = y,
        self.ws = ws

    def _reuse(self, x):
        reuse_buffer(self.x, x)

    def match(self, inputs, n, k, alpha, beta):
        x = inputs[0]
        return ((self.x.shape == x.shape) and
                (self.n == n) and
                (self.k == k) and
                (self.alpha == alpha) and
                (self.beta == beta))


class LrnBackward(ComputeComplex):
    cc_type = 'bd'

    def __init__(self, inputs, gy, hint, ws, n=5, k=2,
                 alpha=1e-4, beta=.75, pos=None, e=Engine()):
        super(LrnBackward, self).__init__()

        x = inputs[0]
        if self.new:
            self._create_cc(x, gy, hint, ws, n, k, alpha, beta, e)
        else:
            self._reuse(x, gy)

    def _create_cc(self, x, gy, hint, ws, n, k, alpha, beta, e):
        self.n = n
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.x = array(x, m.memory.nchw, e)
        x_md = self.x.memory.get_primitive_desc().desc()
        # TODO: check avx512?
        gy = array(gy, m.memory.nchw, e)
        gy_md = gy.memory.get_primitive_desc().desc()
        cc_d = lrn_backward.desc(lrn_across_channels, x_md, gy_md, n, alpha, beta, k)
        cc_pd = lrn_backward.primitive_desc(cc_d, e, hint)

        gx = mdarray(cc_pd.diff_src_primitive_desc())
        self.dag_.push_back(lrn_backward.lrn_backward(cc_pd, at(self.x.memory), at(gy.memory),
                            at(ws.memory), gx.memory))
        self._hint = hint
        self.gy = gy
        self.outputs = gx,

    def _reuse(self, x, gy):
        reuse_buffer(self.x, x)
        reuse_buffer(self.gy, gy)

    def match(self, inputs, gy, hint, ws, n, k, alpha, beta):
        return (hint is self._hint)


class LrnMKLDNN(function.Function):

    """Cross-channel normalization function used in AlexNet."""

    def __init__(self, n=5, k=2, alpha=1e-4, beta=.75):
        self.n = n
        self.k = k
        self.alpha = alpha
        self.beta = beta

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(
            x_type.dtype.kind == 'f',
            x_type.ndim >= 2,
        )

    def forward_cpu(self, x):
        cc = LrnForward(x, n=self.n, k=self.k, alpha=self.alpha*self.n, beta=self.beta,
                        pos=(self.rank, self.fanout))

        self.hint = cc.hint
        self.ws = cc.ws
        y, = cc.execute_on()
        return y,

    def backward_cpu(self, x, gy):
        cc = LrnBackward(x, gy[0], self.hint, self.ws,
                         n=self.n, k=self.k, alpha=self.alpha*self.n, beta=self.beta,
                         pos=(self.rank, self.fanout))

        gx, = cc.execute_on()
        return gx,

    def cpu_cosim_dump_inner(self, in_data, out_grad=None):
        cd = None
        if out_grad is None:
            cd = cdump.cosim_dump(cdump_op_lrn_forward)
        else:
            cd = cdump.cosim_dump(cdump_op_lrn_backward)

        x = array(in_data[0], m.memory.nchw, Engine())
        cd.dump_memory(cdump_src_memory, x.memory)

        if out_grad is not None:
            gy = array(out_grad[0], m.memory.nchw, Engine())
            cd.dump_memory(cdump_diff_dst_memory, gy.memory)

        cd.dump_int_parms(cdump_lrn_local_size, 1, self.n)
        cd.dump_double_parms(cdump_lrn_doulbe_parms, 3, self.k, self.alpha, self.beta)
