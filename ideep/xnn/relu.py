from ideep.cpu_engine import Engine
from ideep.mdarray import mdarray
from ideep.array import array
from ideep.api.support import forward, eltwise_relu, at

import ideep.compute_complex as CC
import ideep.api.memory as m
import ideep.api.eltwise_forward as eltwise_forward
import ideep.api.eltwise_backward as eltwise_backward


class ReLUForward(CC.ComputeComplex):
    cc_type = 'f'

    def __init__(self, inputs, pos=(0, 0), e=Engine()):
        x = inputs[0]
        super(ReLUForward, self).__init__()

        if self.new:
            self._create_cc(x, e)
        else:
            self._reuse_cc(x)

    def _create_cc(self, x, e=Engine()):
        if x.ndim == 2:
            fmt = m.memory.nc
        elif x.ndim == 4:
            fmt = m.memory.nchw

        x = array(x, fmt, e)
        mem_pd = x.memory.get_primitive_desc()

        cc_d = eltwise_forward.desc(
            forward, eltwise_relu, mem_pd.desc(), 0.0, 0.0)
        cc_pd = eltwise_forward.primitive_desc(cc_d, e)

        y = mdarray(cc_pd.dst_primitive_desc())

        self.x = x
        self.dag.push_back(eltwise_forward.eltwise_forward(cc_pd,
                            at(x.memory), y.memory))

        self._hint = cc_pd
        self.outputs = y,

#    def match(self, inputs, *args):
#        # TODO: refine it
#        x = inputs[0]
#        if(isinstance(x, mdarray) and (x is not self.x)):
#            return False
#        return self.x.shape == x.shape
#
#    def _reuse_cc(self, x):
#        reuse_buffer(self.x, x)


class ReLUBackward(CC.ComputeComplex):
    cc_type = 'bd'

    def __init__(self, inputs, grad_outputs, hint, pos=(0, 0), e=Engine()):
        x = inputs[0]
        gy = grad_outputs[0]
        super(ReLUBackward, self).__init__()

        if self.new:
            self._create_cc(x, gy, hint, e)
        else:
            self._reuse_cc(x, gy)

#    def match(self, inputs, grad_outpus, hint, *args):
#        # TODO: refine it
#        return (hint is self._hint)

    def _create_cc(self, x, gy, hint, e=Engine()):
        if x.ndim == 2:
            fmt = m.memory.nc
        else:
            fmt = m.memory.nchw

        x = array(x, fmt, e)
        gy = array(gy, fmt, e)

        diff_pd = gy.memory.get_primitive_desc()
        outputs = CC.reorder_if_must(x, diff_pd, e, self.dag)

        if len(outputs) == 2:
            x, self.itm_arr = outputs[:2]
        else:
            x = outputs[0]

        mem_pd = x.memory.get_primitive_desc()

        cc_d = eltwise_backward.desc(eltwise_relu, diff_pd.desc(),
                                     mem_pd.desc(), 0.0, 0.0)
        cc_pd = eltwise_backward.primitive_desc(cc_d, e, hint)

        # gx = mdarray(cc_pd.diff_src_primitive_desc())
        # print("gx.format=", m.get_fmt(cc_pd.diff_src_primitive_desc()))
        gx = gy

        self.dag.push_back(eltwise_backward.eltwise_backward(cc_pd,
                            at(x.memory), at(gy.memory), gx.memory))

        self.x = x
        self.gy = gy
        self._hint = hint
        self.outputs = gx,

#    def _reuse_cc(self, x, gy):
#        reuse_buffer(self.x, x)
#        reuse_buffer(self.gy, gy)
