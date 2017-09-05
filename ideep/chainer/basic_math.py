from mkldnn.chainer.runtime import Engine
from mkldnn.compute_complex import ComputeComplex, reuse_buffer, reorder_if_must

# Most important thing
from mkldnn.api.support import at
import mkldnn.api.memory as m
import mkldnn.api.sum as sum
from mkldnn.mdarray import mdarray


class AddForward(ComputeComplex):
    cc_type = 'f'

    def __init__(self, inputs, pos=None, e=Engine()):
        super(AddForward, self).__init__()

        if self.new:
            self._create_cc(inputs, e)
        else:
            self._reuse(inputs)

    def _create_cc(self, inputs, e):
        x0, x1 = inputs[:2]
        xs_mpdl = m.mpd_list()
        xs_pl = ()
        scales = m.vectord()

        self.x0 = x0
        self.x1 = x1
        self.x1_reordered = reorder_if_must(x1, x0.memory.get_primitive_desc(), e, self.dag_)[0]
        scales.push_back(1.0)
        scales.push_back(1.0)
        xs_mpdl.push_back(x0.memory.get_primitive_desc())
        xs_mpdl.push_back(self.x1_reordered.memory.get_primitive_desc())
        cc_pd = sum.primitive_desc(scales, xs_mpdl)

        xs_pl = (at(x0.memory), at(self.x1_reordered.memory))
        y = mdarray(cc_pd.dst_primitive_desc())

        self.dag_.push_back(sum.sum(cc_pd, xs_pl, y.memory))
        self.outputs = y,

    def _reuse(self, inputs):
        x0, x1 = inputs[:2]
        reuse_buffer(self.x0, x0)
        reuse_buffer(self.x1, x1)

    def match(self, inputs):
        x0, x1 = inputs[:2]
        return (x0 is self.x0) and (x1 is self.x1)


class AddMKLDNN():

    def __call__(self, inputs, in_pos):
        cc = AddForward(inputs, pos=in_pos)
        y, = cc.execute_on()
        return y
