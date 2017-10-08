import numpy

from ideep.api.support import at, primitive_list
from ideep.api import reorder as r
from ideep.chainer.runtime import Stream
from ideep.mdarray import mdarray


def reorder_if_must(x, expect, e, net_):
    usr_m = x.memory
    if (usr_m.get_primitive_desc() != expect):
        reorded_array = mdarray(expect)
        reorded = reorded_array.memory
        reorder = r.reorder(at(usr_m), reorded)
        net_.push_back(reorder)

        return reorded_array,
    else:
        return x,


def reuse_buffer(d, s):
    if isinstance(s, numpy.ndarray):
        s = numpy.ascontiguousarray(s)
        d.setbuffer(s)


class ComputeComplex(object):

    """MKLDNN Compute Complex.

    This class implements abstract interfaces for creating and reusing MKLDNN
    primitive and necessary processes to finish a MKLDNN layer computation.
    Currently it only supports static reuse paradim which rely on (rank,
    fanout) coordinate to record and retrieve the same CC, avoiding the
    overhead of creating and desctroying MKLDNN primitives.

    Attributes:
        rank: Layer number where this Compute Complex is in.
        fanout: Branch number of this Compute Complex in a layer.
        dag: Primitives that comprise a MKLDNN layer computation.
        hint: Forward primitive descriptor for creating backward one
    """

    cache_f = {}
    cache_bd = {}
    cache_bw = {}

    cache = {'f': cache_f, 'bd': cache_bd, 'bw': cache_bw}

    def __new__(cls, *args, **kwargs):
        pos = kwargs.pop('pos')
        assert isinstance(pos, tuple)

        cache = cls.cache[cls.cc_type]
        ret = cache.get(pos)

        if configuration.config.train:
            if ret and isinstance(ret, cls) and ret.match(*args, **kwargs):
                ret.new = False
            else:
                ret = super(ComputeComplex, cls).__new__(cls)
                # print("Create new CC: ", ret)
                ret.new = True
                cache[pos] = ret
                ret.pos = pos
        else:
            ret = super(ComputeComplex, cls).__new__(cls)
            # print("Create new CC: ", ret)
            ret.new = True
            cache[pos] = ret
            ret.pos = pos
        return ret

    def __init__(self):
        if self.new:
            self.rank = -1
            self.fanout = -1
            self.dag = primitive_list()
            self._hint = None
            self.output = None

    def execute_on(self, s=None):
        if s is None:
            # XXX: Refresh everytime
            s = Stream()

        s.submit(self.dag)
        s.wait()
        return self.outputs

    def matching(self, inputs):
        raise NotImplementedError

    @staticmethod
    def get_bd_cc(hint, pos=(0, 0)):
        cc = ComputeComplex.cache_bd.get(pos)
        if cc and (hint is cc._hint):
            ret = cc
        else:
            ret = None
        return ret

    @property
    def hint(self):
        return self._hint
