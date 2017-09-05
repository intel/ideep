from mkldnn.api.support import at, primitive_list
from mkldnn.api import reorder as r
# from mkldnn.api import memory as m
from mkldnn.chainer.runtime import Stream

import mkldnn
import numpy
from mkldnn.mdarray import mdarray


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

# def reorder_if_must(x, expect, e, net_):
#     usr_m = x.memory
#     if (usr_m.get_primitive_desc() != expect):
#         usr_pd = usr_m.get_primitive_desc()
#         usr_fmt = m.get_fmt(usr_pd)
#         expect_fmt = m.get_fmt(expect)
#         reorded_array = mdarray(expect)
#         reorded = reorded_array.memory
#         if (usr_fmt == m.memory.nChw16c and expect_fmt == m.memory.nChw8c) \
#             or (usr_fmt == m.memory.nChw8c and expect_fmt == m.memory.nChw16c):
#             # Only support float32
#             intermediate_arr = mdarray(x.shape, m.memory.f32, m.memory.nchw, e)
#             itm_m = intermediate_arr.memory
#             net_.push_back(r.reorder(at(usr_m), itm_m))
#             net_.push_back(r.reorder(at(itm_m), reorded))
#             return reorded_array, intermediate_arr
#         else:
#             net_.push_back(r.reorder(at(usr_m), reorded))
#             return reorded_array,
#     else:
#         return x,


def reuse_buffer(d, s):
    if isinstance(s, numpy.ndarray):
        s = numpy.ascontiguousarray(s)
        d.setbuffer(s)


# XXX: move this file to another location
def array(obj, *args):
    if isinstance(obj, mkldnn.mdarray):
        return obj
    elif isinstance(obj, numpy.ndarray):
        # TODO: Do we automatically transfer?

        obj = numpy.ascontiguousarray(obj)
        return mkldnn.mdarray(obj, *args)
    else:
        raise NotImplementedError


class ComputeComplex(object):
    """MKLDNN Compute Complex.

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

        if ret and isinstance(ret, cls) and ret.match(*args, **kwargs):
            ret.new = False
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
            self.dag_ = primitive_list()
            self._hint = None

    def execute_on(self, s=None):
        if s is None:
            # XXX: Refresh everytime
            s = Stream()

        s.submit(self.dag_)
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
