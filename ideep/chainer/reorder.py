from mkldnn.chainer.runtime import Engine, Stream
from mkldnn.api.support import primitive_list
import mkldnn.api.memory as m
import numpy


class ReorderMKLDNN(object):

    def __init__(self, src, dst_shape, dst_dtype, dst_format):
        self.src = src

        self.fwd_dag = primitive_list()
        self.bwd_dag = primitive_list()

        self.src_mpd = src.memory.get_primitive_desc()
        dst_dtype = m.memory.f32 if dst_dtype is numpy.float32 or \
            dst_dtype.kind is 'f' else m.memory.s32
        self.expected_mpd = m.primitive_desc(m.desc(dst_shape, dst_dtype, dst_format), Engine())

        if self.src_mpd != self.expected_mpd:
            self.dst = mdarray(self.expected_mpd)
            self.fwd_dag.push_back(
                r.reorder(at(self.src.memory), self.dst.memory))
            self.bwd_dag.push_back(
                r.reorder(at(self.dst.memory), self.src.memory))
        else:
            self.dst = self.src

    def forward(self):
        if self.fwd_dag.empty() is False:
            s = Stream()
            s.submit(self.fwd_dag)
            s.wait()
        return self.dst

    def backward(self):
        if self.bwd_dag.empty() is False:
            s = Stream()
            s.submit(self.bwd_dag)
            s.wait()
        return self.src


def reorder_buffer(src, expected):
    return ReorderMKLDNN(src, expected)
