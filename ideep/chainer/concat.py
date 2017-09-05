import six
import numpy
from chainer import function
from chainer.utils import type_check

from mkldnn.chainer.runtime import Engine
from mkldnn.compute_complex import reuse_buffer
from mkldnn.compute_complex import array
from mkldnn.compute_complex import ComputeComplex

# Most important thing

from mkldnn.api.support import at
import mkldnn.api.memory as m
import mkldnn.api.view as view
import mkldnn.api.concat as concat
import mkldnn.api.reorder as r
from mkldnn.mdarray import mdarray


class ConcatForward(ComputeComplex):
    cc_type = 'f'

    def __init__(self, xs, axis, pos=None, e=Engine()):
        super(ConcatForward, self).__init__()

        if self.new:
            self._create_cc(xs, axis, e)
        else:
            self._reuse(xs)

    def _create_cc(self, xs, axis, e):
        self.axis = axis
        xarrays = ()
        axis_dim = 0
        xs_mpdl = m.mpd_list()
        # xs_pl = primitive_list()
        xs_pl = ()
        for x in xs:
            axis_dim += x.shape[1]
            xarray = array(x, m.memory.nchw, e)
            xarrays += (xarray,)
            xs_mpdl.push_back(xarray.memory.get_primitive_desc())
            # xs_pl.push_back(xarray.memory)
            xs_pl += (at(xarray.memory), )

        cc_pd = concat.primitive_desc(axis, xs_mpdl)
        y = mdarray(cc_pd.dst_primitive_desc())
        self.dag_.push_back(concat.concat(cc_pd, xs_pl, y.memory))

        self._hint = cc_pd
        self.outputs = y,
        self.xarrays = xarrays

    def _reuse(self, xs):
        for xarray, x in zip(self.xarrays, xs):
            reuse_buffer(xarray, x)

    def match(self, inputs, axis):
        if len(self.xarrays) != len(inputs):
            return False
        for xarray, x in zip(self.xarrays, inputs):
            if xarray.shape != x.shape:
                return False
            if (isinstance(x, mdarray) and (x is not xarray)):
                return False
        return True


class ConcatBackward(ComputeComplex):
    cc_type = 'bd'

    def __init__(self, xs, gy, hint, axis,
                 pos=None, e=Engine()):
        super(ConcatBackward, self).__init__()

        if self.new:
            self._create_cc(xs, gy, hint, axis, e)
        else:
            self._reuse(xs, gy)

    def _create_cc(self, xs, gy, hint, axis, e):
        self.axis = axis
        gy = array(gy[0], m.memory.nchw, e)
        fmt = m.memory.nchw
        gy_mpd = gy.memory.get_primitive_desc()
        offsets = (0, 0, 0, 0)
        self.outputs = ()
        for x in xs:
            view_pd = view.primitive_desc(gy_mpd, x.shape, offsets)
            fmt = m.get_fmt(gy_mpd)
            assert x.dtype == numpy.dtype('float32')
            gx = mdarray(x.shape, m.memory.f32, fmt, e)
            # gx = mdarray(x.memory.get_primitive_desc())
            # gx = array(x, m.memory.nchw, e)
            reorder_pd = r.primitive_desc(view_pd.dst_primitive_desc(), gx.memory.get_primitive_desc())
            reorder_prim = r.reorder(reorder_pd, at(gy.memory), gx.memory)
            self.dag_.push_back(reorder_prim)
            self.outputs += (gx,)
            new_off = offsets[axis] + x.shape[axis]
            offsets = offsets[:axis] + (new_off,) + offsets[axis+1:]

        self.gy = gy
        self.xs = xs
        self._hint = hint

    def _reuse(self, inputs, gy):
        reuse_buffer(self.gy, gy)

    def match(self, inputs, gy, hint, axis):
        if len(self.xs) != len(inputs):
            return False
        for xarray, x in zip(self.xs, inputs):
            if xarray.shape != x.shape:
                return False
        return True


class ConcatMKLDNN(function.Function):

    """Concatenate multiple tensors towards specified axis."""

    # concat along the channel dimension by default
    def __init__(self, axis=1):
        if not isinstance(axis, int):
            raise TypeError('axis must be int')

        self.axis = axis

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() > 0)
        type_check.expect(in_types[0].ndim >
                          type_check.make_variable(self.axis, 'axis'))

        type_check.expect(
            -in_types[0].ndim <= self.axis,
            self.axis < in_types[0].ndim
        )
        ndim = type_check.eval(in_types[0].ndim)
        axis = self.axis % ndim
        for i in six.moves.range(1, type_check.eval(in_types.size())):
            type_check.expect(
                in_types[0].dtype == in_types[i].dtype,
                in_types[0].ndim == in_types[i].ndim,
            )
            for d in six.moves.range(0, ndim):
                if d == axis:
                    continue
                type_check.expect(in_types[0].shape[d] == in_types[i].shape[d])

    def forward_cpu(self, xs):
        cc = ConcatForward(xs, self.axis,
                           pos=(self.rank, self.fanout))

        self.hint = cc.hint
        y, = cc.execute_on()
        return y,

    def backward_cpu(self, xs, gy):
        cc = ConcatBackward(xs, gy, self.hint, self.axis,
                            pos=(self.rank, self.fanout))

        gx = cc.execute_on()
        return gx
