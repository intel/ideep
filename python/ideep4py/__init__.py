import numpy
import sys

from ideep4py._ideep4py import intVector  # NOQA

from ideep4py._ideep4py import mdarray  # NOQA
from ideep4py._ideep4py import mdarrayVector  # NOQA

from ideep4py._ideep4py import batchNormalization  # NOQA
from ideep4py._ideep4py import concat  # NOQA
from ideep4py._ideep4py import convolution2D  # NOQA
from ideep4py._ideep4py import convolution2DParam as conv2DParam  # NOQA
from ideep4py._ideep4py import dropout  # NOQA
from ideep4py._ideep4py import linear  # NOQA
from ideep4py._ideep4py import localResponseNormalization  # NOQA
from ideep4py._ideep4py import localResponseNormalizationParam as lrnParam  # NOQA
from ideep4py._ideep4py import pooling2D  # NOQA
from ideep4py._ideep4py import pooling2DParam as pol2DParam  # NOQA
from ideep4py._ideep4py import relu  # NOQA

from ideep4py._ideep4py import basic_acc_sum  # NOQA
from ideep4py._ideep4py import basic_copyto  # NOQA

# from ideep4py._ideep4py import dlCompression  # NOQA

from ideep4py import cosim  # NOQA


# ------------------------------------------------------------------------------
# ideep4py.mdarray allocation
# ------------------------------------------------------------------------------
dat_array = 'd'  # data array
wgt_array = 'w'  # weight array


def array(x, itype=dat_array):
    """Create a :class:`ideep4py.mdarray` object according to ``x``.

    Args:
        array (numpy.ndarray or ideep4py.mdarray):
            if ``x`` is numpy.ndarray not in C contiguous, it will be
            converted to C contiguous before ideep4py.mdarray created.
        itype (=data_type): ideep4py.mdarray created is optimized according
            ``itype`` flag.

    Returns:
        Instance of :class:`ideep4py.mdarray`.

    """
    if isinstance(x, numpy.ndarray) and \
            x.dtype == numpy.dtype('float32'):
        if x.flags.contiguous is False:
            x = numpy.ascontiguousarray(x)
        return mdarray(x, itype)
    else:
        return x


_ideep4py_ = sys.modules[__name__]


def get_array_module(array):
    return _ideep4py_


def check_ndim(inputs, supported_ndim=(2, 4)):
    # Check with ideep4py supported dimension of input data
    valid_ndim = False
    for ndim in supported_ndim:
        valid_ndim = valid_ndim or inputs[0].ndim == ndim

    if supported_ndim and not valid_ndim:
        return False
    else:
        return True


def check_type(inputs):
    if isinstance(inputs[0], numpy.ndarray):
        _should_use_ideep = True

        for x in inputs:
            _should_use_ideep = _should_use_ideep and \
                x.dtype == numpy.dtype('float32') and \
                x.size != 0
        return _should_use_ideep
    else:
        return False


def all_ready(inputs, supported_ndim=(2, 4)):
    """Check inputs dimentions and type

    The function checks ``inputs`` info and ``supported_ndim``.

    Args:
        inputs (numpy.ndarray, ideep.mdarray):
            ``inputs`` to be checked including array type, dimension
            and data type.
        supported_ndim: A tuple of ndim. ideep supports array dimension
            in either 2 or 4 only.

    Returns:
        bool: ``True`` if all conditions meet.

    """

    if check_ndim(inputs, supported_ndim) is False:
        return False
    elif isinstance(inputs[0], mdarray):
        return True
    else:
        return check_type(inputs)


def split(x, indices_or_sections, axis=0):
    if all_ready((x,)):
        offsets = intVector()
        # FIXME
        # bypass python3 issue
        for i in indices_or_sections:
            offsets.push_back(int(i))
        ys = concat.Backward(x, offsets, axis)

        if ys:
            # indices_or_sections = [0, ...]
            # axis = 0
            if axis == 0 and indices_or_sections[0] == 0:
                shape = x.shape
                shape = (0, ) + shape[1:]
                y1 = numpy.ndarray(shape, dtype=x.dtype)
                ys = list((y1,) + ys)
        else:
            # For performance improvement
            # indices_or_sections = [0]
            # axis = 0
            if axis == 0 and indices_or_sections[0] == 0 \
                    and len(indices_or_sections) == 1:
                shape = x.shape
                shape = (0, ) + shape[1:]
                y1 = numpy.ndarray(shape, dtype=x.dtype)
                ys = list((y1,) + (x,))
            # other not support scenarios
            else:
                ys = numpy.split(x, indices_or_sections, axis)
    else:
        ys = numpy.split(x, indices_or_sections, axis)

    return ys


def tanh(x):
    if all_ready((x,)):
        y = _ideep4py.tanh.Forward(array(x))  # NOQA
    else:
        y = numpy.tanh(x)

    return y


def convolution2DParam(out_dims, dy, dx, sy, sx, ph, pw, pd, pr):
    cp = conv2DParam()
    cp.out_dims = intVector()
    for d in out_dims:
        cp.out_dims.push_back(d)
    cp.dilate_y, cp.dilate_x = (dy - 1), (dx - 1)
    cp.sy, cp.sx = sy, sx
    cp.pad_lh, cp.pad_lw = ph, pw
    cp.pad_rh, cp.pad_rw = pd, pr
    return cp


def pooling2DParam(out_dims, kh, kw, sy, sx, ph, pw, pd, pr, algo):
    pp = pol2DParam()
    pp.out_dims = intVector()
    for d in out_dims:
        pp.out_dims.push_back(d)
    pp.kh, pp.kw = kh, kw
    pp.sy, pp.sx = sy, sx
    pp.pad_lh, pp.pad_lw = ph, pw
    pp.pad_rh, pp.pad_rw = pd, pr
    pp.algo_kind = algo
    return pp


pooling2DParam.pooling_max = pol2DParam.pooling_max
pooling2DParam.pooling_avg = pol2DParam.pooling_avg
pooling2DParam.pooling_avg_include_padding = \
    pol2DParam.pooling_avg_include_padding
pooling2DParam.pooling_avg_exclude_padding = \
    pol2DParam.pooling_avg_exclude_padding


def localResponseNormalizationParam(n, k, alpha, beta, algo):
    lp = lrnParam()
    lp.n = n
    lp.k = k
    lp.alpha = alpha
    lp.beta = beta
    lp.algo_kind = algo
    return lp


localResponseNormalizationParam.lrn_across_channels = \
    lrnParam.lrn_across_channels
localResponseNormalizationParam.lrn_within_channel = \
    lrnParam.lrn_within_channel
