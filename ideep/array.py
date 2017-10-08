import numpy

from ideep.api import memory as m
from ideep.chainer.runtime import Engine
from ideep.mdarray import mdarray


def array(obj, *args):
    """Convert the input to an mdarray

    Parameters
    ----------
    obj : numpy ndarray object

    """

    if isinstance(obj, mdarray):
        return obj
    elif isinstance(obj, numpy.ndarray):
        obj = numpy.ascontiguousarray(obj)
        return mdarray(obj, *args)
    else:
        raise NotImplementedError


def warray(w):
    fmt = None
    if w.ndim == 1:
        fmt = m.memory.x
    elif w.ndim == 2:
        fmt = m.memory.oi
    elif w.ndim == 4:
        fmt = m.memory.oihw
    else:
        raise NotImplementedError

    if w.dtype != numpy.float32:
        raise NotImplementedError

    e = Engine()
    return mdarray(w, fmt, e)
