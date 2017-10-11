import numpy

from ideep.api import memory as m
from ideep.cpu_engine import Engine
from ideep.mdarray import mdarray


def as_tensor(obj, fmt):
    """Convert the input to an internal tensor acceptable by MKL-DNN

    Parameters
    ----------
    obj : object support buffer protocol
    fmt : tensor data format (m.nchw, m.oihw, etc.)

    """

    if isinstance(obj, mdarray):
        return obj
    elif isinstance(obj, numpy.ndarray):
        obj = numpy.ascontiguousarray(obj)
        return mdarray(obj, fmt)
    else:
        raise NotImplementedError


def w_tensor(W):
    """Convert the input to an weight tensor of MKL-DNN

    Paramters
    ---------
    W : object support buffer protocol

    """

    if W.ndim == 1:
        fmt = m.memory.x
    elif W.ndim == 2:
        fmt = m.memory.oi
    elif W.ndim == 4:
        fmt = m.memory.oihw
    else:
        raise NotImplementedError

    if W.dtype != numpy.float32:
        raise NotImplementedError

    return mdarray(W, fmt, Engine())
