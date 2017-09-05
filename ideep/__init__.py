import sys
import ctypes

# For C++ extension to work
sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

# API lift
from ideep import api
from ideep import chainer
from ideep import compute_complex

from ideep.mdarray import mdarray

from ideep.chainer.fanout import *
