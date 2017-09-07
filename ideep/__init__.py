import sys  # NOQA
import ctypes  # NOQA

# For C++ extension to work
sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

# API lift
from ideep import api  # NOQA
from ideep import chainer  # NOQA
from ideep import compute_complex  # NOQA

from ideep.mdarray import mdarray  # NOQA

from ideep.chainer.fanout import FanoutRecorder  # NOQA
