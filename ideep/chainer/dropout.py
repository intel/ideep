import mkldnn.api.memory as m
import numpy as np
from chainer import function
from chainer.utils import type_check
from mkldnn.mdarray import mdarray
from mkldnn.api.dropout import dropout_f32
from mkldnn.chainer.runtime import Engine
from mkldnn.compute_complex import array, ComputeComplex


def _format(ndim):
    if ndim == 2:
        return m.memory.nc
    elif ndim == 4:
        return m.memory.nchw
    else:
        return NotImplemented


class DropoutForward(ComputeComplex):
    cc_type = 'f'

    def __init__(self, inputs, dropout_ratio, pos=(0, 0), e=Engine()):
        super(DropoutForward, self).__init__()

        self.x = array(inputs[0], _format(inputs[0].ndim), Engine())

        if self.new:
            self._create_cc(inputs[0], dropout_ratio, e)

    def _create_cc(self, x, dropout_ratio, e=Engine()):
        self.dropout_op = dropout_f32(dropout_ratio)

        self.mask = np.ndarray(shape=x.shape, dtype=np.float32)
        self._mask = array(self.mask, _format(self.mask.ndim), e)

        self._hint = mdarray(self.x.memory.get_primitive_desc())

    def match(self, inputs, *args):
        # TODO: refine it
        x = inputs[0]
        if(isinstance(x, mdarray) and (x is not self.x)):
            return False
        return self.x.shape == x.shape

    def execute_on(self, s=None):
        self.dropout_op.forward(self.x, self._mask, self._hint)
        return self._hint,


class DropoutBackward(ComputeComplex):
    cc_type = 'bd'

    def __init__(self, dropout_op, mask, gy, hint, pos=(0, 0), e=Engine()):
        super(DropoutBackward, self).__init__()
        self._dropout_op = dropout_op
        self._mask = mask
        self.gy = array(gy[0], _format(gy[0].ndim), e)

        if self.new:
            self._create_cc(hint)

    def _create_cc(self, hint):
        self.gx = mdarray(self.gy.memory.get_primitive_desc())
        self._hint = hint

    def match(self, dropout_op, mask, gy, hint, *args):
        # TODO: refine it
        return (hint is self._hint)

    def execute_on(self, s=None):
        self._dropout_op.backward(self.gy, self._mask, self.gx)
        return self.gx,


class DropoutFunctionMKLDNN(function.Function):
    def __init__(self, dropout_ratio):
        self.dropout_ratio = dropout_ratio

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward(self, x):
        cc = DropoutForward(x, self.dropout_ratio, pos=(self.rank, self.fanout))

        self.mask = cc.mask
        self._mask = cc._mask
        self.dropout_op = cc.dropout_op
        self.hint = cc.hint

        return cc.execute_on()

    def backward(self, x, gy):
        cc = DropoutBackward(self.dropout_op, self._mask, gy, self.hint, pos=(self.rank, self.fanout))
        return cc.execute_on()
