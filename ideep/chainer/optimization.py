class WeightReorderOptimization(object):
    def __init__(self):
        self._reorder = -1
        self._optimized = False

    @property
    def reorder(self):
        return self._reorder

    @reorder.setter
    def reorder(self, reorder):
        self._reorder = reorder

    @property
    def optimized(self):
        return self._optimized

    @optimized.setter
    def optimized(self, optimized):
        self._optimized = optimized


def weight_optimization(func, W):
    if hasattr(func, 'W_opt'):
        W.data = func.W_opt


def weight_optimization_trigger(func):
    func.W_opt = func.W


def training_forward_optimization(func, inputs):
    if hasattr(func, 'W'):
        weight_optimization(func, inputs[1])
