import sys
import unittest

import numpy
import ideep4py
from ideep4py import intVector, mdarrayVector, concat

try:
    import testing
except Exception as ex:
    print('*** testing directory is missing: %s' % ex)
    sys.exit(-1)


@testing.parameterize(*testing.product_dict(
    [
        {'shape': (7, 2, 3, 5), 'axis': 0, 'section': [2, 5],
         'slices': [[slice(None, 2)], [slice(2, 5)],
                    [slice(5, None)]]},
        {'shape': (2, 7, 3, 5), 'axis': 1, 'section': [2, 5],
         'slices': [[slice(None), slice(None, 2)], [slice(None), slice(2, 5)],
                    [slice(None), slice(5, None)]]},
        {'shape': (2, 3, 7, 5), 'axis': 2, 'section': [2, 5],
         'slices': [[slice(None), slice(None), slice(None, 2)],
                    [slice(None), slice(None), slice(2, 5)],
                    [slice(None), slice(None), slice(5, None)]]},
        {'shape': (2, 3, 5, 7), 'axis': 3, 'section': [2, 5],
         'slices': [[slice(None), slice(None), slice(None), slice(None, 2)],
                    [slice(None), slice(None), slice(None), slice(2, 5)],
                    [slice(None), slice(None), slice(None), slice(5, None)]]},
        {'shape': (60, 33, 3, 3), 'axis': 0, 'section': [12, 48],
         'slices': [[slice(None, 12)],
                    [slice(12, 48)],
                    [slice(48, None)]]},
        {'shape': (33, 60, 3, 3), 'axis': 1, 'section': [12, 48],
         'slices': [[slice(None), slice(None, 12)],
                    [slice(None), slice(12, 48)],
                    [slice(None), slice(48, None)]]},
        {'shape': (33, 3, 60, 3), 'axis': 2, 'section': [12, 48],
         'slices': [[slice(None), slice(None), slice(None, 12)],
                    [slice(None), slice(None), slice(12, 48)],
                    [slice(None), slice(None), slice(48, None)]]},
        {'shape': (33, 3, 3, 60), 'axis': 3, 'section': [12, 48],
         'slices': [[slice(None), slice(None), slice(None), slice(None, 12)],
                    [slice(None), slice(None), slice(None), slice(12, 48)],
                    [slice(None), slice(None), slice(None), slice(48, None)]]},
    ],
    [
        {'dtype': numpy.float32},
    ],
))
class TestConcatPyF32(unittest.TestCase):

    def setUp(self):
        self.y = numpy.arange(
            numpy.prod(self.shape), dtype=self.dtype).reshape(self.shape)
        self.xs = [self.y[s] for s in self.slices]

    def check_forward(self, xs_data, y_data, axis):
        xs = tuple(x_data for x_data in xs_data)
        xs_mdarray = mdarrayVector()
        for yi in xs:
            if isinstance(yi, numpy.ndarray):
                if yi.flags.contiguous is False:
                    yi = numpy.ascontiguousarray(yi)
            yi = ideep4py.mdarray(numpy.ascontiguousarray(yi))
            xs_mdarray.push_back(yi)
        y_act = concat.Forward(xs_mdarray, self.axis)
        y_act = numpy.array(y_act, dtype=self.dtype)

        numpy.testing.assert_allclose(y_data, y_act, atol=0, rtol=0)

    def test_forward_cpu(self):
        self.check_forward(self.xs, self.y, axis=self.axis)

    def check_backward(self, xs_data, y_data, axis):
        xs = tuple(x_data for x_data in xs_data)
        xs_mdarray = mdarrayVector()
        for yi in xs:
            if isinstance(yi, numpy.ndarray):
                if yi.flags.contiguous is False:
                    yi = numpy.ascontiguousarray(yi)
            yi = ideep4py.mdarray(numpy.ascontiguousarray(yi))
            xs_mdarray.push_back(yi)
        y_data = ideep4py.mdarray(y_data)
        offsets = intVector()
        # FIXME
        for i in self.section:
            offsets.push_back(i)
        x_act_mdarray = concat.Backward(y_data, offsets, self.axis)
        i = 0
        for x in xs:
            x_act = numpy.array(x_act_mdarray[i], dtype=self.dtype)
            numpy.testing.assert_allclose(
                x, x_act, atol=0, rtol=0)
            i = i + 1

    def test_backward_cpu(self):
        self.check_backward(self.xs, self.y, axis=self.axis)


testing.run_module(__name__, __file__)
