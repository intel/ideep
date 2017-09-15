import numpy as np
import unittest
import ideep.api.memory as m

from ideep import testing
from ideep.compute_complex import array
from ideep.chainer.runtime import Engine


@testing.parameterize(*testing.product({
    'shape': [
        (1, 1), (2, 2), (4, 4), (8, 8), (16, 16), (32, 32), (64, 64), (128, 128), (256, 256),
        (4, 3), (3, 7), (47, 31)
    ],
    'format': [m.memory.nc, m.memory.oi, m.memory.io]
}))
class TestElementwise2D(unittest.TestCase):
    def setUp(self):
        self.x = np.random.random(self.shape).astype(np.float32)
        x_md = np.array(self.x)
        self.y = np.random.random(self.shape).astype(np.float32)
        y_md = np.array(self.y)

        self.x_md = array(x_md, self.format, Engine())
        self.y_md = array(y_md, self.format, Engine())

    def test_add(self):
        z = self.x + self.y
        z_md = self.x_md + self.y_md
        testing.assert_allclose(z, z_md)

    def test_mul(self):
        z = self.x * self.y
        z_md = self.x_md * self.y_md
        testing.assert_allclose(z, z_md)

    def test_div(self):
        z = self.x / self.y
        z_md = self.x_md / self.y_md
        testing.assert_allclose(z, z_md)

    def test_min(self):
        z = self.x - self.y
        z_md = self.x_md - self.y_md
        testing.assert_allclose(z, z_md)

    def test_add_inplace(self):
        self.x += self.y
        self.x_md += self.y_md
        testing.assert_allclose(self.x, self.x_md)

    def test_mul_inplace(self):
        self.x *= self.y
        self.x_md *= self.y_md
        testing.assert_allclose(self.x, self.x_md)

    def test_div_inplace(self):
        self.x /= self.y
        self.x_md /= self.y_md
        testing.assert_allclose(self.x, self.x_md)

    def test_min_inplace(self):
        self.x -= self.y
        self.x_md -= self.y_md
        testing.assert_allclose(self.x, self.x_md)


testing.run_module(__name__, __file__)
