import nose

from ideep.testing import array  # NOQA
from ideep.testing import helper  # NOQA
from ideep.testing import parameterized  # NOQA


from ideep.testing.array import assert_allclose  # NOQA
from ideep.testing.helper import with_requires  # NOQA
from ideep.testing.parameterized import parameterize  # NOQA
from ideep.testing.parameterized import product  # NOQA
from ideep.testing.parameterized import product_dict  # NOQA


def run_module(name, file):
    """Run current test cases of the file.

    Args:
        name: __name__ attribute of the file.
        file: __file__ attribute of the file.
    """
    if name == '__main__':

        nose.runmodule(argv=[file, '-vvs', '-x', '--ipdb', '--ipdb-failure'],
                       exit=False)
