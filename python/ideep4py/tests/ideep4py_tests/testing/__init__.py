from testing import parameterized  # NOQA
from testing.parameterized import parameterize  # NOQA
from testing.parameterized import product  # NOQA
from testing.parameterized import product_dict  # NOQA
from testing.random import fix_random  # NOQA


def run_module(name, file):
    """Run current test cases of the file.

    Args:
        name: __name__ attribute of the file.
        file: __file__ attribute of the file.
    """

    if name == '__main__':
        import pytest
        pytest.main([file, '-vvs', '-x', '--pdb'])
