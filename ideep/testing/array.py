import numpy as np
from ideep import mdarray


def assert_allclose(x, y, atol=1e-5, rtol=1e-4, verbose=True):
    """Asserts if some corresponding element of x and y differs too much.

    This function can handle both `ndarray` and `ideep.mdarray` arrays simultaneously.

    Args:
        x: Left-hand-side array.
        y: Right-hand-side array.
        atol (float): Absolute tolerance.
        rtol (float): Relative tolerance.
        verbose (bool): If ``True``, it outputs verbose messages on error.

    """
    if isinstance(x, mdarray):
        x = np.array(x)
    if isinstance(y, mdarray):
        y = np.array(y)

    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)

    try:
        np.testing.assert_allclose(
            x, y, atol=atol, rtol=rtol, verbose=verbose)
    except Exception:
        print('error:', np.abs(x - y).max())
        raise

