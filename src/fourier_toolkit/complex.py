import numpy as np

import fourier_toolkit.util as ftk_util

__all__ = [
    "cexp",
]


def cexp(x: np.ndarray) -> np.ndarray:
    """
    Computes code below more efficiently:

    .. code-block:: python

       y = np.exp(1j * x)

    Parameters
    ----------
    x: ndarray[float]

    Returns
    -------
    y: ndarray[complex]
    """
    translate = ftk_util.TranslateDType(x.dtype)
    cdtype = translate.to_complex()

    y = np.empty(x.shape, dtype=cdtype)
    np.cos(x, out=y.real)
    np.sin(x, out=y.imag)
    return y
