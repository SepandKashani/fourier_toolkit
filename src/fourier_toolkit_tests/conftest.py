import numpy as np
import numpy.typing as npt
import opt_einsum as oe

import fourier_toolkit.util as ftk_util


def allclose(
    a: np.ndarray,
    b: np.ndarray,
    dtype: npt.DTypeLike,
) -> bool:
    dtype = ftk_util.TranslateDType(dtype).to_float()
    atol = {
        np.dtype(np.float32): 1e-4,
        np.dtype(np.float64): 1e-8,
    }[dtype]

    match = np.allclose(a, b, atol=atol)
    return match


def relclose(
    a: np.ndarray,
    b: np.ndarray,
    D: int,
    eps: float,
) -> bool:
    r"""
    Are `a` and `b` close up to a prescribed relative error?

    Parameters
    ----------
    a: ndarray[float/complex]
        (..., N1,...,ND)
    b: ndarray[float/complex]
        (..., N1,...,ND)
    D: int
    eps: float
        [0, 1[ tolerance.

    Returns
    -------
    close: bool
        \norm{a - b}{2} <= eps * \norm{b}{2}
    """
    assert 0 <= eps < 1
    axes = tuple(np.arange(-D, 0))
    lhs = np.sum((a - b) * (a - b).conj(), axis=axes).real  # (...,)
    rhs = np.sum(b * b.conj(), axis=axes).real  # (...,)
    close = np.all(lhs <= (eps**2) * rhs)
    return close


def inner_product(
    x: np.ndarray,
    y: np.ndarray,
    D: int,
) -> np.ndarray:
    """
    Compute stack-wize inner-product.

    Parameters
    ----------
    x: ndarray
        (..., N1,...,ND)
    y: ndarray
        (..., N1,...,ND)
    D: int
        Rank of inputs.

    Returns
    -------
    z: ndarray
        (...,) inner-products <x,y>.
    """
    x_ind = y_ind = (Ellipsis, *range(D))
    z_ind = (Ellipsis,)

    z = oe.contract(
        *(x, x_ind),
        *(y.conj(), y_ind),
        z_ind,
    )
    return z
