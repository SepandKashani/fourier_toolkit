import numpy as np
import opt_einsum as oe
from numpy.typing import DTypeLike, NDArray

import fourier_toolkit.util as ftku


def allclose(a: NDArray, b: NDArray, dtype: DTypeLike) -> bool:
    dtype = ftku.TranslateDType(dtype).to_float()
    atol = {
        np.dtype(np.float32): 1e-5,
        np.dtype(np.float64): 1e-10,
    }[dtype]

    match = np.allclose(a, b, atol=atol)
    return match


def rel_error(a: NDArray, b: NDArray, D: int) -> NDArray:
    r"""
    Relative L2-distance between 2 vectors.

    Parameters
    ----------
    a: NDArray[float/complex]
        (..., N1,...,ND)
    b: NDArray[float/complex]
        (..., N1,...,ND)
    D: int

    Returns
    -------
    rel: NDArray[float]
        (...,)  \norm{a-b}{2} / \norm{b}{2}
    """
    axes = tuple(range(-D, 0))
    num = np.sum((a - b) * (a - b).conj(), axis=axes).real
    den = np.sum(b * b.conj(), axis=axes).real
    rel = np.sqrt(num / den)
    return rel


def max_error(a: NDArray, b: NDArray, D: int) -> NDArray:
    r"""
    Relative Linf-distance between 2 vectors.

    Parameters
    ----------
    a: NDArray[float/complex]
        (..., N1,...,ND)
    b: NDArray[float/complex]
        (..., N1,...,ND)
    D: int

    Returns
    -------
    rel: NDArray[float]
        (...,)  \norm{a-b}{\inf} / \norm{b}{\inf}
    """
    axes = tuple(range(-D, 0))
    num = np.max(abs(a - b), axis=axes)
    den = np.max(abs(b), axis=axes)
    rel = num / den
    return rel


def relclose(a: NDArray, b: NDArray, D: int, eps: float) -> bool:
    r"""
    Are `a` and `b` close up to a prescribed relative error?

    Parameters
    ----------
    a: NDArray[float/complex]
        (..., N1,...,ND)
    b: NDArray[float/complex]
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
    r_err = rel_error(a, b, D)
    close = np.all(r_err <= eps)
    return close


def inner_product(x: NDArray, y: NDArray, D: int) -> NDArray:
    """
    Compute stack-wize inner-product.

    Parameters
    ----------
    x: NDArray[float/complex]
        (..., N1,...,ND)
    y: NDArray[float/complex]
        (..., N1,...,ND)
    D: int
        Rank of inputs.

    Returns
    -------
    z: NDArray[float/complex]
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
