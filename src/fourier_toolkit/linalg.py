import numpy as np
import opt_einsum as oe

__all__ = [
    "hadamard_outer",
    "hadamard_outer2",
    "hadamard_outer3",
]


def hadamard_outer(x: np.ndarray, *args: list[np.ndarray]) -> np.ndarray:
    r"""
    Compute Hadamard product of `x` with outer product of `args`:

    .. math::

       y = x \odot (A_{1} \otimes\cdots\otimes A_{D})

    Parameters
    ----------
    x: ndarray
        (..., N1,...,ND)
    args[k]: ndarray
        (Nk,)

    Returns
    -------
    y: ndarray
        (..., N1,...,ND)

    Note
    ----
    All inputs must share the same dtype precision.
    """
    D = len(args)
    assert all(A.ndim == 1 for A in args)
    sh = tuple(A.size for A in args)

    assert x.ndim >= D
    assert x.shape[-D:] == sh

    x_ind = (Ellipsis, *range(D))
    o_ind = (Ellipsis, *range(D))
    outer_args = [None] * (2 * D)
    for d in range(D):
        outer_args[2 * d] = args[d]
        outer_args[2 * d + 1] = (d,)

    y = oe.contract(  # (..., N1,...,ND)
        *(x, x_ind),
        *outer_args,
        o_ind,
        use_blas=True,
        optimize="auto",
    )
    return y


def hadamard_outer2(x: np.ndarray, *args: list[np.ndarray]) -> np.ndarray:
    r"""
    Compute Hadamard-reduction of `x` with (broadcasted) outer product of `args`.

    In other words, this function computes the following:

    .. math::

       y[a1,...,aD] = \sum_{p} x[p, a1,...,aD] * A1[p, a1] * ... * AD[p, aD]

    Parameters
    ----------
    x: ndarray
        (P, ..., N1,...,ND)
    args[k]: ndarray
        (P, Nk)

    Returns
    -------
    y: ndarray
        (..., N1,...,ND)

    Note
    ----
    All inputs must share the same dtype precision.
    """
    D = len(args)
    assert all(A.ndim == 2 for A in args)

    P, sh = x.shape[0], tuple(A.shape[1] for A in args)
    assert all(A.shape[0] == P for A in args)

    assert x.ndim >= D + 1
    assert x.shape[-D:] == sh

    x_ind = (0, Ellipsis, *range(1, D + 1))
    o_ind = (Ellipsis, *range(1, D + 1))
    outer_args = [None] * (2 * D)
    for d in range(D):
        outer_args[2 * d] = args[d]
        outer_args[2 * d + 1] = (0, d + 1)

    y = oe.contract(  # (..., N1,...,ND)
        *(x, x_ind),
        *outer_args,
        o_ind,
        use_blas=True,
        optimize="auto",
    )
    return y


def hadamard_outer3(x: np.ndarray, *args: list[np.ndarray]) -> np.ndarray:
    r"""
    Compute Hadamard-expansion of `x` with (broadcasted) outer product of `args`.

    In other words, this function computes the following:

    .. math::

       y[p, a1,...,aD] = x[a1,...,aD] * A1[p, a1] * ... * AD[p, aD]

    Parameters
    ----------
    x: ndarray
        (..., N1,...,ND)
    args[k]: ndarray
        (P, Nk)

    Returns
    -------
    y: ndarray
        (P, ..., N1,...,ND)

    Note
    ----
    All inputs must share the same dtype precision.
    """
    D = len(args)
    assert all(A.ndim == 2 for A in args)

    P, sh = args[0].shape[0], tuple(A.shape[1] for A in args)
    assert all(A.shape[0] == P for A in args)

    assert x.ndim >= D
    assert x.shape[-D:] == sh

    x_ind = (Ellipsis, *range(1, D + 1))
    o_ind = (0, Ellipsis, *range(1, D + 1))
    outer_args = [None] * (2 * D)
    for d in range(D):
        outer_args[2 * d] = args[d]
        outer_args[2 * d + 1] = (0, d + 1)

    y = oe.contract(  # (P, ..., N1,...,ND)
        *(x, x_ind),
        *outer_args,
        o_ind,
        use_blas=True,
        optimize="auto",
    )
    return y
