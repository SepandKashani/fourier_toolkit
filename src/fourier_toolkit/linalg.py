import numpy as np
import opt_einsum as oe

__all__ = [
    "hadamard_outer",
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
