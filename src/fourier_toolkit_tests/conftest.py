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


def rel_error(
    a: np.ndarray,
    b: np.ndarray,
    D: int,
) -> np.ndarray:
    r"""
    Relative distance between 2 vectors.

    Parameters
    ----------
    a: ndarray[float/complex]
        (..., N1,...,ND)
    b: ndarray[float/complex]
        (..., N1,...,ND)
    D: int

    Returns
    -------
    rel: ndarray[float]
        (...,)  \norm{a-b}{2} / \norm{b}{2}
    """
    axes = tuple(range(-D, 0))
    num = np.sum((a - b) * (a - b).conj(), axis=axes).real
    den = np.sum(b * b.conj(), axis=axes).real
    rel = np.sqrt(num / den)
    return rel


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
    r_err = rel_error(a, b, D)
    close = np.all(r_err <= eps)
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


def generate_point_cloud(
    N_point: int,
    D: int,
    bbox_dim: np.ndarray[float],
    N_blk: np.ndarray[int],
    sparsity_ratio: float,
    rng: np.random.Generator,
):
    # Generate (N_point+2, D) points distributed into sub-sections of a `bbox_dim`-sized cubeoid.
    # There are `N_blk` sub-sections in the `bbox_dim`-sized volume, but only a fraction `sparsity_ratio` of them contain points.
    # The 2 extra points guarantee point cloud fully fills `bbox_dim`.

    assert N_point > 0
    assert D > 0
    bbox_dim = ftk_util.broadcast_seq(bbox_dim, D, np.double)
    assert np.all(bbox_dim > 0)
    N_blk = ftk_util.broadcast_seq(N_blk, D, np.int64)
    assert np.all(N_blk > 0)
    assert 0 < sparsity_ratio <= 1

    # find blk centers
    blk_idx = np.stack(  # (N_blk1,...,N_blkD, D)
        np.meshgrid(*[np.arange(n) for n in N_blk], indexing="ij"),
        axis=-1,
    )
    blk_dim = bbox_dim / N_blk  # (D,)
    blk_center = (blk_idx * blk_dim + blk_dim / 2).reshape(-1, D)  # (N_blk.prod, D)
    # keep fraction `sparsity_ratio` of the blocks
    blk_idx = rng.choice(
        a=len(blk_center),
        size=int(np.ceil(len(blk_center) * sparsity_ratio)),
        replace=False,
    )
    blk_center = blk_center[blk_idx]  # (N_blk, D)
    N_blk = len(blk_center)

    # add points to each sub-partition
    blk_idx = rng.integers(0, N_blk, N_point)  # sub-block each point located in
    offset = rng.uniform(-0.49, 0.49, (N_point, D)) * blk_dim  # (N_point, D)
    pts = blk_center[blk_idx] + offset

    # the 2 extra points guarantee point cloud fully fills `bbox_dim`.
    pts = np.pad(pts, pad_width=((0, 2), (0, 0)))
    pts[-2] = 0
    pts[-1] = bbox_dim

    # add global offset
    pts += rng.uniform(-1, 1, D)
    return pts
