import array_api_compat as aac
import array_api_extra as aae
import opt_einsum as oe

import fourier_toolkit.typing as ftkt
import fourier_toolkit.util as ftku


def similar(a: ftkt.Array, b: ftkt.Array) -> bool:
    """
    True if `a` and `b` have the same (backend, dtype, device).
    """
    same_backend = aac.array_namespace(a) == aac.array_namespace(b)
    same_dtype = a.dtype == b.dtype
    same_device = a.device == b.device

    return all([same_backend, same_dtype, same_device])


def allclose(a: ftkt.Array, b: ftkt.Array, dtype: ftkt.DType) -> bool:
    r"""
    Absolute closeness between 2 arrays.

    Parameters
    ----------
    a, b: Array[int,float,complex]
    dtype: DType
        Precision at which closeness should be assessed.
        `dtype` should be transformable into the float32/64 type via :py:class:`fourier_toolkit.util.TranslateDType`.
    """
    xp = aac.array_namespace(a)

    fdtype = ftku.TranslateDType(
        xp.asarray([], dtype=dtype),
    ).to_float()

    if fdtype == xp.float32:
        atol = 1e-6
    elif fdtype == xp.float64:
        atol = 1e-12
    else:
        raise ValueError

    match = xp.all(aae.isclose(a, b, atol=atol, equal_nan=True))
    return bool(match)


def rel_l2_distance(a: ftkt.Array, b: ftkt.Array, D: int) -> ftkt.Array:
    r"""
    Relative L2 distance between 2 vectors.

    Parameters
    ----------
    a: Array[float,complex]
        (..., N1,...,ND)
    b: Array[float,complex]
        (..., N1,...,ND)
    D: int

    Returns
    -------
    rel_dist: Array[float]
        (...,)  \norm{a-b}{2} / \norm{b}{2}
    """
    assert D > 0
    xp = aac.array_namespace(a)

    axis = tuple(range(-D, 0))
    num = xp.real(xp.sum((a - b) * xp.conj(a - b), axis=axis))
    den = xp.real(xp.sum(b * xp.conj(b), axis=axis))
    rel_dist = xp.sqrt(num / den)
    return rel_dist


def rel_linf_distance(a: ftkt.Array, b: ftkt.Array, D: int) -> ftkt.Array:
    r"""
    Relative L-infinity distance between 2 vectors.

    Parameters
    ----------
    a: Array[float,complex]
        (..., N1,...,ND)
    b: Array[float,complex]
        (..., N1,...,ND)
    D: int

    Returns
    -------
    rel_dist: Array[float]
        (...,)  \norm{a-b}{\inf} / \norm{b}{\inf}
    """
    assert D > 0
    xp = aac.array_namespace(a)

    axis = tuple(range(-D, 0))
    num = xp.max(xp.abs(a - b), axis=axis)
    den = xp.max(xp.abs(b), axis=axis)
    rel_dist = num / den
    return rel_dist


def rel_l2_close(a: ftkt.Array, b: ftkt.Array, D: int, eps: float) -> bool:
    r"""
    Are `a` and `b` close up to a prescribed relative L2 error?

    Parameters
    ----------
    a: Array[float,complex]
        (..., N1,...,ND)
    b: Array[float,complex]
        (..., N1,...,ND)
    D: int
    eps: float
        [0, 1] tolerance.

    Returns
    -------
    close: bool
        \norm{a - b}{2} <= eps * \norm{b}{2}
    """
    assert D > 0
    assert 0 <= eps <= 1
    xp = aac.array_namespace(a)

    rel_dist = rel_l2_distance(a, b, D)
    close = xp.all(rel_dist <= eps)
    return bool(close)


def inner_product(a: ftkt.Array, b: ftkt.Array, D: int) -> ftkt.Array:
    r"""
    Compute stack-wize inner-product.

    Parameters
    ----------
    a: Array[float,complex]
        (..., N1,...,ND)
    b: Array[float,complex]
        (..., N1,...,ND)
    D: int

    Returns
    -------
    c: Array[float,complex]
        (...,) inner-products <a,b> = \sum_{k} a_{k} \conj{b_{k}}
    """
    xp = aac.array_namespace(a)

    a_ind = b_ind = (Ellipsis, *range(D))
    c_ind = (Ellipsis,)

    c = oe.contract(
        *(a, a_ind),
        *(xp.conj(b), b_ind),
        c_ind,
    )
    return c
