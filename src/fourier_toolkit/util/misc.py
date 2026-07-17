import importlib.resources as ir
import warnings
from collections import namedtuple
from collections.abc import Callable, Iterable
from typing import NamedTuple, Optional

import array_api_compat as aac
import numpy as np

import fourier_toolkit.typing as ftkt


def as_namedtuple(**kwargs) -> NamedTuple:
    """
    Store mapping as named-tuple.

    The goal is to provide more convenient access to dictionary entries via `.key` syntax.
    """
    nt_t = namedtuple("nt_t", kwargs.keys())
    y = nt_t(**kwargs)
    return y


def broadcast_seq(
    x,
    N: Optional[int] = None,
    cast: Callable = lambda _: _,
) -> tuple:
    """
    Broadcast `x` to a tuple of length `N`.

    If `N` is omitted, then no broadcasting takes place, only tupling.
    """
    if isinstance(x, Iterable):
        y = tuple(x)
    else:
        y = (x,)

    if N is not None:
        if len(y) == 1:
            y *= N  # broadcast
        assert len(y) == N

    y = tuple(map(cast, y))
    return y


def cast_warn(x: ftkt.ArrayRC, dtype: ftkt.DType) -> ftkt.ArrayRC:
    """
    Cast `x` to `dtype` if type mis-match.

    Emit warning when cast occurs.
    """
    xp = aac.array_namespace(x)
    y = xp.astype(x, dtype, copy=False)
    if y.dtype != x.dtype:
        msg = f"{x.shape}: {x.dtype} -> {y.dtype} cast performed."
        warnings.warn(msg)
    return y


def next_fast_len(n: int) -> int:
    """
    Find a good FFT size.

    For broadest compatibility with FFT libraries, the search is restricted to 5-smooth numbers, i.e. numbers of the form :math:`2^{a} 3^{b} 5^{c}`.

    Parameters
    ----------
    n: int
        Lower bound on FFT length.

    Returns
    -------
    n_next: int
        A 5-smooth FFT length such that `n_next >= n`.
    """
    assert 1 <= n <= (2**10) * (3**10) * (5**10)  # known pre-computed range

    path = ir.files("fourier_toolkit") / "resources" / "5_smooth.txt"
    with ir.as_file(path) as f:
        candidates = np.loadtxt(f, dtype=int, comments="#")
    idx = np.searchsorted(candidates, n, side="right")

    if candidates[idx - 1] == n:
        n_next = n
    else:
        n_next = int(candidates[idx])
    return n_next
