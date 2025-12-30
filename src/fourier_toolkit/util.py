import warnings

from typing import Optional
from collections.abc import Iterable, Callable
from numpy.typing import NDArray, DTypeLike


__all__ = [
    "broadcast_seq",
    "cast_warn",
]


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


def cast_warn(x: NDArray, dtype: DTypeLike) -> NDArray:
    """
    Cast `x` to `dtype` if type mis-match.

    Emit warning when cast occurs.
    """
    y = x.astype(dtype, copy=False)
    if x.dtype != y.dtype:
        msg = f"{x.shape}: {x.dtype} -> {y.dtype} cast performed."
        warnings.warn(msg)
    return y
