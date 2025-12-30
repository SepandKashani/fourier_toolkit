import warnings

from typing import Optional
from collections.abc import Iterable, Callable
from numpy.typing import NDArray, DTypeLike
import numpy as np

__all__ = [
    "broadcast_seq",
    "cast_warn",
    "TranslateDType",
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


class TranslateDType:
    """
    (int,float,complex) dtype translator.
    """

    map_to_float = {
        np.dtype(np.int32): np.dtype(np.float32),
        np.dtype(np.int64): np.dtype(np.float64),
        np.dtype(np.float32): np.dtype(np.float32),
        np.dtype(np.float64): np.dtype(np.float64),
        np.dtype(np.complex64): np.dtype(np.float32),
        np.dtype(np.complex128): np.dtype(np.float64),
    }
    map_from_float = {
        (np.dtype(np.float32), "i"): np.dtype(np.int32),
        (np.dtype(np.float64), "i"): np.dtype(np.int64),
        (np.dtype(np.float32), "f"): np.dtype(np.float32),
        (np.dtype(np.float64), "f"): np.dtype(np.float64),
        (np.dtype(np.float32), "c"): np.dtype(np.complex64),
        (np.dtype(np.float64), "c"): np.dtype(np.complex128),
    }

    def __init__(self, dtype: DTypeLike):
        dtype = np.dtype(dtype)
        assert dtype in self.map_to_float
        self._fdtype = self.map_to_float[dtype]

    def to_int(self) -> np.dtype:
        return self.map_from_float[(self._fdtype, "i")]

    def to_float(self) -> np.dtype:
        return self.map_from_float[(self._fdtype, "f")]

    def to_complex(self) -> np.dtype:
        return self.map_from_float[(self._fdtype, "c")]
