import importlib.resources as ir
import warnings
from dataclasses import dataclass
from typing import Optional, NamedTuple
from collections.abc import Iterable, Callable, Iterator
from collections import namedtuple
from numpy.typing import NDArray, DTypeLike
import numpy as np

__all__ = [
    "as_namedtuple",
    "broadcast_seq",
    "cast_warn",
    "Interval",
    "next_fast_len",
    "TranslateDType",
    "UniformSpec",
]


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


@dataclass
class UniformSpec:
    r"""
    Multi-dimensional uniform mesh specifier.

    Defines points :math:`\bbx_{m} \in \bR^{D}` where each point lies on the regular lattice

    .. math::

       \bbx_{\bbm} = \bbx_{0} + \Delta_{\bbx} \odot \bbm,
       \qquad
       [\bbm]_{d} \in \discreteRange{0}{M_{d}-1}

    """

    start: tuple[float]
    step: tuple[float]
    num: tuple[int]

    def __init__(
        self,
        start=None,
        step=None,
        num=None,
        *,
        center=None,
        span=None,
    ):
        r"""
        Initialize mesh spec from (`start`, `step`, `num`) or (`center`, `span`, `num`).

        Parameters (option 1)
        ---------------------
        start: tuple[float]
            :math:`\bbx_{0} \in \bR^{D}`
        step: tuple[float]
            :math:`\Delta_{\bbx} \in \bR_{+}^{D}`
        num: tuple[int]
            (M1,...,MD) lattice size

        Parameters (option 2)
        ---------------------
        center: tuple[float]
            (D,) geometric center
        span: tuple[float]
            (D,) axial widths
        num: tuple[int]
            (M1,...,MD) lattice size

        Scalars are broadcast to all dimensions.
        """

        provided = lambda _: _ is not None  # noqa: E731
        if all(map(provided, (start, step, num))):
            start = broadcast_seq(start, None, float)
            step = broadcast_seq(step, None, float)
            assert all(s > 0 for s in step)
            num = broadcast_seq(num, None, int)
            assert all(n >= 1 for n in num)

            D = max(map(len, (start, step, num)))
            self.start = broadcast_seq(start, D)
            self.step = broadcast_seq(step, D)
            self.num = broadcast_seq(num, D)
        elif all(map(provided, (center, span, num))):
            center = broadcast_seq(center, None, float)
            span = broadcast_seq(span, None, float)
            assert all(s > 0 for s in span)
            num = broadcast_seq(num, None, int)
            assert all(n >= 2 for n in num)

            D = max(map(len, (center, span, num)))
            center = broadcast_seq(center, D)
            span = broadcast_seq(span, D)
            self.num = broadcast_seq(num, D)
            self.start = tuple(c - 0.5 * s for (c, s) in zip(center, span))
            self.step = tuple(s / (n - 1) for (s, n) in zip(span, self.num))
        else:
            raise ValueError

    @property
    def ndim(self) -> int:
        return len(self.start)

    def __iter__(self) -> Iterator:
        for d in range(self.ndim):
            yield (self.start[d], self.step[d], self.num[d])

    @property
    def span(self) -> tuple[float]:
        return tuple(dx * (nx - 1) for (_, dx, nx) in self)

    @property
    def center(self) -> tuple[float]:
        return tuple(x0 + 0.5 * dx * (nx - 1) for (x0, dx, nx) in self)

    def __neg__(self) -> "UniformSpec":
        """
        Negate all mesh coordinates.

        This is useful if you want to implicitly flip the sign of the exponent in a Fourier transform.
        """
        neg_x0 = tuple(-(x0 + dx * (nx - 1)) for (x0, dx, nx) in self)
        return UniformSpec(start=neg_x0, step=self.step, num=self.num)

    def meshgrid(self, xp, sparse: bool = False) -> tuple[NDArray]:
        """
        Equivalent of :py:func:`numpy.meshgrid`.

        Returns
        -------
        mesh: tuple[NDArray]
            (D,) axial knot coordinates
        """
        mesh_1D = tuple(x0 + dx * xp.arange(nx) for (x0, dx, nx) in self)
        mesh = xp.meshgrid(*mesh_1D, indexing="ij", sparse=sparse)
        return mesh

    def knots(self, xp) -> NDArray:
        """
        Returns
        -------
        x: NDArray
            (M1,...,MD, D) mesh coordinates
        """
        x = xp.stack(self.meshgrid(xp), axis=-1)
        return x


@dataclass
class Interval:
    r"""
    Multi-dimensional interval specifier.

    Defines a box given its center :math:`\bbx_{c} \in \bR^{D}` and width :math:`\Delta_{\bbx} \in \bR_{+}^{D}`.
    """

    center: tuple[float]
    span: tuple[float]

    def __init__(self, center, span):
        r"""
        Parameters
        ----------
        center: tuple[float]
            \bbx_{c} \in \bR^{D}
        span: tuple[float]
            \Delta_{\bbx} \in \bR_{+}^{D}
        """
        # parameter validation
        uspec = UniformSpec(center=center, span=span, num=2)

        self.center = uspec.center
        self.span = uspec.span

    @property
    def ndim(self) -> int:
        return len(self.center)

    def bounds(self) -> tuple:
        """
        Axial interval (start, stop) values.

        Returns
        -------
        interval: tuple[tuple[float]]
            ((L1,R1),...,(LD,RD)) interval limits.
        """
        return tuple(
            (c - 0.5 * s, c + 0.5 * s)
            for (c, s) in zip(
                self.center,
                self.span,
            )
        )
