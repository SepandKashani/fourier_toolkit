import importlib.resources as ir
import math
import warnings
from collections import namedtuple
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import NamedTuple, Optional

import array_api_compat as aac
import numpy as np
from numpy.typing import DTypeLike

import fourier_toolkit.typing as ftkt

__all__ = [
    "as_namedtuple",
    "broadcast_seq",
    "cast_warn",
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


class TranslateDType:
    """
    (int,float,complex) dtype translator.
    """

    map_to_float = {
        "int32": "float32",
        "int64": "float64",
        "float32": "float32",
        "float64": "float64",
        "complex64": "float32",
        "complex128": "float64",
    }
    map_from_float = {
        ("float32", "i"): "int32",
        ("float64", "i"): "int64",
        ("float32", "f"): "float32",
        ("float64", "f"): "float64",
        ("float32", "c"): "complex64",
        ("float64", "c"): "complex128",
    }

    def __init__(self, x: ftkt.ArrayRC):
        # Find float-equivalent name of `x.dtype`
        xp = aac.array_namespace(x)
        info = xp.__array_namespace_info__()
        dtypes = info.dtypes()

        found = False
        for name, dtype in dtypes.items():
            if x.dtype == dtype:
                fdtype = self.map_to_float[name]
                found = True
        assert found, f"Un-recognized input dtype '{x.dtype}'."

        # Store (float-equivalent name, backend name->dtype mapping)
        self._fdtype = fdtype
        self._name2dtype = dtypes

    def to_int(self) -> DTypeLike:
        name = self.map_from_float[(self._fdtype, "i")]
        return self._name2dtype[name]

    def to_float(self) -> DTypeLike:
        name = self.map_from_float[(self._fdtype, "f")]
        return self._name2dtype[name]

    def to_complex(self) -> DTypeLike:
        name = self.map_from_float[(self._fdtype, "c")]
        return self._name2dtype[name]


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
            :math:`\Delta_{\bbx} \in \bR^{D}` (signed)
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
            assert not any(math.isclose(s, 0) for s in step)
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

    @property
    def span(self) -> tuple[float]:
        _span = [None] * self.ndim
        for d in range(self.ndim):
            dx = self.step[d]
            nx = self.num[d]

            _span[d] = abs(dx) * (nx - 1)
        return tuple(_span)

    @property
    def center(self) -> tuple[float]:
        _center = [None] * self.ndim
        for d in range(self.ndim):
            x0 = self.start[d]
            dx = self.step[d]
            nx = self.num[d]

            _center[d] = x0 + 0.5 * dx * (nx - 1)
        return tuple(_center)

    def meshgrid(
        self,
        sparse: bool = False,
        like: Optional[ftkt.ArrayRC] = None,
    ) -> tuple[ftkt.ArrayR, ...]:
        """
        Equivalent of :py:func:`numpy.meshgrid`.

        Parameters
        ----------
        like: ArrayRC
            Reference object to allow the creation of arrays which are not NumPy arrays.
            Will inherit the (backend,device,fdtype) of `like`.

        Returns
        -------
        mesh: tuple[ArrayR]
            (D,) axial knot coordinates
        """
        if like is None:
            like = np.asarray([], dtype=np.float64)

        xp = aac.array_namespace(like)
        fdtype = TranslateDType(like).to_float()
        device = like.device

        mesh_1D = [None] * self.ndim
        for d in range(self.ndim):
            x0 = self.start[d]
            dx = self.step[d]
            nx = self.num[d]

            shape = [1] * self.ndim
            shape[d] = nx

            mesh_1D[d] = xp.reshape(
                x0 + dx * xp.arange(nx, dtype=fdtype, device=device),
                shape=shape,
            )

        if sparse:
            mesh = mesh_1D
        else:
            mesh = [None] * self.ndim
            for d in range(self.ndim):
                mesh[d] = xp.broadcast_to(mesh_1D[d], shape=self.num)

        return mesh

    def knots(self, like: Optional[ftkt.ArrayRC] = None) -> ftkt.ArrayR:
        """
        Parameters
        ----------
        like: ArrayRC
            Reference object to allow the creation of arrays which are not NumPy arrays.
            Will inherit the (backend,device,fdtype) of `like`.

        Returns
        -------
        x: ArrayR
            (M1,...,MD, D) mesh coordinates
        """
        mesh = self.meshgrid(like=like)
        xp = aac.array_namespace(mesh[0])
        x = xp.stack(mesh, axis=-1)
        return x

    def __getitem__(self, idx: tuple[int]) -> tuple[float]:
        """
        Extract a mesh coordinate from its index.

        Parameters
        ----------
        idx: tuple[int]

        Returns
        -------
        m: tuple[float]
            Mesh coordinate at `idx`.

        Notes
        -----
        Similar to Python sequences, negative indices can be used to index from the end of an axis.
        """
        idx = broadcast_seq(idx, None, int)
        assert len(idx) == self.ndim

        m = [None] * self.ndim
        for d in range(self.ndim):
            x0 = self.start[d]
            dx = self.step[d]
            nx = self.num[d]
            assert -nx <= idx[d] < nx

            _idx = idx[d] % nx
            m[d] = x0 + _idx * dx
        return tuple(m)
