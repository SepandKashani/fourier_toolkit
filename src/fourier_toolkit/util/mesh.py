import math
from typing import Optional

import array_api_compat as aac
import numpy as np

import fourier_toolkit.typing as ftkt

from .dtype import TranslateDType
from .misc import broadcast_seq


class UniformSpec:
    r"""
    Multi-dimensional uniform mesh specifier.

    Defines points :math:`\bbx_{m} \in \bR^{D}` where each point lies on the regular lattice

    .. math::

       \bbx_{\bbm} = \bbx_{0} + \Delta_{\bbx} \odot \bbm,
       \qquad
       [\bbm]_{d} \in \discreteRange{0}{M_{d}-1}

    """

    __slots__ = ("_start", "_step", "_num")

    def __init__(
        self,
        *,
        start=None,
        step=None,
        num=None,
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
            self._start = broadcast_seq(start, D)
            self._step = broadcast_seq(step, D)
            self._num = broadcast_seq(num, D)
        elif all(map(provided, (center, span, num))):
            center = broadcast_seq(center, None, float)
            span = broadcast_seq(span, None, float)
            assert all(s > 0 for s in span)
            num = broadcast_seq(num, None, int)
            assert all(n >= 2 for n in num)

            D = max(map(len, (center, span, num)))
            center = broadcast_seq(center, D)
            span = broadcast_seq(span, D)
            self._num = broadcast_seq(num, D)
            self._start = tuple(c - 0.5 * s for (c, s) in zip(center, span))
            self._step = tuple(s / (n - 1) for (s, n) in zip(span, self._num))
        else:
            raise ValueError

    @property
    def start(self) -> ftkt.TupleR:
        return self._start

    @property
    def step(self) -> ftkt.TupleR:
        return self._step

    @property
    def num(self) -> ftkt.TupleI:
        return self._num

    @property
    def ndim(self) -> int:
        return len(self.start)

    @property
    def span(self) -> ftkt.TupleR:
        _span = [None] * self.ndim
        for d in range(self.ndim):
            dx = self.step[d]
            nx = self.num[d]

            _span[d] = abs(dx) * (nx - 1)
        return tuple(_span)

    @property
    def center(self) -> ftkt.TupleR:
        _center = [None] * self.ndim
        for d in range(self.ndim):
            x0 = self.start[d]
            dx = self.step[d]
            nx = self.num[d]

            _center[d] = x0 + 0.5 * dx * (nx - 1)
        return tuple(_center)

    def __eq__(self, other):
        if not isinstance(other, UniformSpec):
            return NotImplemented
        return (self.start, self.step, self.num) == (other.start, other.step, other.num)

    def __hash__(self):
        return hash((self.start, self.step, self.num))

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"start={self.start}, step={self.step}, num={self.num})"
        )

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

    def __getitem__(self, m: ftkt.TupleI) -> ftkt.TupleR:
        r"""
        Extract a mesh coordinate from its index.

        Parameters
        ----------
        m: tuple[int]

        Returns
        -------
        x_m: tuple[float]
            Mesh coordinate :math:`\bbx_{\bbm}`.

        Notes
        -----
        Similar to Python sequences, negative indices can be used to index from the end of an axis.
        """
        m = broadcast_seq(m, None, int)
        assert len(m) == self.ndim

        x_m = [None] * self.ndim
        for d in range(self.ndim):
            x0 = self.start[d]
            dx = self.step[d]
            nx = self.num[d]
            assert -nx <= m[d] < nx

            _idx = m[d] % nx
            x_m[d] = x0 + _idx * dx
        return tuple(x_m)
