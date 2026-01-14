import cmath
import math
from typing import Literal

import numpy as np

import fourier_toolkit.linalg as ftkl
import fourier_toolkit.typing as ftkt
import fourier_toolkit.util as ftku

__all__ = [
    "CZT",
    "DFT",
    "u2u",
]


ExponentSign = Literal[-1, +1]


def u2u(
    x_spec: ftku.UniformSpec,
    v_spec: ftku.UniformSpec,
    w: ftkt.ArrayRC,
    isign: ExponentSign,
) -> ftkt.ArrayC:
    r"""
    Multi-dimensional Uniform-to-Uniform Fourier Transform. (:math:`\tuu`)

    Computes the Fourier sum

    .. math::

       \bbz_{n} = \sum_{m} w_{m} \ee^{ \pm \cj 2\pi \innerProduct{\bbx_{m}}{\bbv_{n}} },

    where :math:`(\bbx_{m}, \bbv_{n})` lie on the regular lattice

    .. math::

       \bbx_{\bbm} &= \bbx_{0} + \Delta_{\bbx} \odot \bbm, \qquad [\bbm]_{d} \in \discreteRange{0}{M_{d}-1}, \\
       \bbv_{\bbn} &= \bbv_{0} + \Delta_{\bbv} \odot \bbn, \qquad [\bbn]_{d} \in \discreteRange{0}{N_{d}-1},

    with :math:`M = \prod_{d} M_{d}` and :math:`N = \prod_{d} N_{d}`.

    Parameters
    ----------
    x_spec: UniformSpec
        :math:`\bbx_{m}` lattice.
    v_spec: UniformSpec
        :math:`\bbv_{n}` lattice.
    isign: +1, -1
        Exponent sign.
    w: ArrayRC
        (..., M1,...,MD) weights :math:`w_{m} \in \bC`.

    Returns
    -------
    z: ArrayC
        (..., N1,...,ND) weights :math:`z_{n} \in \bC`.

    Notes
    -----
    :math:`\tuu` transforms for arbitrary (x_spec,v_spec) can be implemented using the CZT algorithm (using 2 FFTs), but a single FFT can be used in some cases.
    This implementation chooses the (FFT, CZT) per axis to maximize efficiency.
    """
    assert (isign := int(isign)) in (-1, +1)

    op = _U2U(x_spec=x_spec, v_spec=v_spec)
    if isign == -1:
        z = op.apply(w)
    else:
        z = op.apply(w.conj()).conj()
    return z


# Helper routines (internal) ---------------------------------------------------


class DFT:
    r"""
    Multi-dimensional Discrete Fourier Transform (DFT) :math:`F: \bC^{N_{1} \times\cdots\times N_{D}} \to \bC^{N_{1} \times\cdots\times N_{D}}`.

    The 1D DFT is defined as:

    .. math::

       \bby[k]
       =
       (F \, \bbx)[k]
       =
       \sum_{n=0}^{N-1} \bbx[n] \ee^{-\cj \frac{2\pi}{N} nk},

    where :math:`\bbx \in \bC^{N}`, and :math:`k \in \discreteRange{0}{N-1}`.

    A D-dimensional DFT corresponds to taking a 1D DFT along each transform axis.

    This implementation is a thin shell around the FFT algorithm intended to simplify multi-backend (CPU, GPU) use.
    """

    def __init__(self, D: int):
        """
        Parameters
        ----------
        D: int
            Dimension of the transform.
        """
        assert D >= 1
        self.cfg = ftku.as_namedtuple(
            D=D,
        )

    def apply(self, x: ftkt.ArrayRC) -> ftkt.ArrayC:
        r"""
        Compute :math:`\bby = F \bbx`.

        Parameters
        ----------
        x: ArrayRC
            (..., N1,...,ND) input :math:`\bbx \in \bC^{N_{1} \times\cdots\times N_{D}}`.

        Returns
        -------
        y: ArrayC
            (..., N1,...,ND) output :math:`\bby \in \bC^{N_{1} \times\cdots\times N_{D}}`.
        """
        xp = x.__array_namespace__()
        y = xp.fft.fftn(x, axes=tuple(range(-self.cfg.D, 0)), norm="backward")
        return y

    def adjoint(self, y: ftkt.ArrayRC) -> ftkt.ArrayC:
        r"""
        Compute :math:`\bbx = F^{\adj} \bby`.

        Parameters
        ----------
        y: ArrayRC
            (..., N1,...,ND) input :math:`\bby \in \bC^{N_{1} \times\cdots\times N_{D}}`.

        Returns
        -------
        x: ArrayC
            (..., N1,...,ND) output :math:`\bbx \in \bC^{N_{1} \times\cdots\times N_{D}}`.
        """
        xp = y.__array_namespace__()
        x = xp.fft.ifftn(y, axes=tuple(range(-self.cfg.D, 0)), norm="forward")
        return x

    def inverse(self, y: ftkt.ArrayRC) -> ftkt.ArrayC:
        r"""
        Inverse transform :math:`\bbx = F^{-1} \bby`.

        Parameters
        ----------
        y: ArrayRC
            (..., N1,...,ND) input :math:`\bby = \in \bC^{N_{1} \times\cdots\times N_{D}}`.

        Returns
        -------
        x: ArrayC
            (..., N1,...,ND) output :math:`\bbx \in \bC^{N_{1} \times\cdots\times N_{D}}`.
        """
        xp = y.__array_namespace__()
        x = xp.fft.ifftn(y, axes=tuple(range(-self.cfg.D, 0)), norm="backward")
        return x


class CZT:
    r"""
    Multi-dimensional Chirp Z-Transform (CZT) :math:`C: \bC^{N_{1} \times\cdots\times N_{D}} \to
    \bC^{M_{1} \times\cdots\times M_{D}}`.

    The 1D CZT of parameters :math:`(A, W, M)` is defined as:

    .. math::

       \bby[k]
       =
       (C \, \bbx)[k]
       =
       \sum_{n=0}^{N-1} \bbx[n] A^{-n} W^{nk},

    where :math:`\bbx \in \bC^{N}`, :math:`(A, W) \in \bC`, and :math:`k \in \discreteRange{0}{M-1}`.

    A D-dimensional CZT corresponds to taking a 1D CZT along each transform axis.

    For stability reasons, this implementation assumes :math:`\abs{A} = \abs{W} = 1`.
    """

    def __init__(
        self,
        N: tuple[int],
        M: tuple[int],
        A: tuple[complex],
        W: tuple[complex],
    ):
        r"""
        Parameters
        ----------
        N: tuple[int]
            (N1,...,ND) dimensions of the input :math:`\bbx`.
        M: tuple[int]
            (M1,...,MD) dimensions of the output :math:`\bby = (C \, \bbx)`.
        A: tuple[complex]
            (D,) circular offsets.
        W: tuple[complex]
            (D,) circular spacings between transform points.
        """
        M = ftku.broadcast_seq(M, None, int)
        D = len(M)
        N = ftku.broadcast_seq(N, D, int)
        A = ftku.broadcast_seq(A, D, complex)
        W = ftku.broadcast_seq(W, D, complex)
        assert all(m > 0 for m in M)
        assert all(n > 0 for n in N)
        assert all(math.isclose(abs(a), 1) for a in A)
        assert all(math.isclose(abs(w), 1) for w in W)

        self.cfg = ftku.as_namedtuple(
            M=M,
            D=D,
            N=N,
            A=A,
            W=W,
            L=tuple(ftku.next_fast_len(n + m - 1) for (n, m) in zip(N, M)),
        )

    def apply(self, x: ftkt.ArrayRC) -> ftkt.ArrayC:
        r"""
        Compute :math:`\bby = C \bbx`.

        Parameters
        ----------
        x: ArrayRC
            (..., N1,...,ND) input :math:`\bbx \in \bC^{N_{1} \times\cdots\times N_{D}}`.

        Returns
        -------
        y: ArrayC
            (..., M1,...,MD) output :math:`\bby \in \bC^{M_{1} \times\cdots\times M_{D}}`.
        """
        AWk2, FWk2, Wk2, extract = self._mod_params_apply(x)
        pad_width = [(0, 0)] * (x.ndim - self.cfg.D)  # stack dimensions
        pad_width += [  # core dimensions
            (0, l - n)
            for (l, n) in zip(self.cfg.L, self.cfg.N)  # noqa: E741
        ]

        _x = ftkl.hadamard_outer(x, *AWk2)
        _x = np.pad(_x, pad_width)
        _x = DFT(self.cfg.D).apply(_x)
        _x = ftkl.hadamard_outer(_x, *FWk2)
        _x = DFT(self.cfg.D).inverse(_x)
        y = ftkl.hadamard_outer(_x[..., *extract], *Wk2)
        return y

    def adjoint(self, y: ftkt.ArrayRC) -> ftkt.ArrayC:
        r"""
        Compute :math:`\bbx = C^{\adj} \bby`.

        Parameters
        ----------
        y: ArrayRC
            (..., M1,...,MD) input :math:`\bby \in \bC^{M_{1} \times\cdots\times M_{D}}`.

        Returns
        -------
        x: ArrayC
            (..., N1,...,ND) output :math:`\bbx \in \bC^{N_{1} \times\cdots\times N_{D}}`.
        """
        # CZT^{\adjoint}(y,M,A,W)[n] = CZT(y,N,A=1,W=W*)[n] * A^{n}
        czt = CZT(
            N=self.cfg.M,
            M=self.cfg.N,
            A=1,
            W=tuple(w.conjugate() for w in self.cfg.W),
        )
        An = self._mod_params_adjoint(y)

        _y = czt.apply(y)
        x = ftkl.hadamard_outer(_y, *An)
        return x

    # Helper routines (internal) ----------------------------------------------
    def _mod_params_apply(self, x: ftkt.ArrayRC):
        """
        Parameters
        ----------
        x: ArrayRC

        Returns
        -------
        AWk2: ArrayC
            (N1,),...,(ND,) pre-FFT modulation vectors.
        FWk2: ArrayC
            (L1,),...,(LD,) FFT of convolution filters.
        Wk2: ArrayC
            (M1,),...,(MD,) post-FFT modulation vectors.
        extract: list[slice]
            (slice1,...,sliceD) FFT interval to extract.
        """
        translate = ftku.TranslateDType(x.dtype)
        cdtype = translate.to_complex()

        # Build modulation vectors (Wk2, AWk2, FWk2).
        Wk2 = [None] * self.cfg.D
        AWk2 = [None] * self.cfg.D
        FWk2 = [None] * self.cfg.D
        for d in range(self.cfg.D):
            A = self.cfg.A[d]
            W = self.cfg.W[d]
            N = self.cfg.N[d]
            M = self.cfg.M[d]
            L = self.cfg.L[d]

            k = np.arange(max(M, N), dtype=int, like=x)
            _Wk2 = W ** ((k**2) / 2)
            _AWk2 = (A ** -k[:N]) * _Wk2[:N]
            _FWk2 = np.fft.fft(
                np.concatenate([_Wk2[(N - 1) : 0 : -1], _Wk2[:M]]).conj(),
                n=L,
            )
            _Wk2 = _Wk2[:M]

            Wk2[d] = _Wk2.astype(cdtype)
            AWk2[d] = _AWk2.astype(cdtype)
            FWk2[d] = _FWk2.astype(cdtype)

        # Build (extract,)
        extract = [slice(None)] * self.cfg.D
        for d in range(self.cfg.D):
            N = self.cfg.N[d]
            M = self.cfg.M[d]
            L = self.cfg.L[d]
            extract[d] = slice(N - 1, N + M - 1)

        return AWk2, FWk2, Wk2, extract

    def _mod_params_adjoint(self, y: ftkt.ArrayRC):
        """
        Parameters
        ----------
        y: ArrayRC

        Returns
        -------
        An: ArrayC
            (N1,),...,(ND,) vectors.
        """
        translate = ftku.TranslateDType(y.dtype)
        cdtype = translate.to_complex()

        An = [None] * self.cfg.D
        for d in range(self.cfg.D):
            _A = self.cfg.A[d]
            _N = self.cfg.N[d]
            _An = _A ** np.arange(_N, dtype=int, like=y)

            An[d] = _An.astype(cdtype)

        return An


class _U2U:
    r"""
    Object-oriented interface to a :math:`\tuu` transform, with exponent sign set to :math:`-1`.

    For internal use only.
    """

    def __init__(
        self,
        x_spec: ftku.UniformSpec,
        v_spec: ftku.UniformSpec,
    ):
        r"""
        Parameters
        ----------
        x_spec: UniformSpec
            :math:`\bbx_{m}` lattice.
        v_spec: UniformSpec
            :math:`\bbv_{n}` lattice.
        """
        assert x_spec.ndim == v_spec.ndim
        D = x_spec.ndim

        # decide on algorithm per axis
        fft_axes = []
        czt_axes = []
        for d in range(D):
            dx = x_spec.step[d]
            dv = v_spec.step[d]
            M = x_spec.num[d]
            N = v_spec.num[d]

            if (M == N) and math.isclose(dx * dv * N, 1):
                fft_axes.append(d)
            else:
                czt_axes.append(d)

        self.cfg = ftku.as_namedtuple(
            D=D,
            x_spec=x_spec,
            v_spec=v_spec,
            fft_axes=tuple(fft_axes),
            czt_axes=tuple(czt_axes),
        )

    def apply(self, w: ftkt.ArrayRC) -> ftkt.ArrayC:
        r"""
        Compute :math:`\bbz = U \bbw`.

        Parameters
        ----------
        w: ArrayRC
            (..., M1,...,MD) weights :math:`w_{m} \in \bC`.

        Returns
        -------
        z: ArrayC
            (..., N1,...,ND) weights :math:`z_{n} \in \bC`.
        """
        _w = w

        # Processing FFT axes
        if self.cfg.fft_axes:
            (ax_fft, Cp, fft, Bp, ax_ifft) = self._fft_params(w)
            _w = _w.transpose(ax_fft)
            _w = ftkl.hadamard_outer(_w, *Cp)
            _w = fft.apply(_w)
            _w = ftkl.hadamard_outer(_w, *Bp)
            _w = _w.transpose(ax_ifft)

        # Processing CZT axes
        if self.cfg.czt_axes:
            (ax_czt, czt, B, ax_iczt) = self._czt_params(w)
            _w = _w.transpose(ax_czt)
            _w = czt.apply(_w)
            _w = ftkl.hadamard_outer(_w, *B)
            _w = _w.transpose(ax_iczt)

        z = _w
        return z

    # Helper routines (internal) ----------------------------------------------
    def _fft_params(self, y: ftkt.ArrayRC):
        """
        Parameters
        ----------
        y: ArrayRC

        Returns
        -------
        ax_fft: tuple[int]
            Permutation tuple to move FFT axes to end of `y`.
        Cp: ArrayC
            (N1,),...,(ND,) pre-FFT modulation vectors.
        fft: FFT
            FFT() instance.
        Bp: ArrayC
            (N1,),...,(ND,) post-FFT modulation vectors.
        ax_ifft: tuple[int]
            Permutation tuple to undo initial axis transposition.
        """
        translate = ftku.TranslateDType(y.dtype)
        cdtype = translate.to_complex()

        # Build (ax_fft, ax_ifft)
        sh = y.shape[: -self.cfg.D]  # stack dimensions
        stk_axes = tuple(range(len(sh)))
        fft_axes = tuple(ax + len(sh) for ax in self.cfg.fft_axes)
        czt_axes = tuple(ax + len(sh) for ax in self.cfg.czt_axes)
        ax_fft = (*stk_axes, *czt_axes, *fft_axes)
        ax_ifft = tuple(np.argsort(ax_fft))

        # Build FFT operator
        D_fft = len(self.cfg.fft_axes)
        fft = DFT(D_fft)

        # Build modulation vectors (Cp, Bp)
        Cp = [None] * D_fft
        Bp = [None] * D_fft
        for d in range(D_fft):
            ax = self.cfg.fft_axes
            x0 = self.cfg.x_spec.start
            dx = self.cfg.x_spec.step
            nx = self.cfg.x_spec.num
            v0 = self.cfg.v_spec.start
            dv = self.cfg.v_spec.step
            nv = self.cfg.v_spec.num

            phase_scale_c = -2 * math.pi * dx[ax[d]] * v0[ax[d]]
            m = np.arange(nx[ax[d]], dtype=int, like=y)
            _Cp = np.exp(1j * phase_scale_c * m)
            Cp[d] = _Cp.astype(cdtype)

            phase_scale_b = -2 * math.pi * x0[ax[d]]
            v = v0[ax[d]] + dv[ax[d]] * np.arange(nv[ax[d]], dtype=int, like=y)
            _Bp = np.exp(1j * phase_scale_b * v)
            Bp[d] = _Bp.astype(cdtype)

        return (ax_fft, Cp, fft, Bp, ax_ifft)

    def _czt_params(self, y: ftkt.ArrayRC):
        """
        Parameters
        ----------
        y: ArrayRC

        Returns
        -------
        ax_czt: tuple[int]
            Permutation tuple to move CZT axes to end of `y`.
        czt: CZT
            CZT(A,W,M,N) instance.
        B: ArrayC
            (N1,),...,(ND,) post-CZT modulation vectors.
        ax_iczt: tuple[int]
            Permutation tuple to undo initial axis transposition.
        """
        translate = ftku.TranslateDType(y.dtype)
        cdtype = translate.to_complex()

        # Build (ax_czt, ax_iczt)
        sh = y.shape[: -self.cfg.D]  # stack dimensions
        stk_axes = tuple(range(len(sh)))
        fft_axes = tuple(ax + len(sh) for ax in self.cfg.fft_axes)
        czt_axes = tuple(ax + len(sh) for ax in self.cfg.czt_axes)
        ax_czt = (*stk_axes, *fft_axes, *czt_axes)
        ax_iczt = tuple(np.argsort(ax_czt))

        # Build CZT operator
        D_czt = len(self.cfg.czt_axes)
        N = [None] * D_czt
        M = [None] * D_czt
        A = [None] * D_czt
        W = [None] * D_czt
        for d in range(D_czt):
            ax = self.cfg.czt_axes
            dx = self.cfg.x_spec.step
            nx = self.cfg.x_spec.num
            v0 = self.cfg.v_spec.start
            dv = self.cfg.v_spec.step
            nv = self.cfg.v_spec.num

            N[d] = nx[ax[d]]
            M[d] = nv[ax[d]]
            A[d] = cmath.exp(+1j * 2 * math.pi * dx[ax[d]] * v0[ax[d]])
            W[d] = cmath.exp(-1j * 2 * math.pi * dx[ax[d]] * dv[ax[d]])
        czt = CZT(N, M, A, W)

        # Build modulation vector (B,)
        B = [None] * D_czt
        for d in range(D_czt):
            ax = self.cfg.czt_axes
            x0 = self.cfg.x_spec.start
            v0 = self.cfg.v_spec.start
            dv = self.cfg.v_spec.step
            nv = self.cfg.v_spec.num

            phase_scale = -2 * math.pi * x0[ax[d]]
            v = v0[ax[d]] + dv[ax[d]] * np.arange(nv[ax[d]], dtype=int, like=y)
            _B = np.exp(1j * phase_scale * v)

            B[d] = _B.astype(cdtype)

        return ax_czt, czt, B, ax_iczt
