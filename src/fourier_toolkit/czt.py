import os

import ducc0.fft as dfft
import numpy as np

import fourier_toolkit.linalg as ftk_linalg
import fourier_toolkit.util as ftk_util

__all__ = [
    "CZT",
]


class CZT:
    r"""
    Multi-dimensional Chirp Z-Transform (CZT) :math:`C: \bC^{N_{1} \times\cdots\times N_{D}} \to
    \bC^{M_{1} \times\cdots\times M_{D}}`.

    The 1D CZT of parameters :math:`(A, W, M)` is defined as:

    .. math::

       (C \, \bbx)[k]
       =
       \sum_{n=0}^{N-1} \bbx[n] A^{-n} W^{nk},

    where :math:`\bbx \in \bC^{N}`, :math:`A, W \in \bC`, and :math:`k = \{0, \ldots, M-1\}`.

    A D-dimensional CZT corresponds to taking a 1D CZT along each transform axis.

    For stability reasons, this implementation assumes :math:`A, W \in \bC` lie on the unit circle.
    """

    def __init__(
        self,
        N: tuple[int],
        M: tuple[int],
        A: tuple[complex],
        W: tuple[complex],
        *,
        nthreads: int = 0,
    ):
        r"""
        Parameters
        ----------
        N: tuple[int]
            (N1,...,ND) dimensions of the input :math:`\bbx`.
        M: tuple[int]
            (M1,...,MD) dimensions of the output :math:`(C \, \bbx)`.
        A: tuple[complex]
            (D,) circular offsets.
        W: tuple[complex]
            (D,) circular spacings between transform points.
        nthreads: int
            Number of threads to use. If 0, use all cores.
        """
        M = ftk_util.broadcast_seq(M, None, np.int64)
        D = len(M)
        N = ftk_util.broadcast_seq(N, D, np.int64)
        A = ftk_util.broadcast_seq(A, D, np.complex128)
        W = ftk_util.broadcast_seq(W, D, np.complex128)
        assert np.all(M > 0)
        assert np.all(N > 0)
        assert np.allclose(np.abs(A), 1)
        assert np.allclose(np.abs(W), 1)

        if nthreads == 0:
            nthreads = os.cpu_count()
        assert 1 <= nthreads <= os.cpu_count()

        self._D = D
        self._M = M
        self._N = N
        self._L = np.array(list(map(dfft.good_size, N + M - 1)), dtype=int)
        self._A = A
        self._W = W
        self._nthreads = int(nthreads)

    def apply(self, x: np.ndarray) -> np.ndarray:
        r"""
        Parameters
        ----------
        x: ndarray[float/complex]
            (..., N1,...,ND) inputs :math:`\bbx \in \bC^{N_{1} \times\cdots\times N_{D}}`.

        Returns
        -------
        y: ndarray[complex]
            (..., M1,...,MD) outputs :math:`\bby = (C \, \bbx) \in \bC^{M_{1} \times\cdots\times M_{D}}`.
        """
        axes = tuple(range(-self._D, 0))

        AWk2, FWk2, Wk2, extract = self._mod_params_apply(x.dtype)
        pad_width = [(0, 0)] * (x.ndim - self._D)
        pad_width += [(0, p) for p in self._L - self._N]

        _x = ftk_linalg.hadamard_outer(x, *AWk2)
        _x = np.pad(_x, pad_width)
        _x = dfft.c2c(_x, axes=axes, forward=True, inorm=0, nthreads=self._nthreads)
        _x = ftk_linalg.hadamard_outer(_x, *FWk2)
        _x = dfft.c2c(_x, axes=axes, forward=False, inorm=2, nthreads=self._nthreads)
        y = ftk_linalg.hadamard_outer(_x[..., *extract], *Wk2)
        return y

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        r"""
        Parameters
        ----------
        y: ndarray[float/complex]
            (..., M1,...,MD) outputs :math:`\bby = (C \, \bbx) \in \bC^{M_{1} \times\cdots\times M_{D}}`.

        Returns
        -------
        x: ndarray[complex]
            (..., N1,...,ND) inputs :math:`\bbx \in \bC^{N_{1} \times\cdots\times N_{D}}`.
        """
        # CZT^{*}(y,M,A,W)[n] = CZT(y,N,A=1,W=W*)[n] * A^{n}
        czt = CZT(
            N=self._M,
            M=self._N,
            A=1,
            W=self._W.conj(),
            nthreads=self._nthreads,
        )
        An = self._mod_params_adjoint(y.dtype)

        _x = czt.apply(y)
        x = ftk_linalg.hadamard_outer(_x, *An)
        return x

    # Helper routines (internal) ----------------------------------------------
    def _mod_params_apply(self, dtype: np.dtype):
        """
        Parameters
        ----------
        dtype: float/complex

        Returns
        -------
        AWk2: ndarray
            (N1,),...,(ND,) pre-FFT modulation vectors.
        FWk2: ndarray
            (L1,),...,(LD,) FFT of convolution filters.
        Wk2: ndarray
            (M1,),...,(MD,) post-FFT modulation vectors.
        extract: list[slice]
            (slice1,...,sliceD) FFT interval to extract.
        """
        translate = ftk_util.TranslateDType(dtype)
        cdtype = translate.to_complex()

        # Build modulation vectors (Wk2, AWk2, FWk2).
        Wk2 = [None] * self._D
        AWk2 = [None] * self._D
        FWk2 = [None] * self._D
        for d in range(self._D):
            A = self._A[d]
            W = self._W[d]
            N = self._N[d]
            M = self._M[d]
            L = self._L[d]

            k = np.arange(max(M, N))
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
        extract = [slice(None)] * self._D
        for d in range(self._D):
            N = self._N[d]
            M = self._M[d]
            L = self._L[d]
            extract[d] = slice(N - 1, N + M - 1)

        return AWk2, FWk2, Wk2, extract

    def _mod_params_adjoint(self, dtype: np.dtype):
        """
        Parameters
        ----------
        dtype: float/complex

        Returns
        -------
        An: ndarray
            (N1,),...,(ND,) vectors.
        """
        translate = ftk_util.TranslateDType(dtype)
        cdtype = translate.to_complex()

        An = [None] * self._D
        for d in range(self._D):
            _A = self._A[d]
            _N = self._N[d]
            _An = _A ** np.arange(_N)

            An[d] = _An.astype(cdtype)

        return An
