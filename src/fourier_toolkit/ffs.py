import os

import ducc0.fft as dfft
import numpy as np

import fourier_toolkit.linalg as ftk_linalg
import fourier_toolkit.util as ftk_util

__all__ = [
    "FFS",
]


class FFS:
    r"""
    Multi-dimensional Fast Fourier Series Transform.

    Computes FS coefficients of bandlimited T-periodic signals :math:`\tilde{g}: [-T/2, T/2] -> \bC` from its samples.

    This is a modified version of FFS where both the input/output samples to/from FFS are linearly ordered.
    """

    def __init__(
        self,
        T: tuple[float],
        K: tuple[int],
        L: tuple[int],
        *,
        nthreads: int = 0,
    ):
        r"""
        Parameters
        ----------
        T: tuple[float]
            (T1,...,TD) period of :math:`\tilde{g}`.
        K: tuple[int]
            (K1,...,KD) single-sided bandwidth of :math:`\tilde{g}`.
        L: tuple[int]
            (L1,...,LD) FFT length.
            This parameter fixes the dimensionality `D` of the transform.
        nthreads: int
            Number of threads to use. If 0, use all cores.
        """
        L = ftk_util.broadcast_seq(L, None, np.int64)
        D = len(L)
        K = ftk_util.broadcast_seq(K, D, np.int64)
        T = ftk_util.broadcast_seq(T, D, np.double)
        assert np.all(T > 0)
        assert np.all(K >= 0)
        assert np.all(L >= 2 * K + 1)

        if nthreads == 0:
            nthreads = os.cpu_count()
        assert 1 <= nthreads <= os.cpu_count()

        self._D = D
        self._T = T
        self._K = K
        self._L = L
        self._nthreads = int(nthreads)

    def apply(self, G: np.ndarray) -> np.ndarray:
        r"""
        Parameters
        ----------
        G: ndarray[float/complex]
            (..., L1,...,LD) samples of :math:`\tilde{g} \in \bC^{L_{1} \times\cdots\times L_{D}}` at
            locations specified by :py:meth:`~FFS.sample_points`.

        Returns
        -------
        G_FS: ndarray[complex]
            (..., L1,...,LD) FS coefficients :math:`\{\tilde{g}_{\bbk}^{FS}\}_{k=-\bbK}^{\bbK} \in
            \bC` in increasing order. Trailing entries are 0.
        """
        axes = tuple(range(-self._D, 0))

        B1, B2, E1, E2 = self._mod_params(G.dtype)
        mod1 = [B1[d].conj() ** E1[d] for d in range(self._D)]
        mod2 = [B2[d].conj() ** E2[d] for d in range(self._D)]

        _G = np.fft.ifftshift(G, axes)
        _G = ftk_linalg.hadamard_outer(_G, *mod2)
        _G = dfft.c2c(_G, axes=axes, forward=True, inorm=2, nthreads=self._nthreads)
        G_FS = ftk_linalg.hadamard_outer(_G, *mod1)
        return G_FS

    def adjoint(self, G_FS: np.ndarray) -> np.ndarray:
        r"""
        Parameters
        ----------
        G_FS: ndarray[float/complex]
            (..., L1,...,LD) FS coefficients :math:`\{\tilde{g}_{\bbk}^{FS}\}_{k=-\bbK}^{\bbK} \in
            \bC` in increasing order. Trailing entries are 0.

        Returns
        -------
        G: ndarray[complex]
            (..., L1,...,LD) samples of :math:`\tilde{g} \in \bC^{L_{1} \times\cdots\times L_{D}}` at
            locations specified by :py:meth:`~FFS.sample_points`.
        """
        axes = tuple(range(-self._D, 0))

        B1, B2, E1, E2 = self._mod_params(G_FS.dtype)
        mod1 = [B1[d] ** E1[d] for d in range(self._D)]
        mod2 = [B2[d] ** E2[d] for d in range(self._D)]

        _G_FS = ftk_linalg.hadamard_outer(G_FS, *mod1)
        _G_FS = dfft.c2c(_G_FS, axes=axes, forward=False, inorm=2, nthreads=self._nthreads)
        _G_FS = ftk_linalg.hadamard_outer(_G_FS, *mod2)
        G = np.fft.fftshift(_G_FS, axes)
        return G

    def sample_points(self, dtype: np.dtype) -> tuple[np.ndarray]:
        """
        Sampling positions for FFS forward transform. (spatial samples -> FS coefficients.)

        Returns
        -------
        S1,...,SD : tuple[ndarray]
            (L1,),...,(LD,) mesh-points at which to sample a signal in the d-th dimension (in the right order).
        """
        translate = ftk_util.TranslateDType(dtype)
        fdtype = translate.to_float()

        S = [None] * self._D
        for d, (_L, _T) in enumerate(zip(self._L, self._T)):
            if _L % 2 == 1:  # odd case
                _M = (_L - 1) // 2
                idx = np.array([*range(0, _M + 1), *range(-_M, 0)])
                _S = (_T / _L) * idx
            else:  # even case
                _M = _L // 2
                idx = np.array([*range(0, _M), *range(-_M, 0)])
                _S = (_T / _L) * (0.5 + idx)
            S[d] = np.fft.fftshift(_S).astype(fdtype)
        return tuple(S)

    def freq_points(self, dtype: np.dtype) -> tuple[np.ndarray]:
        r"""
        Fourier Transform positions computed by FFS forward transform.

        .. math::

           \tilde{g}^{\fs}_{k}
           =
           \frac{1}{T} g^{\ctft}(k / T)

        Returns
        -------
        V1,...,VD : tuple[ndarray]
            (2K1+1,),...,(2KD+1,) mesh-points at which :math:`g^{\ctft}` is computed in the d-th dimension (in the right order).
        """
        translate = ftk_util.TranslateDType(dtype)
        fdtype = translate.to_float()

        V = [None] * self._D
        for d, (_K, _T) in enumerate(zip(self._K, self._T)):
            _V = np.arange(-_K, _K + 1) / _T
            V[d] = _V.astype(fdtype)
        return tuple(V)

    @staticmethod
    def next_fast_len(K: tuple[int]) -> np.ndarray:
        r"""
        Provide good values of `L` for a fast transform.

        Parameters
        ----------
        K: tuple[int]
            (K1,...,KD) single-sided bandwidth of :math:`\tilde{g}`.

        Returns
        -------
        L: ndarray[int]
            (L1,...,LD) FFT lengths.
        """
        K = ftk_util.broadcast_seq(K, None, int)
        D = len(K)

        L_opt = np.zeros(D, dtype=int)
        for d in range(D):
            L_opt[d] = dfft.good_size(2 * K[d] + 1)
        return L_opt

    # Helper routines (internal) ----------------------------------------------
    def _mod_params(self, dtype: np.dtype):
        """
        Parameters
        ----------
        dtype: float/complex

        Returns
        -------
        B1, B2: ndarray
            (D,) base terms.
        E1, E2: tuple[ndarray]
            (L1,),...,(LD,) exponent vectors.
        """
        translate = ftk_util.TranslateDType(dtype)
        fdtype = translate.to_float()
        cdtype = translate.to_complex()

        B1 = np.zeros(self._D, dtype=cdtype)
        B2 = np.zeros(self._D, dtype=cdtype)
        E1 = [None] * self._D
        E2 = [None] * self._D

        for d in range(self._D):
            _K, _L = [_[d] for _ in (self._K, self._L)]
            _Q = _L - (2 * _K + 1)
            if _L % 2 == 1:  # odd case
                _M = (_L - 1) // 2
                B1[d] = 1
                E2[d] = np.array([*range(0, _M + 1), *range(-_M, 0)], dtype=fdtype)
            else:  # even case
                _M = _L // 2
                B1[d] = np.exp(1j * np.pi / _L)
                E2[d] = np.array([*range(0, _M), *range(-_M, 0)], dtype=fdtype)
            E1[d] = np.array([*range(-_K, _K + 1), *((0,) * _Q)], dtype=fdtype)
            B2[d] = np.exp(-2j * np.pi * _K / _L)

        E1, E2 = map(tuple, [E1, E2])
        return B1, B2, E1, E2
