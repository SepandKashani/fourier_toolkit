import warnings

import numba as nb
import numpy as np

import fourier_toolkit.nufft1 as ftk_nufft1

__all__ = [
    "NUFFT2",
]

# Disable all warnings for the entire module
warnings.filterwarnings("ignore", category=nb.NumbaExperimentalFeatureWarning)


class NUFFT2(ftk_nufft1.NUFFT1):
    r"""
    Multi-dimensional Uniform-to-NonUniform Fourier Transform.

    Given the Dirac stream

    .. math::

       f(\bbx) = \sum_{m} w_{m} \delta(\bbx - \bbx_{m}),

    computes samples of :math:`f^{F}`, i.e.,

    .. math::

       f^{F}(\bbv_{n}) = \bbz_{n} = \sum_{m} w_{m} \ee^{ -\cj 2\pi \innerProduct{\bbx_{m}}{\bbv_{n}} },

    where :math:`\bbv_{n}` lie in a :math:`T \in \bR^{D}`-sized interval and :math:`\bbx_{m}` lies on the regular lattice

    .. math::

       \begin{align}
           \bbx_{\bbm} &= \bbx_{0} + \Delta_{\bbx} \odot \bbm, & [\bbm]_{d} \in \{0,\ldots,M_{d}-1\},
       \end{align}

    with :math:`M = \prod_{d} M_{d}` and :math:`\Delta_{\bbx} = 1 / \bbT`.
    """

    def __init__(
        self,
        x_spec: dict[str, np.ndarray],
        v: np.ndarray,
        *,
        isign: int = -1,
        # Accuracy-related ------------
        eps: float = 1e-6,
        upsampfac: tuple[float] = 2,
        kernel_param_type: str = "bounded",
        # Runtime behavior ------------
        kernel_type: str = "kb_ppoly",
        fft_nthreads: int = 0,
        spread_nthreads: int = 0,
        max_cluster_size: int = 10_000,
        max_window_ratio: tuple[float] = 10,
    ):
        r"""
        Parameters
        ----------
        x_spec: dict[str, ndarray]
            :math:`\bbx_{m}` lattice specifier, with keys:

            * `start`: (D,) values :math:`\bbx_{0} \in \bR^{D}`.
            * `step` : (D,) values :math:`\Delta_{\bbx} \in \bR^{D}`.
            * `num`  : (D,) values :math:`\{ M_{1},\ldots,M_{D} \} \in \bN^{D}`.
              This parameter fixes the dimensionality `D` of the transform.

            Scalars are broadcasted to all dimensions.
        v: ndarray
            (N, D) support points :math:`\bbv_{n} \in \bbV_{c} + [-\bbT / 2, \bbT / 2]`, where :math:`\bbV_{c} \in \bR^{D}` is an arbitrary constant.
        isign: +1, -1
            Sign of the exponent.
        eps: float
            [kernel_param_type=bounded] Kernel stopband relative energy :math:`\epsilon \in ]0, 1[`.
            [kernel_param_type=finufft] Target relative error :math:`\epsilon \in ]0, 1[`.
        upsampfac: tuple[float]
            Total upsampling factor :math:`\sigma > 1`.
        kernel_param_type: str
            How to choose kernel parameters.

            Must be one of:

            * "bounded": ensures eps-bandwidths of (\psi_{x}, \psi_{v}) are located in the safe zone.
            * "finufft": uses relations derived in FINUFFT paper.
        kernel_type: str
            Which kernel to use for spreading/interpolation.

            Must be one of:

            * "kb": exact Kaiser-Bessel pulse.
            * "kb_ppoly": piece-wise polynomial approximation of Kaiser-Bessel pulse.
        fft_nthreads: int
            Number of threads used to perform FFTs. If 0, use all cores.
        spread_nthreads: int
            Number of threads used to perform spreading/interpolation. If 0, use all cores.
        max_cluster_size: int
            As described in ``UniformSpread.__init__``.
        max_window_ratio: tuple[float]
            As described in ``UniformSpread.__init__``.
        """
        super().__init__(
            x=v,
            v_spec=x_spec,
            isign=-isign,
            # Accuracy-related ------------
            eps=eps,
            upsampfac=upsampfac,
            kernel_param_type=kernel_param_type,
            # Runtime behavior ------------
            kernel_type=kernel_type,
            fft_nthreads=fft_nthreads,
            spread_nthreads=spread_nthreads,
            max_cluster_size=max_cluster_size,
            max_window_ratio=max_window_ratio,
        )

    def apply(self, w: np.ndarray) -> np.ndarray:
        r"""
        Parameters
        ----------
        w: ndarray[float/complex]
            (..., M1,...,MD) weights :math:`w_{m} \in \bC^{D}`.

        Returns
        -------
        z: ndarray[complex]
            (..., N) weights :math:`z_{n} \in \bC^{D}`.
        """
        z = super().adjoint(w)
        return z

    def adjoint(self, z: np.ndarray) -> np.ndarray:
        r"""
        Parameters
        ----------
        z: ndarray[float/complex]
            (..., N) weights :math:`z_{n} \in \bC^{D}`.

        Returns
        -------
        w: ndarray[complex]
            (..., M1,...,MD) weights :math:`w_{m} \in \bC^{D}`.
        """
        w = super().apply(z)
        return w

    # Helper routines (internal) ----------------------------------------------
