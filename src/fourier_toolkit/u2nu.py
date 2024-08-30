import warnings

import numba as nb
import numpy as np

import fourier_toolkit.nu2u as ftk_nu2u

__all__ = [
    "Uniform2NonUniform",
    "U2NU",
]

# Disable all warnings for the entire module
warnings.filterwarnings("ignore", category=nb.NumbaExperimentalFeatureWarning)


class Uniform2NonUniform(ftk_nu2u.NU2U):
    r"""
    Multi-dimensional Uniform-to-NonUniform Fourier Transform.

    Given the Dirac stream

    .. math::

       f(\bbx) = \sum_{m} w_{m} \delta(\bbx - \bbx_{m}),

    computes samples of :math:`f^{F}`, i.e.,

    .. math::

       f^{F}(\bbv_{n}) = \bbz_{n} = \sum_{m} w_{m} \ee^{ -\cj 2\pi \innerProduct{\bbx_{m}}{\bbv_{n}} },

    where :math:`\bbv_{n} \in \bR^{D}`, and :math:`\bbx_{m}` lies on the regular lattice

    .. math::

       \begin{align}
           \bbx_{\bbm} &= \bbx_{0} + \Delta_{\bbx} \odot \bbm, & [\bbm]_{d} \in \{0,\ldots,M_{d}-1\},
       \end{align}

    with :math:`M = \prod_{d} M_{d}`.
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
        upsampfac_ratio: tuple[float] = 0.5,
        kernel_param_type: str = "bounded",
        # Runtime behavior ------------
        kernel_type: str = "kb_ppoly",
        fft_nthreads: int = 0,
        spread_nthreads: int = 0,
        max_cluster_size: int = 10_000,
        max_window_ratio: tuple[float] = 10,
        # Chunking behavior -----------
        chunked: bool = False,
        max_bbox_ratio: tuple[float] = 10,
        max_bbox_anisotropy: float = 5,
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
            (N, D) support points :math:`\bbv_{n} \in \bR^{D}`.
        isign: +1, -1
            Sign of the exponent.
        eps: float
            [kernel_param_type=bounded] Kernel stopband relative energy :math:`\epsilon \in ]0, 1[`.
            [kernel_param_type=finufft] Target relative error :math:`\epsilon \in ]0, 1[`.
        upsampfac: tuple[float]
            Total upsampling factor :math:`\sigma = \sigma_{x} \sigma_{v} > 1`.
        upsampfac_ratio: tuple[float]
            Constant :math:`\epsilon_{\sigma} \in ]0, 1[` controlling upsampling ratio between (x, v) domains:

            .. math::

               \begin{align}
                   \sigma_{x} & = \sigma^{\epsilon_{\sigma}} \\
                   \sigma_{v} & = \sigma^{1 - \epsilon_{\sigma}}
               \end{align}
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
        chunked: bool
            Perform U->NU transform by partitioning the v-domain.
        max_bbox_ratio: tuple[float]
            Maximum sub-partition size

            .. math::

               c(\epsilon)
               \le
               [ X_{d}^{0} V_{d}^{0} ]_{k}
               \le
               \rho_{k} c(\epsilon),

            where :math:`\rho_{k} > 1` grows the (x,v) volume to some size larger than :math:`c(\epsilon) = \sqrt{w_{x} w_{v}}`.

            `max_bbox_ratio` defines the quantities :math:`\{ \rho_{1},\ldots,\rho_{D} \}`.
            Setting too small reduces FFT memory, but nudges complexity towards :math:`\cO(M N)`.
            Setting too large increases FFT memory, but may reduce FFT speed-up if point density is low.

            The FFT length per sub-partition (in number of cells) is

            .. math::

               \prod_{k=1}^{D}
               [\sigma]_{k}
               \bigBrack{
                   [X_{d}^{0} V_{d}^{0}]_{k} +
                   [w_{x} + w_{v}]_{k} +
                   \frac{ [w_{x} w_{v}]_{k} }
                        { [X_{d}^{0} V_{d}^{0}]_{k} }
               }

        max_bbox_anisotropy: float
            Max tolerated (normalized) anisotropy ratio >= 1.

            Setting close to 1 favors cubeoid-shaped partitions of v-space.
            Setting large allows v-partitions to be highly asymmetric.
        """
        super().__init__(
            x=v,
            v_spec=x_spec,
            isign=-isign,
            # Accuracy-related ------------
            eps=eps,
            upsampfac=upsampfac,
            upsampfac_ratio=upsampfac_ratio,
            kernel_param_type=kernel_param_type,
            # Runtime behavior ------------
            kernel_type=kernel_type,
            fft_nthreads=fft_nthreads,
            spread_nthreads=spread_nthreads,
            max_cluster_size=max_cluster_size,
            max_window_ratio=max_window_ratio,
            # Chunking behavior -----------
            chunked=chunked,
            max_bbox_ratio=max_bbox_ratio,
            max_bbox_anisotropy=max_bbox_anisotropy,
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

    def diagnostic_plot(
        self,
        domain: str,
        axes: tuple[int] = None,
        ax=None,
    ):
        """
        Plot (2D projection of) decomposed domain.

        This method is not thread safe! (See implementation notes as to why.)

        Parameters
        ----------
        domain: "x", "v"
            Domain to plot.
        axes: tuple[int]
            (2,) projection axes.

            Can be unspecified for (1D, 2D) transforms.
        ax: Axes
            Axes to draw on. A new ax/figure is allocated if unspecified.

        Returns
        -------
        ax: Axes
        """
        domain = domain.lower().strip()
        assert domain in ("x", "v")
        domain = {
            "x": "v",
            "v": "x",
        }[domain]

        msg = "diagnostic_plot(): x/v domain labels are reversed (due to implementation reasons)."
        warnings.warn(msg)

        ax = super().diagnostic_plot(
            domain=domain,
            axes=axes,
            ax=ax,
        )
        return ax

    # Helper routines (internal) ----------------------------------------------


U2NU = Uniform2NonUniform  # alias
