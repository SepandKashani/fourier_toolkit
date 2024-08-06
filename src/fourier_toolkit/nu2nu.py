import concurrent.futures as cf
import importlib
import os
import warnings

import numba as nb
import numpy as np
import scipy.optimize as sopt

import fourier_toolkit.cluster as ftk_cluster
import fourier_toolkit.complex as ftk_complex
import fourier_toolkit.ffs as ftk_ffs
import fourier_toolkit.kernel as ftk_kernel
import fourier_toolkit.linalg as ftk_linalg
import fourier_toolkit.numba as ftk_numba
import fourier_toolkit.spread as ftk_spread
import fourier_toolkit.util as ftk_util

__all__ = [
    "NonUniform2NonUniform",
    "NU2NU",
]

# Disable all warnings for the entire module
warnings.filterwarnings("ignore", category=nb.NumbaExperimentalFeatureWarning)


class NonUniform2NonUniform:
    r"""
    Multi-dimensional NonUniform-to-NonUniform Fourier Transform.

    Given the Dirac stream

    .. math::

       f(\bbx) = \sum_{m} w_{m} \delta(\bbx - \bbx_{m}),

    computes samples of :math:`f^{F}`, i.e.,

    .. math::

       f^{F}(\bbv_{n}) = \bbz_{n} = \sum_{m} w_{m} \ee^{ -\cj 2\pi \innerProduct{\bbx_{m}}{\bbv_{n}} },

    where :math:`(\bbx_{m}, \bbv_{n}) \in \bR^{D}`.
    """

    def __init__(
        self,
        x: np.ndarray,
        v: np.ndarray,
        *,
        isign: int = -1,
        # Accuracy-related ------------
        eps: float = 1e-6,
        upsampfac: tuple[float] = 2,
        upsampfac_ratio: tuple[float] = 0.5,
        # Runtime behavior ------------
        kernel_type: str = "kb_ppoly",
        fft_nthreads: int = 0,
        spread_nthreads: int = 0,
        max_cluster_size: int = 10_000,
        max_window_ratio: tuple[float] = 10,
        # Chunking behavior -----------
        chunked: bool = False,
        domain: str = "xv",
        max_bbox_ratio: tuple[float] = 10,
        max_bbox_anisotropy: float = 5,
    ):
        r"""
        Parameters
        ----------
        x: ndarray
            (M, D) support points :math:`\bbx_{m} \in \bR^{D}`.
        v: ndarray
            (N, D) frequencies :math:`\bbv_{n} \in \bR^{D}`.
        isign: +1, -1
            Sign of the exponent.
        eps: float
            Kernel stopband relative energy :math:`\epsilon \in ]0, 1[`.
        upsampfac: tuple[float]
            Total upsampling factor :math:`\sigma = \sigma_{x} \sigma_{v} > 1`.
        upsampfac_ratio: tuple[float]
            Constant :math:`\epsilon_{\sigma} \in ]0, 1[` controlling upsampling ratio between (x, v) domains:

            .. math::

               \begin{align}
                   \sigma_{x} & = \sigma^{\epsilon_{\sigma}} \\
                   \sigma_{v} & = \sigma^{1 - \epsilon_{\sigma}}
               \end{align}
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
            Perform NU->NU transform by partitioning (x, v) domains.
        domain: str
            Which domains to partition. (`chunked=True` only.)

            Must be one of:

            * "x":  partition x-domain;
            * "v":  partition v-domain;
            * "xv": partition (x,v)-domains.
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

            Setting close to 1 favors cubeoid-shaped partitions of (x,v) space.
            Setting large allows (x,v)-partitions to be highly asymmetric.
        """
        if x.ndim == 1:
            x = x[:, np.newaxis]
        if v.ndim == 1:
            v = v[:, np.newaxis]
        _, Dx = x.shape
        _, Dv = v.shape
        assert (D := Dx) == Dv

        # validate non-(x,v) inputs
        p = self._validate_inputs(
            D=D,
            isign=isign,
            # Accuracy-related ------------
            eps=eps,
            upsampfac=upsampfac,
            upsampfac_ratio=upsampfac_ratio,
            # Runtime behavior ------------
            kernel_type=kernel_type,
            fft_nthreads=fft_nthreads,
            spread_nthreads=spread_nthreads,
            max_cluster_size=max_cluster_size,
            max_window_ratio=max_window_ratio,
            # Chunking behavior -----------
            chunked=chunked,
            domain=domain,
            max_bbox_ratio=max_bbox_ratio,
            max_bbox_anisotropy=max_bbox_anisotropy,
        )

        # partition data (part 1) =============================================
        w_x, w_v = self._infer_kernel_widths(
            p.eps,
            p.upsampfac,
            p.upsampfac_ratio,
        )
        x_bbox_dim, v_bbox_dim = self._infer_bbox_dims(
            *(
                np.ptp(x, axis=0) + np.finfo(np.double).eps,  # guard against single-point transforms
                np.ptp(v, axis=0) + np.finfo(np.double).eps,
            ),
            *(w_x, w_v),
            p.domain,
            *(p.max_bbox_ratio, p.max_bbox_anisotropy),
        )
        x_blk_info = self._nu_partition_info("x", x, x_bbox_dim)
        v_blk_info = self._nu_partition_info("v", v, v_bbox_dim)
        # =====================================================================

        cfg = self._init_metadata(
            x=x,
            **x_blk_info._asdict(),
            # -------------------------
            v=v,
            **v_blk_info._asdict(),
            # -------------------------
            isign=p.isign,
            eps=p.eps,
            upsampfac=p.upsampfac,
            upsampfac_ratio=p.upsampfac_ratio,
            kernel_type=p.kernel_type,
            fft_nthreads=p.fft_nthreads,
            spread_nthreads=p.spread_nthreads,
        )

        # partition data (part 2) =============================================
        x_info = self._nu_cluster_info(
            *("x", x, cfg.x_idx, cfg.x_blk_bound, cfg.Xc),
            *(cfg.F_x0, cfg.F_dx, cfg.L),
            cfg.alpha_x,
            *(p.max_cluster_size, p.max_window_ratio),
        )
        v_info = self._nu_cluster_info(
            *("v", v, cfg.v_idx, cfg.v_blk_bound, cfg.Vc),
            *(cfg.F_v0, cfg.F_dv, cfg.L),
            cfg.alpha_v,
            *(p.max_cluster_size, p.max_window_ratio),
        )
        # =====================================================================

        # adjust contents of `cfg`:
        #   - (x, v) -> partition/cluster order. [Was user-order before.]
        #   - (x_idx, v_idx) -> partition AND cluster order. [Was partition-order before.]
        #   - adds extra fields from _nu_cluster_info().
        cfg = cfg._asdict()
        cfg.update(
            x=x[x_info.x_idx],  # canonical partition/cluster order
            Qx=len(x_info.x_cl_bound) - 1,
            **x_info._asdict(),
            # -------------------------
            v=v[v_info.v_idx],  # canonical partition/cluster order
            Qv=len(v_info.v_cl_bound) - 1,
            **v_info._asdict(),
        )
        self.cfg = ftk_util.as_namedtuple(**cfg)

    def apply(self, w: np.ndarray) -> np.ndarray:
        r"""
        Parameters
        ----------
        w: ndarray[float/complex]
            (..., M) weights :math:`w_{m} \in \bC^{D}`.

        Returns
        -------
        z: ndarray[complex]
            (..., N) weights :math:`z_{n} \in \bC^{D}`.
        """
        w = self._re_order_x(w, forward=True)  # (..., M)

        w = self._flip_sign(w)  # (..., M)
        g0 = self._fw_spread(w)  # (Px, Pv, ..., L1,...,LD)
        h = self._de_window(g0)  # (Px, Pv, ..., L1,...,LD)

        hFS = self._ffs_transform(h, forward=True)  # (Px, Pv, ..., L1,...,LD)
        hFS = self._transpose_blks(hFS)  # (Pv, Px, ..., L1,...,LD)

        gF = self._fw_interpolate(hFS)  # (..., N)
        z = self._de_convolve(gF)  # (..., N)
        z = self._flip_sign(z)  # (..., N)

        z = self._re_order_v(z, forward=True)  # (..., N)
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
            (..., M) weights :math:`w_{m} \in \bC^{D}`.
        """
        z = self._re_order_v(z, forward=False)  # (..., N)

        z = self._flip_sign(z)  # (..., N)
        gF = self._de_convolve(z)  # (..., N)
        hFS = self._bw_spread(gF)  # (Pv, Px, ..., L1,...,LD)

        h = self._ffs_transform(hFS, forward=False)  # (Pv, Px, ..., L1,...,LD)
        h = self._transpose_blks(h)  # (Px, Pv, ..., L1,...,LD)

        g0 = self._de_window(h)  # (Px, Pv, ..., L1,...,LD)
        w = self._bw_interpolate(g0)  # (..., M)
        w = self._flip_sign(w)

        w = self._re_order_x(w, forward=False)  # (..., M)
        return w

    def diagnostic_plot(
        self,
        domain: str,
        axes: tuple[int] = None,
        ax=None,
    ):
        """
        Plot (2D projection of) decomposed domain.

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
        try:
            plt = importlib.import_module("matplotlib.pyplot")
            collections = importlib.import_module("matplotlib.collections")
            patches = importlib.import_module("matplotlib.patches")
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Install Matplotlib to use diagnostic_plot()")

        if ax is None:
            _, ax = plt.subplots()

        domain = domain.lower().strip()
        assert domain in ("x", "v")
        if self.cfg.D == 1:
            raise NotImplementedError  # needs different plotting mechanism
        elif self.cfg.D == 2:
            if axes is None:
                axes = [0, 1]
            axes = ftk_util.broadcast_seq(axes, 2, np.int64)
        elif self.cfg.D > 2:
            assert axes is not None, "Parameter[axes] must be provided for 3D+ transforms."
            axes = ftk_util.broadcast_seq(axes, 2, np.int64)
        assert np.all((-self.cfg.D <= axes) & (axes < self.cfg.D))
        assert len(np.unique(axes)) == 2

        pts = getattr(self.cfg, domain)[:, axes]  # (M, 2)
        Pc = getattr(self.cfg, f"{domain.upper()}c")[:, axes]  # (P, 2)
        Pd = getattr(self.cfg, f"{domain.upper()}d")[axes]  # (2,)
        Pd0 = getattr(self.cfg, f"{domain.upper()}d0")[axes]  # (2,)
        P = getattr(self.cfg, f"P{domain}")  # P

        # Draw Maximum BBox ---------------------------------------------------
        pts_min = np.min(pts, axis=0)
        pts_max = np.max(pts, axis=0)
        bbox_dim = pts_max - pts_min  # (2,)
        rect = patches.Rectangle(
            xy=pts_min,
            width=bbox_dim[0],
            height=bbox_dim[1],
            facecolor="none",
            edgecolor="k",
            label=f"{domain} BBox",
        )
        ax.add_patch(rect)

        # Draw Partition BBox -------------------------------------------------
        bbox_d = [None] * P
        bbox_d_kwargs = dict(
            facecolor="r",
            edgecolor="r",
            alpha=0.5,
        )
        for p in range(P):
            bbox_d[p] = patches.Rectangle(
                xy=Pc[p] - Pd / 2,
                width=Pd[0],
                height=Pd[1],
                **bbox_d_kwargs,
                label=rf"${{{domain.upper()}}}_{{d}}$",
            )
        pc_d = collections.PatchCollection(bbox_d, **bbox_d_kwargs)
        ax.add_collection(pc_d)

        bbox_d0 = [None] * P
        bbox_d0_kwargs = dict(
            facecolor="none",
            edgecolor="k",
            linestyle="--",
        )
        for p in range(P):
            bbox_d0[p] = patches.Rectangle(
                xy=Pc[p] - Pd0 / 2,
                width=Pd0[0],
                height=Pd0[1],
                **bbox_d0_kwargs,
                label=rf"${{{domain.upper()}}}_{{d}}^{{0}}$",
            )
        pc_d0 = collections.PatchCollection(bbox_d0, **bbox_d0_kwargs)
        ax.add_collection(pc_d0)

        # Draw Partition Centroids --------------------------------------------
        p_centroids = ax.scatter(
            Pc[:, 0],
            Pc[:, 1],
            c="k",
            marker="x",
            label=rf"${{{domain.upper()}}}_{{c}}$",
        )

        # Misc Details --------------------------------------------------------
        pad_width = 0.1 * bbox_dim  # 10% axial pad
        ax.set_xlabel(rf"${{{domain}}}_{axes[0]}$")
        ax.set_ylabel(rf"${{{domain}}}_{axes[1]}$")
        ax.set_xlim(pts_min[0] - pad_width[0], pts_max[0] + pad_width[0])
        ax.set_ylim(pts_min[1] - pad_width[1], pts_max[1] + pad_width[1])
        ax.legend(handles=[p_centroids, bbox_d[0], bbox_d0[0]])
        ax.set_aspect(1)

        return ax

    # Helper routines (internal) ----------------------------------------------
    @staticmethod
    def _validate_inputs(
        D,
        isign,
        # Accuracy-related ------------
        eps,
        upsampfac,
        upsampfac_ratio,
        # Runtime behavior ------------
        kernel_type,
        fft_nthreads,
        spread_nthreads,
        max_cluster_size,
        max_window_ratio,
        # Chunking behavior -----------
        chunked,
        domain,
        max_bbox_ratio,
        max_bbox_anisotropy,
    ):
        isign = int(isign / abs(isign))
        assert isign in (-1, 1)

        eps = float(eps)
        assert 0 < eps < 1

        upsampfac = ftk_util.broadcast_seq(upsampfac, D, np.double)
        assert np.all(upsampfac > 1)

        upsampfac_ratio = ftk_util.broadcast_seq(upsampfac_ratio, D, np.double)
        assert np.all((0 < upsampfac_ratio) & (upsampfac_ratio < 1))

        kernel_type = kernel_type.strip().lower()
        assert kernel_type in ("kb", "kb_ppoly")

        if fft_nthreads == 0:
            fft_nthreads = os.cpu_count()
        assert 1 <= fft_nthreads <= os.cpu_count()

        if spread_nthreads == 0:
            spread_nthreads = os.cpu_count()
        assert 1 <= spread_nthreads <= os.cpu_count()

        max_cluster_size = int(max_cluster_size)
        assert max_cluster_size > 0

        max_window_ratio = ftk_util.broadcast_seq(max_window_ratio, D, np.double)
        assert np.all(max_window_ratio >= 2)

        chunked = bool(chunked)
        domain = domain.strip().lower()
        assert domain in ("x", "v", "xv")

        max_bbox_ratio = ftk_util.broadcast_seq(
            max_bbox_ratio if chunked else np.finfo(np.double).max,
            D,
            np.double,
        )
        assert np.all(max_bbox_ratio > 1)

        max_bbox_anisotropy = float(max_bbox_anisotropy)
        assert max_bbox_anisotropy >= 1

        params = ftk_util.as_namedtuple(
            D=D,
            isign=isign,
            # Accuracy-related ------------
            eps=eps,
            upsampfac=upsampfac,
            upsampfac_ratio=upsampfac_ratio,
            # Runtime behavior ------------
            kernel_type=kernel_type,
            fft_nthreads=fft_nthreads,
            spread_nthreads=spread_nthreads,
            max_cluster_size=max_cluster_size,
            max_window_ratio=max_window_ratio,
            # Chunking behavior -----------
            chunked=chunked,
            domain=domain,
            max_bbox_ratio=max_bbox_ratio,
            max_bbox_anisotropy=max_bbox_anisotropy,
        )
        return params

    @staticmethod
    def _infer_kernel_widths(
        eps: float,
        upsampfac: np.ndarray,
        upsampfac_ratio: np.ndarray,
    ) -> tuple[np.ndarray]:
        r"""
        Compute support (in #samples) of \psi_{x}, \psi_{v}.

        Parameters
        ----------
        eps: float
            Kernel stopband relative energy :math:`\epsilon \in ]0, 1[`.
        upsampfac: ndarray[float]
            Total upsampling factor :math:`\sigma = \sigma_{x} \sigma_{v} > 1`.
        upsampfac_ratio: ndarray[float]
            Constant :math:`\epsilon_{\sigma} \in ]0, 1[` controlling upsampling ratio between (x, v) domains.

        Returns
        -------
        w_x, w_v: ndarray[int]
            (D,)
        """
        idtype = np.int64

        sigma_x = upsampfac**upsampfac_ratio
        sigma_v = upsampfac ** (1 - upsampfac_ratio)

        beta = ftk_util.broadcast_seq(
            ftk_kernel.KaiserBessel.beta_from_eps(eps),
            len(upsampfac),
        )
        c = 2 * beta / np.pi
        gamma = 1.1  # FFT oversampling ratio \gamma

        lhs_x = c * gamma
        rhs_x = c * sigma_v
        w_x = np.ceil(lhs_x).astype(idtype)

        lhs_v = c * sigma_x / (2 * sigma_x - 1)
        rhs_v = c * sigma_x
        w_v = np.ceil(lhs_v).astype(idtype)

        if np.any((w_x >= rhs_x) | (w_v >= rhs_v)):
            msg = " ".join(
                [
                    "Valid (w_x, w_v) does not exist for chosen (eps, upsampfac, upsampfac_ratio).",
                    "Poor precision may occur.",
                ]
            )
            warnings.warn(msg)
        return w_x, w_v

    @classmethod
    def _infer_bbox_dims(
        cls,
        x_ptp: np.ndarray,
        v_ptp: np.ndarray,
        w_x: np.ndarray,
        w_v: np.ndarray,
        domain: str,
        max_bbox_ratio: np.ndarray,
        max_bbox_anisotropy: float,
    ) -> tuple[np.ndarray]:
        r"""
        Find
            X box dimensions X_{d}^{0} \in \bR^{D}
        and
            V box dimensions V_{d}^{0} \in \bR^{D}
        such that:

        * number of NUFFT sub-problems is minimized;
        * NUFFT sub-problems (indirectly) limited to user-specified memory budget;
        * box dimensions are not too rectangular, i.e. anisotropic.

        Parameters
        ----------
        x_ptp, v_ptp: ndarray[float]
            (D,) spread of the data in each domain.
        w_x, w_v: ndarray[int]
            (D,) \psi_{x}, \psi_{v} support in #samples.
        domain: "x", "v", "xv"
            Domain(s) to partition.
            Some constraints below are dropped if only (x,) or (v,) is to be sharded.
        max_bbox_ratio: ndarray[float]
            (D,) maximum sub-partition ratios \{ \rho_{1},\ldots,\rho_{D} \}
        max_bbox_anisotropy: float
            Max tolerated (normalized) anisotropy ratio \alpha

        Returns
        -------
        x_bbox_dim: ndarray[float]
            (D,) X-box dimensions :math:`X_{d}^{0}`
        v_bbox_dim: ndarray[float]
            (D,) V-box dimensions :math:`V_{d}^{0}`
        """
        # (X_{d}^{0}, V_{d}^{0}) are referred to as (X, V) below for notational simplicity.
        #
        # Given that
        #
        #     FFT length \approx
        #     \prod_{k=1..D}
        #     \sigma_k \bigBrack{
        #         X_k V_k +
        #         wx_k + wv_k +
        #         \frac{ wx_k wv_k }{ X_k V_k }
        #     }
        #
        # we can formulate an optimization problem to find good (X, V) values which limit FFT memory use.
        #
        #
        # Mathematical Formulation
        # ------------------------
        # User input:
        #     1. \rho: max FFT axial size per partition
        #     2. \alpha: max anisotropy
        #
        # minimize objective function
        #     \prod_{k=1..D} X_k^{tot} / X_k                                   # X-domain box-count
        #     *                                                                #      \times
        #     \prod_{k=1..D} V_k^{tot} / V_k                                   # V-domain box-count
        # subject to
        #     1. \sqrt{wx_k wv_k} <= X_k V_k <= \rho_k \sqrt{wx_k wv_k}        # min/max partition Heisenberg volume constraint -> indirectly limits partition FFT size
        #     2. X_k <= X_k^{tot}                                              # X-domain: at least 1 box
        #     3. V_k <= V_k^{tot}                                              # V-domain: at least 1 box
        #     4. 1/alpha <= (X_k / X_k^{tot}) / (X_q / X_q^{tot}) <= alpha     # X-domain: limit box anisotropy
        #     5. 1/alpha <= (V_k / V_k^{tot}) / (V_q / V_q^{tot}) <= alpha     # V-domain: limit box anisotropy
        #     6. 1/alpha <= (X_k / X_k^{tot}) / (V_q / V_q^{tot}) <= alpha     # XV-domain: limit cross-anisotropy
        #
        # The problem can be recast as a small LP and easily solved:
        #     - (4, 5) dropped for 1D problems.
        #     - (5, 6) dropped if sharding (x,) only.
        #     - (4, 6) dropped if sharding (v,) only.
        #
        #
        # Mathematical Formulation (LinProg)
        # ----------------------------------
        # minimize
        #     c^{T} x
        # subject to
        #        A x <= b
        #    lb <= x <= ub
        # where
        #     x = [ln(X_1) ... ln(X_D), ln(V_1) ... ln(V_D)] \in \bR^{2D}
        #     c = [-1 ... -1]
        #     ub = [ln(X_1^{tot}) ... ln(X_D^{tot}), ln(V_1^{tot}) ... ln(V_D^{tot})]
        #     lb = [-inf ... -inf, -inf ... -inf]                   (domain = "xv")
        #          [-inf ... -inf, ln(V_1^{tot}) ... ln(V_D^{tot})] (domain = "x")
        #          [ln(X_1^{tot}) ... ln(X_D^{tot}), -inf ... -inf] (domain = "v")
        #     [A | b] = [ E ,  E | b1 = ln(rho)   + ln(\sqrt{wx wv})               ],  # max partition Heisenberg volume constraint (upper limit, vector form)
        #               [-E , -E | b2 =           - ln(\sqrt{wx wv})               ],  # min partition Heisenberg volume constraint (lower limit, vector form)
        #               [ M1,  Z | b3 = ln(alpha) + ln(X_k^{tot}) - ln(X_q^{tot})  ],  # X-domain box size anisotropy limited (upper limit, vector form)
        #               [-M1,  Z | b4 = ln(alpha) - ln(X_k^{tot}) + ln(X_q^{tot})  ],  # X-domain box size anisotropy limited (lower limit, vector form)
        #               [ Z , M1 | b5 = ln(alpha) + ln(V_k^{tot}) - ln(V_q^{tot})  ],  # V-domain box size anisotropy limited (upper limit, vector form)
        #               [ Z ,-M1 | b6 = ln(alpha) - ln(V_k^{tot}) + ln(V_q^{tot})  ],  # V-domain box size anisotropy limited (lower limit, vector form)
        #               [   M2   | b7 = ln(alpha) + ln(V_k^{tot}) - ln(X_q^{tot})  ],  # cross-domain box size anisotropy limited (upper limit, vector form)
        #               [  -M2   | b8 = ln(alpha) - ln(V_k^{tot}) + ln(X_q^{tot})  ],  # cross-domain box size anisotropy limited (lower limit, vector form)
        #     ]
        #     E = eye(D)
        #     Z = zeros(D_choose_2, D)
        #     M1 = (D_choose_2, D) (M)ask containing [-1, 1] per row
        #     M2 = (D**2, 2D) (M)ask containing [-1, 1] per row

        # Validate inputs ---------------------------------
        assert np.all((x_ptp > 0) & (v_ptp > 0))
        assert np.all((w_x > 0) & (w_v > 0))
        assert domain in ("x", "v", "xv")
        assert np.all(max_bbox_ratio > 1)
        assert max_bbox_anisotropy >= 1

        # Expand (x_ptp, v_ptp) if too small --------------
        D = len(x_ptp)
        x_ptp, v_ptp = cls._grow_to_min_vol(x_ptp, v_ptp, w_x, w_v)

        # Build (c, lb, ub) -------------------------------
        c = -np.ones(2 * D)
        ub = np.log(np.r_[x_ptp, v_ptp])
        lb = np.log(np.r_[x_ptp, v_ptp])
        if "x" in domain:
            lb[:D] = -np.inf
        if "v" in domain:
            lb[-D:] = -np.inf

        # Build (A, b) ------------------------------------
        D_choose_2 = D * (D - 1) // 2
        D_pow_2 = D**2
        E = np.eye(D)
        Z = np.zeros((D_choose_2, D))
        M1 = np.zeros((D_choose_2, D))
        M2 = np.zeros((D_pow_2, 2 * D))
        i, j = np.triu_indices(D, k=1)
        q, k = np.meshgrid(np.arange(D), np.arange(D), indexing="ij")
        q, k = q.ravel(), k.ravel()
        for _r, (_i, _j) in enumerate(zip(i, j)):
            M1[_r, _i] = -1
            M1[_r, _j] = 1
        for _r, (_q, _k) in enumerate(zip(q, k)):
            M2[_r, _q] = -1
            M2[_r, D + _k] = 1
        A = np.block(
            [
                [E, E],  # Heisenberg volume upper-bound
                [-E, -E],  # Heisenberg volume lower-bound
                [M1, Z],  # X_k anisotropy upper-bound
                [-M1, Z],  # X_k anisotropy lower-bound
                [Z, M1],  # V_k anisotropy upper-bound
                [Z, -M1],  # V_k anisotropy lower-bound
                [M2],  # XV_k anisotropy upper-bound
                [-M2],  # XV_k anisotropy lower-bound
            ]
        )
        Mw = np.log(np.sqrt(w_x * w_v))
        Mx = np.log(x_ptp[j]) - np.log(x_ptp[i])
        Mv = np.log(v_ptp[j]) - np.log(v_ptp[i])
        Mxv = np.log(v_ptp[k]) - np.log(x_ptp[q])
        b = np.r_[
            np.log(max_bbox_ratio) + Mw,  # Heisenberg volume upper-bound
            -Mw,  # Heisenberg volume lower-bound
            np.log(max_bbox_anisotropy) + Mx,  # X_k anisotropy upper-bound
            np.log(max_bbox_anisotropy) - Mx,  # X_k anisotropy lower-bound
            np.log(max_bbox_anisotropy) + Mv,  # V_k anisotropy upper-bound
            np.log(max_bbox_anisotropy) - Mv,  # V_k anisotropy lower-bound
            np.log(max_bbox_anisotropy) + Mxv,  # XV_k anisotropy upper-bound
            np.log(max_bbox_anisotropy) - Mxv,  # XV_k anisotropy lower-bound
        ]

        # Filter anisotropy constraints -------------------
        if domain in ("x", "v"):
            # Drop XV anisotropy constraints
            idx = slice(0, -2 * D_pow_2)
            A, b = A[idx], b[idx]
        if D > 1:
            # Drop self-domain anisotropy constraints if needed
            idx = dict(
                x=np.r_[np.arange(2 * D), np.arange(2 * D, -2 * D_choose_2)],  # constraints (1), (4)
                v=np.r_[np.arange(2 * D), np.arange(-2 * D_choose_2, 0)],  # constraints (1), (5)
                xv=slice(None),
            )[domain]
            A, b = A[idx], b[idx]

        # Solve LinProg -----------------------------------
        res = sopt.linprog(
            c=c,
            A_ub=A,
            b_ub=b,
            bounds=np.stack([lb, ub], axis=1),
            method="highs",
        )
        if res.success:
            x_bbox_dim = np.exp(res.x[:D])
            v_bbox_dim = np.exp(res.x[-D:])
            return x_bbox_dim, v_bbox_dim
        else:
            msg = "Auto-chunking failed given memory/anisotropy constraints."
            raise ValueError(msg)

    @staticmethod
    def _nu_partition_info(
        label: str,
        pts: np.ndarray,
        bbox_dim: np.ndarray,
    ):
        """
        Split `pts` into P `bbox_dim`-sized partitions.

        Parameters
        ----------
        label: "x", "v"
            Variable label.
        pts: ndarray[float]
            (M, D) point cloud.
        bbox_dim: ndarray[float]
            (D,) partition size.

        Returns
        -------
        info: namedtuple
            (P,) partition metadata, with fields:

            * ${label}_idx: ndarray[int]
                (M,) indices to re-order `pts` s.t. points in each partition are sequential.
            * ${label}_blk_bound: ndarray[int]
                (P+1,) indices into `${label}_idx` indicating where the p-th partition's points start/end.
                Partition `p` contains points
                    `pts[${label}_idx][${label}_blk_bound[p] : ${label}_blk_bound[p+1]]`
        """
        # Slightly increase bbox_dim to avoid "loner" partitions with single points.
        bbox_dim = bbox_dim * (1 + 1e-3)

        idx, bounds = ftk_cluster.grid_cluster(pts, bbox_dim)
        idx, bounds = ftk_cluster.fuse_cluster(pts, idx, bounds, bbox_dim)

        info = ftk_util.as_namedtuple(
            **{
                f"{label}_idx": idx,
                f"{label}_blk_bound": bounds,
            }
        )
        return info

    @staticmethod
    def _nu_cluster_info(
        label: str,
        pts: np.ndarray,
        idx: np.ndarray,
        blk_bound: np.ndarray,
        blk_center: np.ndarray,
        p0: np.ndarray,
        dp: np.ndarray,
        pN: np.ndarray,
        alpha: np.ndarray,
        max_cluster_size: int,
        max_window_ratio: np.ndarray,
    ):
        r"""
        Split `pts` into Q clusters spread across partitions.

        All partitions share the same lattice (p0, dp, pN) once shifted to the origin.

        Parameters
        ----------
        label: "x", "v"
            Variable label.
        pts: ndarray[float]
            (M, D) point cloud.
        idx: ndarray[int]
            (M,) indices to re-order `pts` s.t. points in each partition are sequential.
        blk_bound: ndarray[int]
            (P+1,) indices into `pts[idx]` indicating where the p-th partition starts/ends.
        blk_center: ndarray[float]
            (P, D) partition centroids.
        p0: ndarray[float]
            (D,) lattice starting point :math:`\bbp_{0} \in \bR^{D}`.
        dp: ndarray[float]
            (D,) lattice step :math:`\Delta_{p} \in \bR^{D}`.
        pN: ndarray[int]
            (D,) lattice node-count :math:`\{ N_{1},\ldots,N_{D} \} \in \bN^{D}`.
        alpha: ndarray[float]
            (D,) kernel scale factors :math:`\{ \alpha_{1},\ldots,\alpha_{D} \}`.
        max_cluster_size: int
            As described in ``UniformSpread.__init__``.
        max_window_ratio: ndarray[float]
            As described in ``UniformSpread.__init__``.

        Returns
        -------
        info: namedtuple
            (Q,) cluster metadata, with fields:

            * ${label}_idx: ndarray[int]
                (M,) indices to re-order `pts` s.t. points in each partition/cluster are sequential.
            * ${label}_cl_bound: ndarray[int]
                (Q+1,) indices into `${label}_idx` indicating where the q-th cluster's points start/end.
                Cluster `q` contains points
                    `pts[${label}_idx][${label}_cl_bound[q] : ${label}_cl_bound[q+1]]`
            * ${label}_anchor: ndarray[int]
                (Q, D) lower-left coordinate of each cluster w.r.t. the global grid.
            * ${label}_num: ndarray[int]
                (Q, D) cluster size w.r.t. the global grid.
        """
        idtype = np.int64

        P = len(blk_center)
        s = 1 / alpha  # kernel one-sided support
        bbox_dim = (2 * s) * max_window_ratio

        pts_idx = np.zeros_like(idx)
        pts_cl_bound = [None] * P
        pts_anchor = [None] * P
        pts_num = [None] * P
        for p in range(P):
            a, b = blk_bound[p : p + 2]
            _idx = idx[a:b]
            _pts = pts[_idx] - blk_center[p]

            # Group support points into clusters to match max window size.
            _pts_idx, _pts_cl_bound = ftk_cluster.grid_cluster(_pts, bbox_dim)
            _pts_idx, _pts_cl_bound = ftk_cluster.fuse_cluster(_pts, _pts_idx, _pts_cl_bound, bbox_dim)

            # Split clusters to match max cluster constraints
            _pts_cl_bound = ftk_cluster.bisect_cluster(_pts_cl_bound, max_cluster_size)

            # Compute off-grid lattice boundaries after spreading.
            _cl_min, _cl_max = ftk_numba.group_minmax(_pts, _pts_idx, _pts_cl_bound)
            LL = _cl_min - s  # lower-left lattice coordinate
            UR = _cl_max + s  # upper-right lattice coordinate

            # Get gridded equivalents.
            LL_idx = np.floor((LL - p0) / dp)
            UR_idx = np.ceil((UR - p0) / dp)

            # Clip LL/UR to lattice boundaries.
            LL_idx = np.fmax(0, LL_idx).astype(idtype)
            UR_idx = np.fmin(UR_idx, pN - 1).astype(idtype)

            pts_idx[a:b] = _idx[_pts_idx]  # indices w.r.t input `pts`
            pts_cl_bound[p] = _pts_cl_bound + a
            pts_anchor[p] = LL_idx
            pts_num[p] = UR_idx - LL_idx + 1

        # `pts_cl_bound[p]` contains (_Q+1,) entries.
        # Need to drop last term, except for final sub-partition.
        for p in range(P - 1):
            pts_cl_bound[p] = pts_cl_bound[p][:-1]

        info = ftk_util.as_namedtuple(
            **{
                f"{label}_idx": pts_idx,
                f"{label}_cl_bound": np.concatenate(pts_cl_bound),
                f"{label}_anchor": np.concatenate(pts_anchor),
                f"{label}_num": np.concatenate(pts_num),
            }
        )
        return info

    @classmethod
    def _init_metadata(
        cls,
        x: np.ndarray,
        x_idx: np.ndarray,
        x_blk_bound: np.ndarray,
        # -----------------------------
        v: np.ndarray,
        v_idx: np.ndarray,
        v_blk_bound: np.ndarray,
        # -----------------------------
        isign: int,
        eps: float,
        upsampfac: np.ndarray,
        upsampfac_ratio: np.ndarray,
        kernel_type: str,
        fft_nthreads: int,
        spread_nthreads: int,
    ):
        r"""
        Compute all NU->NU parameters.

        Returns
        -------
        info: namedtuple
            # general ---------------------------------------------------------
            * D: int                    [Transform Dimensionality]
            * isign: int                [Sign of the exponent]
            * sigma: (D,) float         [Total upsampling factor \sigma]
            * phi: Kernel               [Spread/interp pulse]
            * phiF: Kernel              [Spread/interp pulse FT]
            * beta: float               [\psi_{x/v}(s) = \phi_{\beta}(\alpha_{x/v} s)]
            * f_spread: callable        [Numba-compiled spread() function]
            * f_interpolate: callable   [Numba-compiled interpolate() function]
            # performance -----------------------------------------------------
            * fft_nthreads: int         [Thread-count for FFTs]
            * spread_nthreads: int      [Thread-count for spreading/interpolation]
            # FFS-related -----------------------------------------------------
            * T: (D,) float             [FFS period]
            * K: (D,) int               [Max FS frequency computed]
            * L: (D,) int               [FFS transform length]
            * F_x0: (D,) float          [FFS x-lattice starting point \bbx_{0}]
            * F_dx: (D,) float          [FFS x-lattice pitch \Delta_{x}]
            * F_v0: (D,) float          [FFS v-lattice starting point \bbv_{0}]
            * F_dv: (D,) float          [FFS v-lattice pitch \Delta_{v}]
            # x-related -------------------------------------------------------
            * M: int                    [Number of x-domain points]
            * x: (M, D) float           [x-domain support points; canonical order]
            * x_idx: (M,) int           [Permutation indices; re-orders `w` to partition/cluster order]
            * Px: int                   [Number of x-domain partitions]
            * Qx: int                   [Number of x-domain clusters]
            * x_blk_bound: (Px+1,) int  [Px partition boundaries]
            * x_cl_bound: (Qx+1,) int   [Qx cluster boundaries]
            * x_anchor: (Qx, D) int     [Qx cluster lower-left coordinates on FFS x-lattice]
            * x_num: (Qx, D) int        [Qx cluster dimensions on FFS x-lattice]
            * sigma_x: (D,) float       [upsampling factor \sigma_{x}]
            * alpha_x: (D,) float       [\psi_{x}(s) = \phi_{\beta}(\alpha_{x} s)]
            * w_x: (D,) int             [\psi_{x} support in #samples]
            * Xc: (Px, D) float         [Px partition centroids]
            * Xd0: (D,) float           [Max x-domain partition spread X_{d}^{0}; support of f_{0}]
            * Xd: (D,) float            [Max x-domain partition spread X_{d};     support of g_{0}]
            # v-related -------------------------------------------------------
            * N: int                    [Number of v-domain points]
            * v: (N, D) float           [v-domain support points; canonical order]
            * v_idx: (N,) int           [Permutation indices; re-orders `z` to partition/cluster order]
            * Pv: int                   [Number of v-domain partitions]
            * Qv: int                   [Number of v-domain clusters]
            * v_blk_bound: (Pv+1,) int  [Pv partition boundaries]
            * v_cl_bound: (Qv+1,) int   [Qv cluster boundaries]
            * v_anchor: (Qv, D) int     [Qv cluster lower-left coordinates on FFS v-lattice]
            * v_num: (Qv, D) int        [Qv cluster dimensions on FFS v-lattice]
            * sigma_v: (D,) float       [upsampling factor \sigma_{v}]
            * alpha_v: (D,) float       [\psi_{v}(s) = \phi_{\beta}(\alpha_{v} s)]
            * w_v: (D,) int             [\psi_{v} support in #samples]
            * Vc: (Pv, D) float         [Pv partition centroids]
            * Vd0: (D,) float           [Max v-domain partition spread V_{d}^{0}; extent of freqs wanted]
            * Vd: (D,) float            [Max v-domain partition spread V_{d};     extent of freqs needed]
        """
        D = x.shape[1]

        # As much x-domain stuff as possible --------------
        M = len(x)
        Px = len(x_blk_bound) - 1
        x_min, x_max = ftk_numba.group_minmax(x, x_idx, x_blk_bound)
        Xc = (x_max + x_min) / 2
        Xd0 = (x_max - x_min).max(axis=0)
        sigma_x = upsampfac**upsampfac_ratio

        # As much v-domain stuff as possible --------------
        N = len(v)
        Pv = len(v_blk_bound) - 1
        v_min, v_max = ftk_numba.group_minmax(v, v_idx, v_blk_bound)
        Vc = (v_max + v_min) / 2
        Vd0 = (v_max - v_min).max(axis=0)
        sigma_v = upsampfac ** (1 - upsampfac_ratio)

        # To be overwritten after _nu_cluster_info() ------
        Qx = None
        x = x  # user-order for now
        x_idx = x_idx  # partition-order only for now
        x_cl_bound = None
        x_anchor = None
        x_num = None
        Qv = None
        v = v  # user-order for now
        v_idx = v_idx  # partition-order only for now
        v_cl_bound = None
        v_anchor = None
        v_num = None

        # (x,v)-dependant stuff ---------------------------
        w_x, w_v = cls._infer_kernel_widths(
            eps,
            upsampfac,
            upsampfac_ratio,
        )
        Xd0, Vd0 = cls._grow_to_min_vol(Xd0, Vd0, w_x, w_v)
        Xd = Xd0 + (w_x / Vd0)
        Vd = Vd0 + (w_v / Xd0)

        # FFS-related stuff -------------------------------
        sigma = upsampfac
        T = Xd * sigma_x
        K = np.ceil(sigma * Xd * Vd / 2).astype(int)
        L = ftk_ffs.FFS.next_fast_len(K)
        x_lattice = ftk_ffs.FFS(T, K, L).sample_points(np.double)
        F_x0 = np.array([_l[0] for _l in x_lattice])
        F_dx = np.array([_l[1] - _l[0] for _l in x_lattice])
        v_lattice = ftk_ffs.FFS(T, K, L).freq_points(np.double)
        F_v0 = np.array([_l[0] for _l in v_lattice])
        F_dv = np.array([_l[1] - _l[0] for _l in v_lattice])

        # kernel stuff ------------------------------------
        alpha_x = (2 * L) / (T * w_x)
        alpha_v = (2 * T) / w_v
        beta = ftk_kernel.KaiserBessel.beta_from_eps(eps)
        if kernel_type == "kb":
            phi = ftk_kernel.KaiserBessel(beta)
            phiF = ftk_kernel.KaiserBesselF(beta)
        elif kernel_type == "kb_ppoly":
            log10_eps = np.log10(eps)
            B_ppoly = w_x[0]
            if -np.inf <= log10_eps <= -14:
                N_ppoly = 9
            elif -14 < log10_eps <= -7:
                N_ppoly = 7
            elif -7 < log10_eps <= -5:
                N_ppoly = 6
            elif -5 < log10_eps <= -4:
                N_ppoly = 5
            elif -4 < log10_eps <= 0:
                N_ppoly = 4
            phi = ftk_kernel.PPoly.from_kernel(
                ftk_kernel.KaiserBessel(beta),
                B=B_ppoly,  # number of bins
                N=N_ppoly,  # polynomial order
                sym=True,
            )
            phiF = ftk_kernel.KaiserBesselF(beta)

        # spread/interp compiled code ---------------------
        # We need the compiled [spread,interpolate]() functions which UniformSpread uses, but will apply them manually
        # instead of via UniformSpread.[apply,adjoint]().  We therefore construct a cheap US instance just to extract
        # these objects.
        u_spread = ftk_spread.UniformSpread(
            x=np.zeros((1, D), dtype=np.double),  # doesn't matter
            z_spec=dict(start=F_x0, step=F_dx, num=L),
            phi=phi.low_level_callable(ufunc=False),
            alpha=alpha_x,
        )
        f_spread = u_spread._spread
        f_interpolate = u_spread._interpolate

        info = ftk_util.as_namedtuple(
            # general ------------------
            D=D,
            isign=isign,
            sigma=sigma,
            phi=phi,
            phiF=phiF,
            beta=beta,
            f_spread=f_spread,
            f_interpolate=f_interpolate,
            # performance --------------
            fft_nthreads=fft_nthreads,
            spread_nthreads=spread_nthreads,
            # FFS-related --------------
            T=T,
            K=K,
            L=L,
            F_x0=F_x0,
            F_dx=F_dx,
            F_v0=F_v0,
            F_dv=F_dv,
            # x-related ----------------
            M=M,
            x=x,
            x_idx=x_idx,
            Px=Px,
            Qx=Qx,
            x_blk_bound=x_blk_bound,
            x_cl_bound=x_cl_bound,
            x_anchor=x_anchor,
            x_num=x_num,
            sigma_x=sigma_x,
            alpha_x=alpha_x,
            w_x=w_x,
            Xc=Xc,
            Xd0=Xd0,
            Xd=Xd,
            # v-related ----------------
            N=N,
            v=v,
            v_idx=v_idx,
            Pv=Pv,
            Qv=Qv,
            v_blk_bound=v_blk_bound,
            v_cl_bound=v_cl_bound,
            v_anchor=v_anchor,
            v_num=v_num,
            sigma_v=sigma_v,
            alpha_v=alpha_v,
            w_v=w_v,
            Vc=Vc,
            Vd0=Vd0,
            Vd=Vd,
        )
        return info

    @classmethod
    def _grow_to_min_vol(
        cls,
        x_ptp: np.ndarray,
        v_ptp: np.ndarray,
        w_x: np.ndarray,
        w_v: np.ndarray,
    ) -> tuple[np.ndarray]:
        """
        Grow (x,v) bbox dimensions to guarantee Heisenberg volume lower-bound is attained.

        Parameters
        ----------
        x_ptp, v_ptp: ndarray[float]
            (D,) bbox dimensions in (x,v)
        w_x, w_v: ndarray[int]
            (D,) kernel (x,v) support in #samples

        Returns
        -------
        x_ptp2, v_ptp2: ndarray[float]
            (D,) bbox dimensions in (x,v) at least as large as Heisenberg volume lower-bound.
        """
        D = len(x_ptp)
        min_vol = np.sqrt(w_x * w_v)
        min_vol_sqrt = np.sqrt(min_vol)

        x_ptp2, v_ptp2 = x_ptp.copy(), v_ptp.copy()
        for d in range(D):
            if x_ptp[d] * v_ptp[d] < min_vol[d]:
                if (x_ptp[d] < min_vol_sqrt[d]) and (v_ptp[d] < min_vol_sqrt[d]):
                    x_ptp2[d] = v_ptp2[d] = min_vol_sqrt[d]
                elif x_ptp[d] < v_ptp[d]:
                    x_ptp2[d] = min_vol[d] / v_ptp[d]
                elif x_ptp[d] > v_ptp[d]:
                    v_ptp2[d] = min_vol[d] / x_ptp[d]
        return x_ptp2, v_ptp2

    def _fw_spread(self, w: np.ndarray) -> np.ndarray:
        r"""
        Sample g_{0}(x) on the FFS lattice, where

           g_{0}^{(p_x, p_v)}(x)
           =
           \sum_{m}
               w_{m}^{(p_x)}
               \ee^{ -\cj 2\pi V_{c}^{(p_v)} [ x_{m}^{(p_x)} - X_{c}^{(p_x)} ] }
               \psi_{x}( x - [ x_{m}^{(p_x)} - X_{c}^{(p_x)} ] )

        Parameters
        ----------
        w: ndarray
            (..., M) in canonical x-order.

        Returns
        -------
        g0: ndarray
            (Px, Pv, ..., L1,...,LD) g_{0} samples on FFS(T,K,L) lattice.
        """
        translate = ftk_util.TranslateDType(w.dtype)
        fdtype = translate.to_float()
        cdtype = translate.to_complex()

        # reshape/cast (x, w)
        sh = w.shape[:-1]  # (...,)
        Ns = int(np.prod(sh))
        w = w.reshape(Ns, self.cfg.M)
        x = ftk_util.cast_warn(self.cfg.x, fdtype)
        Xc = self.cfg.Xc.astype(fdtype, copy=False)
        Vc = self.cfg.Vc.astype(fdtype, copy=False)

        # lattice extractor
        lattice = ftk_ffs.FFS(  # (L1,),...,(LD,)
            self.cfg.T,
            self.cfg.K,
            self.cfg.L,
        ).sample_points(fdtype)
        roi = lambda q: tuple(  # int -> tuple[slice]
            slice(n0, n0 + num)
            for (n0, num) in zip(
                self.cfg.x_anchor[q],
                self.cfg.x_num[q],
            )
        )
        l_roi = lambda q: tuple(  # int -> tuple[ndarray]
            _l[_roi]
            for (_l, _roi) in zip(
                lattice,
                roi(q),
            )
        )

        # kernel (func, scale)
        k_func = (self.cfg.phi.low_level_callable(ufunc=False),) * self.cfg.D
        k_scale = self.cfg.alpha_x.astype(fdtype)  # (D,)

        # spread each cluster onto its sub-grid on the right partition
        fs, fs2pq = [None] * self.cfg.Qx, dict()
        with cf.ThreadPoolExecutor(max_workers=self.cfg.spread_nthreads) as executor:
            px = 0
            for qx in range(self.cfg.Qx):
                a, b = self.cfg.x_blk_bound[px : px + 2]
                c, d = self.cfg.x_cl_bound[qx : qx + 2]
                px += int(b == c)

                Mx = d - c
                _x = x[c:d, :] - Xc[px]  # (Mx, D)
                _mod = ftk_complex.cexp((-2 * np.pi) * (Vc @ _x.T))  # (Pv, Mx)
                _w = np.reshape(  # (Pv*Ns, Mx)
                    w[:, c:d] * _mod[:, np.newaxis, :],
                    (self.cfg.Pv * Ns, Mx),
                )

                future = executor.submit(
                    self.cfg.f_spread,
                    x=_x,
                    w=_w,
                    z=l_roi(qx),
                    phi=k_func,
                    a=k_scale,
                )

                fs[qx], fs2pq[future] = future, (px, qx)

        # update global grid
        g0 = np.zeros((self.cfg.Px, self.cfg.Pv, *sh, *self.cfg.L), dtype=cdtype)
        for future in cf.as_completed(fs):
            px, qx = fs2pq[future]
            g0_roi = future.result()

            g0[px, ..., *roi(qx)] += g0_roi.reshape(
                self.cfg.Pv,
                *sh,
                *self.cfg.x_num[qx],
            )
        return g0

    def _bw_interpolate(self, g0: np.ndarray) -> np.ndarray:
        r"""
        Adjoint of _fw_spread()

        Parameters
        ----------
        g0: ndarray
            (Px, Pv, ..., L1,...,LD) g_{0} samples on FFS(T,K,L) lattice.

        Returns
        -------
        w: ndarray
            (..., M) in canonical x-order.
        """
        translate = ftk_util.TranslateDType(g0.dtype)
        fdtype = translate.to_float()
        cdtype = translate.to_complex()

        # reshape/cast (x, g0)
        sh = g0.shape[2 : -self.cfg.D]  # (...,)
        Ns = int(np.prod(sh))
        g0 = g0.reshape(self.cfg.Px, self.cfg.Pv * Ns, *self.cfg.L)  # (Px, Pv*Ns, L1,...,LD)
        x = ftk_util.cast_warn(self.cfg.x, fdtype)
        Xc = self.cfg.Xc.astype(fdtype, copy=False)
        Vc = self.cfg.Vc.astype(fdtype, copy=False)

        # lattice extractor
        lattice = ftk_ffs.FFS(  # (L1,),...,(LD,)
            self.cfg.T,
            self.cfg.K,
            self.cfg.L,
        ).sample_points(fdtype)
        roi = lambda q: tuple(  # int -> tuple[slice]
            slice(n0, n0 + num)
            for (n0, num) in zip(
                self.cfg.x_anchor[q],
                self.cfg.x_num[q],
            )
        )
        l_roi = lambda q: tuple(  # int -> tuple[ndarray]
            _l[_roi]
            for (_l, _roi) in zip(
                lattice,
                roi(q),
            )
        )

        # kernel (func, scale)
        k_func = (self.cfg.phi.low_level_callable(ufunc=False),) * self.cfg.D
        k_scale = self.cfg.alpha_x.astype(fdtype)  # (D,)

        # interpolate each sub-grid onto sample points in the right partition.
        fs, fs2pq = [None] * self.cfg.Qx, dict()
        with cf.ThreadPoolExecutor(max_workers=self.cfg.spread_nthreads) as executor:
            px = 0
            for qx in range(self.cfg.Qx):
                a, b = self.cfg.x_blk_bound[px : px + 2]
                c, d = self.cfg.x_cl_bound[qx : qx + 2]
                px += int(b == c)

                _x = x[c:d, :] - Xc[px]  # (Mx, D)

                future = executor.submit(
                    self.cfg.f_interpolate,
                    x=_x,
                    g=g0[px, :, *roi(qx)],
                    z=l_roi(qx),
                    phi=k_func,
                    a=k_scale,
                )

                fs[qx], fs2pq[future] = future, (px, qx)

        # mod/reduce v-partitions
        w = np.zeros((*sh, self.cfg.M), dtype=cdtype)
        for future in cf.as_completed(fs):
            px, qx = fs2pq[future]
            c, d = self.cfg.x_cl_bound[qx : qx + 2]
            Mx = d - c

            _w = np.reshape(  # (Pv*Ns, Mx) -> (Pv, Ns, Mx)
                future.result(),
                (self.cfg.Pv, Ns, Mx),
            )
            _x = x[c:d, :] - Xc[px]  # (Mx, D)
            _mod = ftk_complex.cexp((2 * np.pi) * (Vc @ _x.T))  # (Pv, Mx)
            _w = np.sum(  # (Ns, Mx)
                _w * _mod[:, np.newaxis, :],
                axis=0,
            )

            w[..., c:d] = _w.reshape(*sh, Mx)
        return w

    def _de_window(self, g0: np.ndarray) -> np.ndarray:
        r"""
        Sample h(x) on the FFS lattice, where

            h(x)
            =
            g_{0}(x) / \psi_{v}^{\ctft}(x)

        Parameters
        ----------
        g0: ndarray
            (..., L1,...,LD) g_{0} samples on FFS(T,K,L) lattice.

        Returns
        -------
        h: ndarray
            (..., L1,...,LD) h samples on FFS(T,K,L) lattice.
        """
        translate = ftk_util.TranslateDType(g0.dtype)
        fdtype = translate.to_float()

        # build \psiF_v samples
        psiF_v = [None] * self.cfg.D
        ffs = ftk_ffs.FFS(self.cfg.T, self.cfg.K, self.cfg.L)
        for d, (x, av) in enumerate(
            zip(
                ffs.sample_points(fdtype),
                self.cfg.alpha_v.astype(fdtype),
            )
        ):
            psiF_v[d] = self.cfg.phiF(x / av) / av

        w_inv = tuple(map(np.reciprocal, psiF_v))
        h = ftk_linalg.hadamard_outer(g0, *w_inv)
        return h

    def _ffs_transform(self, h: np.ndarray, forward: bool) -> np.ndarray:
        r"""
        Apply FFS or FFS-adjoint.

        Parameters
        ----------
        h: ndarray
            [forward=True]  (..., L1,...,LD) signal samples on FFS(T,K,L) lattice.
            [forward=False] (..., L1,...,LD) FS samples produced by FFS(T,K,L), with padding.
        forward: bool

        Returns
        -------
        hFS: ndarray
            [forward=True]  (..., L1,...,LD) FS estimates (-K,...,K,<padding>).
            [forward=False] (..., L1,...,LD) adjoint signal sampled on the FFS(T,K,L) lattice.
        """
        ffs = ftk_ffs.FFS(
            self.cfg.T,
            self.cfg.K,
            self.cfg.L,
            nthreads=self.cfg.fft_nthreads,
        )
        if forward:
            hFS = ffs.apply(h)
        else:
            hFS = ffs.adjoint(h)
        return hFS

    @staticmethod
    def _transpose_blks(A: np.ndarray) -> np.ndarray:
        """
        Transpose (x, v) partition axes.

        Parameters
        ----------
        A: ndarray
            (Px, Pv, ...)
            OR
            (Pv, Px, ...)

        Returns
        -------
        B: ndarray
            (Pv, Px, ...) C-contiguous
            OR
            (Px, Pv, ...) C-contiguous
        """
        axes = (1, 0, *range(2, A.ndim))
        B = np.require(
            A.transpose(axes),
            requirements="C",
        )
        return B

    def _fw_interpolate(self, hFS: np.ndarray) -> np.ndarray:
        r"""
        Interpolate/reduce h^{\fs} onto NU points

            g^{\ctft (p_v)}(v_{n}^{(p_v)})
            =
            \sum_{p_x = 1...Px} \ee^{ -\cj 2\pi X_{c}^{(p_x)} v_{n}^{(p_v)} }
            \sum_{k} hFS_{k}^{(p_x, p_v)}
                     \psi_{v}(
                        [ v_{n}^{(p_v)} - V_{c}^{(p_v)} ]
                        - \frac{k}{T}
                      )

        Parameters
        ----------
        hFS: ndarray
            (Pv, Px, ..., L1,...,LD) FS coefficients [from FFS w/ padding].

        Returns
        -------
        gF: ndarray
            (..., N) g^{F} in canonical v-order.
        """
        translate = ftk_util.TranslateDType(hFS.dtype)
        fdtype = translate.to_float()
        cdtype = translate.to_complex()

        # reshape/cast (v, hFS)
        sh = hFS.shape[2 : -self.cfg.D]  # (...,)
        Ns = int(np.prod(sh))
        hFS = hFS.reshape(self.cfg.Pv, self.cfg.Px * Ns, *self.cfg.L)  # (Pv, Px*Ns, L1,...,LD)
        v = ftk_util.cast_warn(self.cfg.v, fdtype)
        Xc = self.cfg.Xc.astype(fdtype, copy=False)
        Vc = self.cfg.Vc.astype(fdtype, copy=False)

        # lattice extractor
        lattice = ftk_ffs.FFS(  # (2K1+1,),...,(2KD+1,)
            self.cfg.T,
            self.cfg.K,
            self.cfg.L,
        ).freq_points(fdtype)
        roi = lambda q: tuple(  # int -> tuple[slice]
            slice(n0, n0 + num)
            for (n0, num) in zip(
                self.cfg.v_anchor[q],
                self.cfg.v_num[q],
            )
        )
        l_roi = lambda q: tuple(  # int -> tuple[ndarray]
            _l[_roi]
            for (_l, _roi) in zip(
                lattice,
                roi(q),
            )
        )

        # kernel (func, scale)
        k_func = (self.cfg.phi.low_level_callable(ufunc=False),) * self.cfg.D
        k_scale = self.cfg.alpha_v.astype(fdtype)  # (D,)

        # interpolate each sub-grid onto freq points in the right partition.
        fs, fs2pq = [None] * self.cfg.Qv, dict()
        with cf.ThreadPoolExecutor(max_workers=self.cfg.spread_nthreads) as executor:
            pv = 0
            for qv in range(self.cfg.Qv):
                a, b = self.cfg.v_blk_bound[pv : pv + 2]
                c, d = self.cfg.v_cl_bound[qv : qv + 2]
                pv += int(b == c)

                _v = v[c:d, :] - Vc[pv]  # (Nv, D)

                future = executor.submit(
                    self.cfg.f_interpolate,
                    x=_v,
                    g=hFS[pv, :, *roi(qv)],
                    z=l_roi(qv),
                    phi=k_func,
                    a=k_scale,
                )

                fs[qv], fs2pq[future] = future, (pv, qv)

        # mod/reduce x-partitions
        gF = np.zeros((*sh, self.cfg.N), dtype=cdtype)
        for future in cf.as_completed(fs):
            pv, qv = fs2pq[future]
            c, d = self.cfg.v_cl_bound[qv : qv + 2]
            Nv = d - c

            _g0F = np.reshape(  # (Px*Ns, Nv) -> (Px, Ns, Nv)
                future.result(),
                (self.cfg.Px, Ns, Nv),
            )
            _v = v[c:d, :]  # (Nv, D)
            _mod = ftk_complex.cexp((-2 * np.pi) * (Xc @ _v.T))  # (Px, Nv)
            _gF = np.sum(  # (Ns, Nv)
                _g0F * _mod[:, np.newaxis, :],
                axis=0,
            )

            gF[..., c:d] = _gF.reshape(*sh, Nv)
        return gF

    def _bw_spread(self, gF: np.ndarray) -> np.ndarray:
        r"""
        Adjoint of _fw_interpolate()

        Parameters
        ----------
        gF: ndarray
            (..., N) g^{F} in canonical v-order.

        Returns
        -------
        hFS: ndarray
            (Pv, Px, ..., L1,...,LD) FS coefficients [w/ FFS padding].
        """
        translate = ftk_util.TranslateDType(gF.dtype)
        fdtype = translate.to_float()
        cdtype = translate.to_complex()

        # reshape/cast (v, gF)
        sh = gF.shape[:-1]  # (...,)
        Ns = int(np.prod(sh))
        gF = gF.reshape(Ns, self.cfg.N)
        v = ftk_util.cast_warn(self.cfg.v, fdtype)
        Xc = self.cfg.Xc.astype(fdtype, copy=False)
        Vc = self.cfg.Vc.astype(fdtype, copy=False)

        # lattice extractor
        lattice = ftk_ffs.FFS(  # (2K1+1,),...,(2KD+1,)
            self.cfg.T,
            self.cfg.K,
            self.cfg.L,
        ).freq_points(fdtype)
        roi = lambda q: tuple(  # int -> tuple[slice]
            slice(n0, n0 + num)
            for (n0, num) in zip(
                self.cfg.v_anchor[q],
                self.cfg.v_num[q],
            )
        )
        l_roi = lambda q: tuple(  # int -> tuple[ndarray]
            _l[_roi]
            for (_l, _roi) in zip(
                lattice,
                roi(q),
            )
        )

        # kernel (func, scale)
        k_func = (self.cfg.phi.low_level_callable(ufunc=False),) * self.cfg.D
        k_scale = self.cfg.alpha_v.astype(fdtype)  # (D,)

        # spread each cluster onto its sub-grid on the right partition.
        fs, fs2pq = [None] * self.cfg.Qv, dict()
        with cf.ThreadPoolExecutor(max_workers=self.cfg.spread_nthreads) as executor:
            pv = 0
            for qv in range(self.cfg.Qv):
                a, b = self.cfg.v_blk_bound[pv : pv + 2]
                c, d = self.cfg.v_cl_bound[qv : qv + 2]
                pv += int(b == c)

                Nv = d - c
                _v = v[c:d, :]  # (Nv, D)
                _mod = ftk_complex.cexp((2 * np.pi) * (Xc @ _v.T))  # (Px, Nv)
                _gF = np.reshape(  # (Px*Ns, Nv)
                    gF[:, c:d] * _mod[:, np.newaxis, :],
                    (self.cfg.Px * Ns, Nv),
                )

                future = executor.submit(
                    self.cfg.f_spread,
                    x=_v - Vc[pv],
                    w=_gF,
                    z=l_roi(qv),
                    phi=k_func,
                    a=k_scale,
                )

                fs[qv], fs2pq[future] = future, (pv, qv)

        # update global grid
        hFS = np.zeros((self.cfg.Pv, self.cfg.Px, *sh, *self.cfg.L), dtype=cdtype)
        for future in cf.as_completed(fs):
            pv, qv = fs2pq[future]
            hFS_roi = future.result()

            hFS[pv, ..., *roi(qv)] += hFS_roi.reshape(
                self.cfg.Px,
                *sh,
                *self.cfg.v_num[qv],
            )
        return hFS

    def _de_convolve(self, gF: np.ndarray) -> np.ndarray:
        r"""
        De-convolve g^{\ctft}(v) to obtain f^{\ctft}, where

            f^{\ctft}(v_{n}^{(p_v)})
            =
            g^{\ctft}(v_{n}^{(p_v)})
            /
            \psi_{x}^{\ctft}(v_{n}^{(p_v)} - V_{c}^{(p_v)})

        Parameters
        ----------
        gF: ndarray
            (..., N) g^{\ctft} samples; in canonical v-order.

        Returns
        -------
        z: ndarray
            (..., N) f^{\ctft} samples; in canonical v-order.
        """
        translate = ftk_util.TranslateDType(gF.dtype)
        fdtype = translate.to_float()

        v = self.cfg.v.astype(fdtype)  # (N, D)
        for pv in range(self.cfg.Pv):
            a, b = self.cfg.v_blk_bound[pv : pv + 2]
            v[a:b] -= self.cfg.Vc[pv]

        # build \psiF_x samples
        psiF_x = np.ones(self.cfg.N, dtype=fdtype)
        for _v, ax in zip(
            v.T,
            self.cfg.alpha_x.astype(fdtype),
        ):
            psiF_x *= self.cfg.phiF(_v / ax) / ax

        z = gF / psiF_x
        return z

    def _flip_sign(self, A: np.ndarray) -> np.ndarray:
        """
        Conditionally conjugate input.

        Parameters
        ----------
        A: ndarray[float/complex]
            (...,)

        Returns
        -------
        B: ndarray[float/complex]
            (...,) conjugated array if `isign==1`
        """
        if self.cfg.isign == -1:
            B = A
        else:
            B = A.conj()
        return B

    def _re_order_x(self, w: np.ndarray, forward: bool) -> np.ndarray:
        r"""
        Re-order x-domain coordinates from/to canonical (partition, cluster) order.

        Parameters
        ----------
        w: ndarray
            [forward=True]  (..., M) in user x-order.
            [forward=False] (..., M) in canonical x-order.
        forward: bool

        Returns
        -------
        w2: ndarray
            [forward=True]  (..., M) in canonical x-order.
            [forward=False] (..., M) in user x-order.
        """
        if forward:
            w2 = w[..., self.cfg.x_idx]  # (..., M)
        else:
            w2 = np.zeros_like(w)  # (..., M)
            w2[..., self.cfg.x_idx] = w
        return w2

    def _re_order_v(self, z: np.ndarray, forward: bool) -> np.ndarray:
        r"""
        Re-order v-domain coordinates from/to canonical (partition, cluster) order.

        Parameters
        ----------
        z: ndarray
            [forward=True]  (..., N) in canonical v-order.
            [forward=False] (..., N) in user v-order.
        forward: bool

        Returns
        -------
        z2: ndarray
            [forward=True]  (..., N) in user v-order.
            [forward=False] (..., N) in canonical v-order.
        """
        if forward:
            z2 = np.zeros_like(z)  # (..., N)
            z2[..., self.cfg.v_idx] = z
        else:
            z2 = z[..., self.cfg.v_idx]  # (..., N)
        return z2


NU2NU = NonUniform2NonUniform  # alias
