import warnings

import numba as nb
import numpy as np

import fourier_toolkit.complex as ftk_complex
import fourier_toolkit.ffs as ftk_ffs
import fourier_toolkit.kernel as ftk_kernel
import fourier_toolkit.linalg as ftk_linalg
import fourier_toolkit.nu2nu as ftk_nu2nu
import fourier_toolkit.numba as ftk_numba
import fourier_toolkit.spread as ftk_spread
import fourier_toolkit.util as ftk_util

__all__ = [
    "NonUniform2Uniform",
    "NU2U",
]

# Disable all warnings for the entire module
warnings.filterwarnings("ignore", category=nb.NumbaExperimentalFeatureWarning)


class NonUniform2Uniform(ftk_nu2nu.NU2NU):
    r"""
    Multi-dimensional NonUniform-to-Uniform Fourier Transform.

    Given the Dirac stream

    .. math::

       f(\bbx) = \sum_{m} w_{m} \delta(\bbx - \bbx_{m}),

    computes samples of :math:`f^{F}`, i.e.,

    .. math::

       f^{F}(\bbv_{n}) = \bbz_{n} = \sum_{m} w_{m} \ee^{ -\cj 2\pi \innerProduct{\bbx_{m}}{\bbv_{n}} },

    where :math:`\bbx_{m} \in \bR^{D}`, and :math:`\bbv_{n}` lies on the regular lattice

    .. math::

       \begin{align}
           \bbv_{\bbn} &= \bbv_{0} + \Delta_{\bbv} \odot \bbn, & [\bbn]_{d} \in \{0,\ldots,N_{d}-1\},
       \end{align}

    with :math:`N = \prod_{d} N_{d}`.
    """

    def __init__(
        self,
        x: np.ndarray,
        v_spec: dict[str, np.ndarray],
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
        max_bbox_ratio: tuple[float] = 10,
        max_bbox_anisotropy: float = 5,
    ):
        r"""
        Parameters
        ----------
        x: ndarray
            (M, D) support points :math:`\bbx_{m} \in \bR^{D}`.
        v_spec: dict[str, ndarray]
            :math:`\bbv_{n}` lattice specifier, with keys:

            * `start`: (D,) values :math:`\bbv_{0} \in \bR^{D}`.
            * `step` : (D,) values :math:`\Delta_{\bbv} \in \bR^{D}`.
            * `num`  : (D,) values :math:`\{ N_{1},\ldots,N_{D} \} \in \bN^{D}`.
              This parameter fixes the dimensionality `D` of the transform.

            Scalars are broadcasted to all dimensions.
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
            Perform NU->U transform by partitioning the x-domain.
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

            Setting close to 1 favors cubeoid-shaped partitions of x-space.
            Setting large allows x-partitions to be highly asymmetric.
        """
        if x.ndim == 1:
            x = x[:, np.newaxis]
        _, D = x.shape
        v_spec["start"] = ftk_util.broadcast_seq(v_spec["start"], D, np.double)
        v_spec["step"] = ftk_util.broadcast_seq(v_spec["step"], D, np.double)
        v_spec["num"] = ftk_util.broadcast_seq(v_spec["num"], D, np.int64)
        assert np.all(v_spec["step"] > 0)
        assert np.all(v_spec["num"] > 0)

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
            domain="x",
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
            *(np.ptp(x, axis=0), v_spec["step"] * (v_spec["num"] - 1)),
            *(w_x, w_v),
            p.domain,
            *(p.max_bbox_ratio, p.max_bbox_anisotropy),
        )
        x_blk_info = self._nu_partition_info("x", x, x_bbox_dim)
        # =====================================================================

        cfg = self._init_metadata(
            x=x,
            **x_blk_info._asdict(),
            # -------------------------
            v0=v_spec["start"],
            dv=v_spec["step"],
            N=v_spec["num"],
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
        # =====================================================================

        # adjust contents of `cfg`:
        #   - (x,) -> partition/cluster order. [Was user-order before.]
        #   - (x_idx,) -> partition AND cluster order. [Was partition-order before.]
        #   - adds extra fields from _cluster_info().
        cfg = cfg._asdict()
        cfg.update(
            x=x[x_info.x_idx],  # canonical partition/cluster order
            Qx=len(x_info.x_cl_bound) - 1,
            **x_info._asdict(),
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
            (..., N1,...,ND) weights :math:`z_{n} \in \bC^{D}`.
        """
        z = super().apply(w)
        return z

    def adjoint(self, z: np.ndarray) -> np.ndarray:
        r"""
        Parameters
        ----------
        z: ndarray[float/complex]
            (..., N1,...,ND) weights :math:`z_{n} \in \bC^{D}`.

        Returns
        -------
        w: ndarray[complex]
            (..., M) weights :math:`w_{m} \in \bC^{D}`.
        """
        w = super().adjoint(z)
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
        # We can piggy-back onto NU2NU.diagnostic_plot() so long as we create the `cfg.v` entry.
        # The idea is to swap out `cfg` with a substitute having this property, then put the original one back.
        cfg_orig = self.cfg

        # We don't actually need to synthesize `v` fully: the 2 corners of its bounding box suffice.
        v = np.stack(  # (2, D)
            [
                self.cfg.v0,  # LL node
                self.cfg.v0 + self.cfg.dv * (self.cfg.N - 1),  # UR node
            ],
            axis=0,
        )
        cfg_new = self.cfg._asdict()
        cfg_new.update(v=v)
        self.cfg = ftk_util.as_namedtuple(**cfg_new)

        ax = super().diagnostic_plot(
            domain=domain,
            axes=axes,
            ax=ax,
        )

        self.cfg = cfg_orig  # put original metadata back in place
        return ax

    # Helper routines (internal) ----------------------------------------------
    @classmethod
    def _init_metadata(
        cls,
        x: np.ndarray,
        x_idx: np.ndarray,
        x_blk_bound: np.ndarray,
        # -----------------------------
        v0: np.ndarray,
        dv: np.ndarray,
        N: np.ndarray,
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
        Compute all NU->U parameters.

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
            * N: (N1,...,ND) int        [Number of v-domain points]
            * v0: (D,) float            [v-domain lattice starting point \bbv_{0}]
            * dv: (D,) float            [v-domain lattice pitch \Delta_{v}]
            * Pv: int = 1               [Number of v-domain partitions; formatted as such to re-use NU2NU methods]
            * sigma_v: (D,) float       [upsampling factor \sigma_{v}]
            * alpha_v: (D,) float       [\psi_{v}(s) = \phi_{\beta}(\alpha_{v} s)]
            * w_v: (D,) int             [\psi_{v} support in #samples]
            * Vc: (Pv, D) float         [Pv partition centroids; formatted as such to re-use NU2NU methods]
            * Vd0: (D,) float           [v-domain partition spread V_{d}^{0}; extent of freqs wanted]
            * Vd: (D,) float            [v-domain partition spread V_{d};     extent of freqs needed]
            * Ov: (D,) int >= 1         [v-domain oversampling factor \bbO_{v}]
            * Sv: (D,) int >= 1         [v-domain subsampling factor \bbS_{v}]
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
        N = N
        v0 = v0
        dv = dv
        Pv = 1
        Vc = (v0 + dv * (N - 1) / 2).reshape(Pv, D)
        Vd0 = dv * (N - 1)
        sigma_v = upsampfac ** (1 - upsampfac_ratio)

        # To be overwritten after _cluster_info() ---------
        Qx = None
        x = x  # user-order for now
        x_idx = x_idx  # partition-order only for now
        x_cl_bound = None
        x_anchor = None
        x_num = None

        # (x,v)-dependant stuff ---------------------------
        w_x, w_v = cls._infer_kernel_widths(
            eps,
            upsampfac,
            upsampfac_ratio,
        )
        Xd0, Vd0 = cls._grow_to_min_vol(Xd0, Vd0, w_x, w_v)
        Xd = Xd0 + (w_x / Vd0)
        oversampled = sigma_x * dv * Xd <= 1
        Q_over = np.fmax(1, np.floor(1 / (sigma_x * dv * Xd))).astype(np.int64)
        Q_under = np.fmax(1, np.ceil(sigma_x * dv * Xd)).astype(np.int64)
        Xd = np.where(
            oversampled,
            1 / (sigma_x * dv * Q_over),  # oversampled
            Q_under / (sigma_x * dv),  # undersampled
        )
        Ov = np.where(oversampled, Q_over, 1)
        Sv = np.where(oversampled, 1, Q_under)
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
            v0=v0,
            dv=dv,
            Pv=Pv,
            sigma_v=sigma_v,
            alpha_v=alpha_v,
            w_v=w_v,
            Vc=Vc,
            Vd0=Vd0,
            Vd=Vd,
            Ov=Ov,
            Sv=Sv,
        )
        return info

    def _fw_interpolate(self, hFS: np.ndarray) -> np.ndarray:
        r"""
        Interpolate/reduce h^{\fs} onto U points

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
            (..., N1,...,ND) g^{F} in canonical v-order.
        """
        translate = ftk_util.TranslateDType(hFS.dtype)
        fdtype = translate.to_float()
        cdtype = translate.to_complex()

        # reshape/cast (v, hFS)
        sh = hFS.shape[2 : -self.cfg.D]  # (...,)
        Ns = int(np.prod(sh))
        hFS = hFS.reshape(self.cfg.Px, Ns, *self.cfg.L)  # (Px, Ns, L1,...,LD)
        Xc = self.cfg.Xc.astype(fdtype, copy=False)
        Vc = self.cfg.Vc.astype(fdtype, copy=False)[0]  # (D,)
        v0 = self.cfg.v0.astype(fdtype, copy=False)
        dv = self.cfg.dv.astype(fdtype, copy=False)

        # todo: perform interpolation to g0F(Px, Ns, N1,...,ND) ===============
        v = [None] * self.cfg.D
        for d, (_v0, _dv, _N) in enumerate(zip(v0, dv, self.cfg.N)):
            v[d] = (_v0 - Vc[d]) + _dv * np.arange(_N, dtype=fdtype)
        v = np.stack(np.meshgrid(*v, indexing="ij"), axis=-1)  # (N1,...,ND, D)
        v = v.reshape(self.cfg.N.prod(), self.cfg.D)  # (N1*...*N1, D)
        op = ftk_spread.UniformSpread(
            x=v,  # canonical (centered) v-lattice
            z_spec=dict(  # FFS v-lattice
                start=self.cfg.F_v0,
                step=self.cfg.F_dv,
                num=self.cfg.L,
            ),
            phi=self.cfg.phi.low_level_callable(ufunc=False),
            alpha=self.cfg.alpha_v,
        )
        g0F = op.adjoint(hFS)  # (Px, Ns, L1,...,LD) -> (Px, Ns, N1*...*ND)
        g0F = g0F.reshape(self.cfg.Px, Ns, *self.cfg.N)
        # =====================================================================

        # mod/reduce x-partitions
        _mod = [None] * self.cfg.D
        for d, (_v0, _dv, _N) in enumerate(zip(v0, dv, self.cfg.N)):
            _v = _v0 + _dv * np.arange(_N, dtype=fdtype)
            _mod[d] = ftk_complex.cexp((-2 * np.pi) * (Xc[:, [d]] * _v))
        gF = ftk_linalg.hadamard_outer2(g0F, *_mod)  # (Ns, N1,...,ND)
        gF = gF.reshape(*sh, *self.cfg.N)  # (..., N1,...,ND)
        return gF

    def _bw_spread(self, gF: np.ndarray) -> np.ndarray:
        r"""
        Adjoint of _fw_interpolate()

        Parameters
        ----------
        gF: ndarray
            (..., N1,...,ND) g^{F} in canonical v-order.

        Returns
        -------
        hFS: ndarray
            (Pv, Px, ..., L1,...,LD) FS coefficients [w/ FFS padding].
        """
        raise NotImplementedError  # todo

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
            (..., N1,...,ND) g^{\ctft} samples; in canonical v-order.

        Returns
        -------
        z: ndarray
            (..., N1,...,ND) f^{\ctft} samples; in canonical v-order.
        """
        translate = ftk_util.TranslateDType(gF.dtype)
        fdtype = translate.to_float()

        # build \psiF_x samples
        psiF_x = [None] * self.cfg.D
        for d in range(self.cfg.D):
            v0 = self.cfg.v0[d] - self.cfg.Vc[0, d]
            dv = self.cfg.dv[d]
            N = self.cfg.N[d]
            _v = (v0 + dv * np.arange(N)).astype(fdtype)
            ax = self.cfg.alpha_x[d].astype(fdtype)

            psiF_x[d] = self.cfg.phiF(_v / ax) / ax

        psiF_inv = tuple(map(np.reciprocal, psiF_x))
        z = ftk_linalg.hadamard_outer(gF, *psiF_inv)
        return z

    def _re_order_v(self, z: np.ndarray, forward: bool) -> np.ndarray:
        r"""
        Re-order v-domain coordinates from/to canonical order.

        Parameters
        ----------
        z: ndarray
            [forward=True]  (..., N1,...,ND) in canonical v-order.
            [forward=False] (..., N1,...,ND) in user v-order.
        forward: bool

        Returns
        -------
        z2: ndarray
            [forward=True]  (..., N1,...,ND) in user v-order.
            [forward=False] (..., N1,...,ND) in canonical v-order.
        """
        z2 = z  # (canonical v-order == user v-order) for uniform meshes.
        return z2


NU2U = NonUniform2Uniform  # alias
