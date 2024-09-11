import warnings

import numba as nb
import numpy as np

import fourier_toolkit.complex as ftk_complex
import fourier_toolkit.ffs as ftk_ffs
import fourier_toolkit.kernel as ftk_kernel
import fourier_toolkit.linalg as ftk_linalg
import fourier_toolkit.nu2u as ftk_nu2u
import fourier_toolkit.numba as ftk_numba
import fourier_toolkit.pad as ftk_pad
import fourier_toolkit.spread as ftk_spread
import fourier_toolkit.util as ftk_util

__all__ = [
    "NUFFT1",
]

# Disable all warnings for the entire module
warnings.filterwarnings("ignore", category=nb.NumbaExperimentalFeatureWarning)


class NUFFT1(ftk_nu2u.NU2U):
    r"""
    Multi-dimensional NonUniform-to-Uniform Fourier Transform.

    Given the Dirac stream

    .. math::

       f(\bbx) = \sum_{m} w_{m} \delta(\bbx - \bbx_{m}),

    computes samples of :math:`f^{F}`, i.e.,

    .. math::

       f^{F}(\bbv_{n}) = \bbz_{n} = \sum_{m} w_{m} \ee^{ -\cj 2\pi \innerProduct{\bbx_{m}}{\bbv_{n}} },

    where :math:`\bbx_{m}` lie in a :math:`T \in \bR^{D}`-sized interval and :math:`\bbv_{n}` lies on the regular lattice

    .. math::

       \begin{align}
           \bbv_{\bbn} &= \bbv_{0} + \Delta_{\bbv} \odot \bbn, & [\bbn]_{d} \in \{0,\ldots,N_{d}-1\},
       \end{align}

    with :math:`N = \prod_{d} N_{d}` and :math:`\Delta_{\bbv} = 1 / \bbT`.
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
        x: ndarray
            (M, D) support points :math:`\bbx_{m} \in \bbX_{c} + [-\bbT / 2, \bbT / 2]`, where :math:`\bbX_{c} \in \bR^{D}` is an arbitrary constant.
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
            [kernel_param_type=bounded] Kernel stopband relative energy :math:`\epsilon \in ]0, 1[`.
            [kernel_param_type=finufft] Target relative error :math:`\epsilon \in ]0, 1[`.
        upsampfac: tuple[float]
            Total upsampling factor :math:`\sigma > 1`.
        kernel_param_type: str
            How to choose kernel parameters.

            Must be one of:

            * "bounded": ensures eps-bandwidth of \psi_{x} is located in the safe zone.
            * "finufft": uses relations derived in FINUFFT paper.
        kernel_type: str
            Which kernel to use for spreading.

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
        if x.ndim == 1:
            x = x[:, np.newaxis]
        _, D = x.shape
        v_spec["start"] = ftk_util.broadcast_seq(v_spec["start"], D, np.double)
        v_spec["step"] = ftk_util.broadcast_seq(v_spec["step"], D, np.double)
        v_spec["num"] = ftk_util.broadcast_seq(v_spec["num"], D, np.int64)
        assert np.all(v_spec["step"] > 0)
        assert np.all(v_spec["num"] > 0)

        # Are x_m range constraints met, i.e. T >= Xd0
        Xd0_T = np.ptp(x, axis=0) * v_spec["step"]
        assert np.all((Xd0_T < 1) | np.isclose(Xd0_T, 1))

        # validate non-(x,v) inputs
        p = self._validate_inputs(
            D=D,
            isign=isign,
            # Accuracy-related ------------
            eps=eps,
            upsampfac=upsampfac,
            upsampfac_ratio=1e-6,  # unused
            kernel_param_type=kernel_param_type,
            # Runtime behavior ------------
            kernel_type=kernel_type,
            fft_nthreads=fft_nthreads,
            spread_nthreads=spread_nthreads,
            max_cluster_size=max_cluster_size,
            max_window_ratio=max_window_ratio,
            # Chunking behavior (unused) --
            chunked=False,
            domain="x",
            max_bbox_ratio=1.01,
            max_bbox_anisotropy=1,
        )

        self.cfg = self._init_metadata(
            x=x,
            # -------------------------
            v0=v_spec["start"],
            dv=v_spec["step"],
            N=v_spec["num"],
            # -------------------------
            isign=p.isign,
            eps=p.eps,
            upsampfac=p.upsampfac,
            kernel_param_type=p.kernel_param_type,
            kernel_type=p.kernel_type,
            fft_nthreads=p.fft_nthreads,
            spread_nthreads=p.spread_nthreads,
            max_cluster_size=p.max_cluster_size,
            max_window_ratio=p.max_window_ratio,
        )

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
        w = self._flip_sign(w)  # (..., M)
        g = self._fw_spread(w)  # (..., L1,...,LD)

        gFS = self._ffs_transform(  # (..., L1,...,LD)
            g[np.newaxis, np.newaxis],
            forward=True,
        )[0, 0]

        gF = self._fw_interpolate(gFS)  # (..., N1,...,ND)
        z = self._de_convolve(gF)  # (..., N1,...,ND)
        z = self._flip_sign(z)  # (..., N1,...,ND)

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
        z = self._flip_sign(z)  # (..., N1,...,ND)
        gF = self._de_convolve(z)  # (..., N1,...,ND)
        gFS = self._bw_spread(gF)  # (..., L1,...,LD)

        g = self._ffs_transform(  # (..., L1,...,LD)
            gFS[np.newaxis, np.newaxis],
            forward=False,
        )[0, 0]

        w = self._bw_interpolate(g)  # (..., M)
        w = self._flip_sign(w)

        return w

    # Helper routines (internal) ----------------------------------------------
    @classmethod
    def _init_metadata(
        cls,
        x: np.ndarray,
        # -----------------------------
        v0: np.ndarray,
        dv: np.ndarray,
        N: np.ndarray,
        # -----------------------------
        isign: int,
        eps: float,
        upsampfac: np.ndarray,
        kernel_param_type: str,
        kernel_type: str,
        fft_nthreads: int,
        spread_nthreads: int,
        max_cluster_size: int,
        max_window_ratio: np.ndarray,
    ):
        r"""
        Compute all NUFFT1 parameters.

        Returns
        -------
        info: namedtuple
            # general ---------------------------------------------------------
            * D: int                    [Transform Dimensionality]
            * isign: int                [Sign of the exponent]
            * sigma: (D,) float         [Total upsampling factor \sigma]
            * phi: Kernel               [Spread pulse]
            * phiF: Kernel              [Spread pulse FT]
            * beta: float               [\psi_{x}(s) = \phi_{\beta}(\alpha_{x} s)]
            # operators -------------------------------------------------------
            * fft_nthreads: int         [Thread-count for FFTs]
            * u_spread: UniformSpread   [UniformSpread object]
            * pad: Pad                  [Pad object]
            # FFS-related -----------------------------------------------------
            * T: (D,) float             [FFS period]
            * K: (D,) int               [Max FS frequency computed]
            * L: (D,) int               [FFS transform length]
            # x-related -------------------------------------------------------
            * M: int                    [Number of x-domain points]
            * x: (M, D) float           [x-domain support points; user order]
            * alpha_x: (D,) float       [\psi_{x}(s) = \phi_{\beta}(\alpha_{x} s)]
            * w_x: (D,) int             [\psi_{x} support in #samples]
            * Xc: (1, D) float          [x-domain centroid; formatted as such to re-use NU2U methods]
            * Xd0: (D,) float           [x-domain spread X_{d}^{0}]
            # v-related -------------------------------------------------------
            * N: (N1,...,ND) int        [Number of v-domain points]
            * v0: (D,) float            [v-domain lattice starting point \bbv_{0}]
            * dv: (D,) float            [v-domain lattice pitch \Delta_{v}]
            * Vc: (1, D) float          [v-domain centroid; formatted as such to re-use NU2U methods]
            * Vd0: (D,) float           [v-domain spread V_{d}^{0}]
        """
        # As much general stuff as possible ---------------
        D = x.shape[1]
        isign = isign
        sigma = upsampfac
        beta, w_x, _ = cls._infer_kernel_params(
            eps,
            upsampfac,
            0,  # upsampfac_ratio
            kernel_param_type,
        )

        # As much x-domain stuff as possible --------------
        M = len(x)
        x = x
        x_min, x_max = ftk_numba.minmax(x)
        Xc = (x_max + x_min) / 2
        Xc = Xc[np.newaxis]  # (D,) -> (1, D)
        Xd0 = x_max - x_min

        # As much v-domain stuff as possible --------------
        # Setting (Vc, Vd0) as below always guarantees a lattice node lies at the origin in centered coordinates.
        N = N
        v0 = v0
        dv = dv
        Vc = v0 + (dv / 2) * np.where(N % 2 == 1, N - 1, N)
        Vc = Vc[np.newaxis]  # (D,) -> (1, D)
        Vd0 = dv * np.where(N % 2 == 1, N - 1, N)

        # FFS-related stuff -------------------------------
        T = 1 / dv
        K = np.fmax(
            np.ceil(sigma * T * Vd0 / 2).astype(int),
            np.ceil(w_x / 2).astype(int),
        )
        L = ftk_ffs.FFS.next_fast_len(K)

        # kernel stuff ------------------------------------
        alpha_x = (2 * L) / (T * w_x)
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

        # spread/interp object ----------------------------
        x_lattice = ftk_ffs.FFS(T, K, L).sample_points(np.double)
        F_x0 = np.array([_l[0] for _l in x_lattice])
        F_dx = np.array([_l[1] - _l[0] for _l in x_lattice])
        pad_width = np.ceil(w_x / 2).astype(int)
        F_x0 = F_x0 - F_dx * pad_width

        if kernel_type == "kb":
            u_spread_op = ftk_spread.UniformSpread(
                x=x - Xc[0],
                z_spec=dict(start=F_x0, step=F_dx, num=L + 2 * pad_width),
                phi=phi.low_level_callable(ufunc=False),
                alpha=alpha_x,
                inline_kernel=False,
                max_cluster_size=max_cluster_size,
                max_window_ratio=max_window_ratio,
                nthreads=spread_nthreads,
            )
        elif kernel_type == "kb_ppoly":
            u_spread_op = ftk_spread.UniformSpread(
                x=x - Xc[0],
                z_spec=dict(start=F_x0, step=F_dx, num=L + 2 * pad_width),
                phi=phi,
                alpha=alpha_x,
                inline_kernel=True,
                max_cluster_size=max_cluster_size,
                max_window_ratio=max_window_ratio,
                nthreads=spread_nthreads,
            )

        # pad object --------------------------------------
        pad_op = ftk_pad.Pad(
            dim_shape=L,
            pad_width=pad_width,
            mode="wrap",
        )

        info = ftk_util.as_namedtuple(
            # general ------------------
            D=D,
            isign=isign,
            sigma=sigma,
            phi=phi,
            phiF=phiF,
            beta=beta,
            # operators ---------------
            fft_nthreads=fft_nthreads,
            u_spread=u_spread_op,
            pad=pad_op,
            # FFS-related --------------
            T=T,
            K=K,
            L=L,
            # x-related ----------------
            M=M,
            x=x,
            alpha_x=alpha_x,
            w_x=w_x,
            Xc=Xc,
            Xd0=Xd0,
            # v-related ----------------
            N=N,
            v0=v0,
            dv=dv,
            Vc=Vc,
            Vd0=Vd0,
        )
        return info

    def _fw_spread(self, w: np.ndarray) -> np.ndarray:
        r"""
        Sample \tilde{g}(x) on the FFS lattice, where

           \tilde{g}(x) = \sum_{q} g_{0}(x - qT)

        with

           g_{0}(x)
           =
           \sum_{m}
               w_{m}
               \ee^{ -\cj 2\pi V_{c} [ x_{m} - X_{c} ] }
               \psi_{x}( x - [ x_{m} - X_{c} ] )

        Parameters
        ----------
        w: ndarray
            (..., M) in user x-order.

        Returns
        -------
        g: ndarray
            (..., L1,...,LD) \tilde{g} samples on FFS(T,K,L) lattice.
        """
        translate = ftk_util.TranslateDType(w.dtype)
        fdtype = translate.to_float()

        x = ftk_util.cast_warn(self.cfg.x, fdtype)
        Xc = self.cfg.Xc.astype(fdtype, copy=False)[0]  # (D,)
        Vc = self.cfg.Vc.astype(fdtype, copy=False)[0]  # (D,)

        _mod = ftk_complex.cexp((-2 * np.pi) * ((x - Xc) @ Vc))  # (M,)
        _w = w * _mod  # (..., M)

        g0 = self.cfg.u_spread.apply(_w)  # (..., L1+2P1,...,LD+2PD)
        g = self.cfg.pad.adjoint(g0)  # (..., L1,...,LD)
        return g

    def _bw_interpolate(self, g: np.ndarray) -> np.ndarray:
        r"""
        Adjoint of _fw_spread()

        Parameters
        ----------
        g: ndarray
            (..., L1,...,LD) \tilde{g} samples on FFS(T,K,L) lattice.

        Returns
        -------
        w: ndarray
            (..., M) in user x-order.
        """
        translate = ftk_util.TranslateDType(g.dtype)
        fdtype = translate.to_float()

        x = ftk_util.cast_warn(self.cfg.x, fdtype)
        Xc = self.cfg.Xc.astype(fdtype, copy=False)[0]  # (D,)
        Vc = self.cfg.Vc.astype(fdtype, copy=False)[0]  # (D,)

        g0 = self.cfg.pad.apply(g)  # (..., L1+2P1,...,LD+2PD)
        w = self.cfg.u_spread.adjoint(g0)  # (..., M)

        _mod = ftk_complex.cexp((2 * np.pi) * ((x - Xc) @ Vc))  # (M,)
        w *= _mod
        return w

    def _fw_interpolate(self, gFS: np.ndarray) -> np.ndarray:
        r"""
        "Interpolate" g^{\fs} onto U points

            g^{\ctft}(v_{n})
            =
            \ee^{ -\cj 2\pi X_{c} v_{n} }
            T gFS_{n - K^{\prime} / \sigma}

        Parameters
        ----------
        gFS: ndarray
            (..., L1,...,LD) FS coefficients [from FFS w/ padding].

        Returns
        -------
        gF: ndarray
            (..., N1,...,ND) g^{F} in canonical v-order.
        """
        translate = ftk_util.TranslateDType(gFS.dtype)
        fdtype = translate.to_float()

        Xc = self.cfg.Xc.astype(fdtype, copy=False)[0]  # (D,)
        Vc = self.cfg.Vc.astype(fdtype, copy=False)[0]  # (D,)

        mod = [None] * self.cfg.D  # (N1,),...,(ND,)
        for d in range(self.cfg.D):
            _v = np.arange(self.cfg.N[d], dtype=fdtype)
            _v *= self.cfg.dv[d]
            _v += self.cfg.v0[d]

            mod[d] = ftk_complex.cexp((-2 * np.pi) * (Xc[d] * _v))

        # trim oversampled sections
        select = [None] * self.cfg.D
        for d in range(self.cfg.D):
            K0 = int((Vc[d] - self.cfg.v0[d]) * self.cfg.T[d])
            select[d] = slice(
                self.cfg.K[d] - K0,
                self.cfg.K[d] - K0 + self.cfg.N[d],
            )
        g0F = gFS[..., *select]  # (..., N1,...,ND)

        # re-scale v-domain
        g0F *= self.cfg.T.prod()

        # mod v-domain
        gF = ftk_linalg.hadamard_outer(g0F, *mod)
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
        gFS: ndarray
            (..., L1,...,LD) FS coefficients [w/ FFS padding].
        """
        translate = ftk_util.TranslateDType(gF.dtype)
        fdtype = translate.to_float()

        sh = gF.shape[: -self.cfg.D]
        Xc = self.cfg.Xc.astype(fdtype, copy=False)[0]  # (D,)
        Vc = self.cfg.Vc.astype(fdtype, copy=False)[0]  # (D,)

        mod = [None] * self.cfg.D  # (N1,),...,(ND,)
        for d in range(self.cfg.D):
            _v = np.arange(self.cfg.N[d], dtype=fdtype)
            _v *= self.cfg.dv[d]
            _v += self.cfg.v0[d]

            mod[d] = ftk_complex.cexp((2 * np.pi) * (Xc[d] * _v))

        # mod v-domain
        g0F = ftk_linalg.hadamard_outer(gF, *mod)

        # re-scale v-domain
        g0F *= self.cfg.T.prod()

        # pad to oversampled
        gFS = np.zeros((*sh, *self.cfg.L), dtype=g0F.dtype)
        select = [None] * self.cfg.D
        for d in range(self.cfg.D):
            K0 = int((Vc[d] - self.cfg.v0[d]) * self.cfg.T[d])
            K = self.cfg.K[d]
            N = self.cfg.N[d]

            select[d] = slice(K - K0, K - K0 + N)
        gFS[..., *select] = g0F
        return gFS
