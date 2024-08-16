import concurrent.futures as cf
import os
import pathlib as plib
import warnings

import numba as nb
import numpy as np

import fourier_toolkit.cluster as ftk_cluster
import fourier_toolkit.kernel as ftk_kernel
import fourier_toolkit.numba as ftk_numba
import fourier_toolkit.util as ftk_util
from fourier_toolkit.config import generate_module

__all__ = [
    "UniformSpread",
]


# Disable all warnings for the entire module
warnings.filterwarnings("ignore", category=nb.NumbaExperimentalFeatureWarning)


class UniformSpread:
    r"""
    Multi-dimensional convolution sampled on uniform lattice.

    Given the Dirac stream

    .. math::

       f(\bbz) = \sum_{m} w_{m} \delta(\bbz - \bbx_{m}),

    computes samples of :math:`g = f \conv \psi`, i.e.,

    .. math::

       g(\bbz_{n}) = \sum_{m} w_{m} \psi(\bbz_{n} - \bbx_{m}),

    where :math:`\bbx_{m} \in \bR^{D}` and :math:`\bbz_{n}` lies on the regular lattice

    .. math::

       \begin{align}
           \bbz_{n} &= \bbz_{0} + \Delta_{\bbz} \odot \bbn, & [\bbn]_{d} \in \{0,\ldots,N_{d}-1\},
       \end{align}

    with :math:`N = \prod_{d} N_{d}`.

    The seperable kernel :math:`\psi` takes the form

    .. math::

       \psi(\bbx) &= \prod_{d} \phi_{d}(\alpha_{d} x_{d}),

    where :math:`\phi_{d}: [-1, 1] \to \bR` and :math:`\alpha_{d} > 0`.
    """

    template_path = {  # inline_kernel -> template_file
        True: plib.Path(__file__).parent / "_template_inline.txt",
        False: plib.Path(__file__).parent / "_template_dynamic.txt",
    }

    def __init__(
        self,
        x: np.ndarray,
        z_spec: dict[str, np.ndarray],
        phi: tuple[callable],
        alpha: tuple[float],
        *,
        max_cluster_size: int = 10_000,
        max_window_ratio: tuple[float] = 10,
        nthreads: int = 0,
        inline_kernel: bool = False,
    ):
        r"""
        Parameters
        ----------
        x: ndarray[float]
            (M, D) support points :math:`\bbx_{m} \in \bR^{D}`.
        z_spec: dict[str, ndarray]
            :math:`\bbz_{n}` lattice specifier, with keys:

            * `start`: (D,) values :math:`\bbz_{0} \bR^{D}`.
            * `step` : (D,) values :math:`\Delta_{\bbz} \in \bR^{D}`.
            * `num`  : (D,) values :math:`\{ N_{1},\ldots,N_{D} \} \in \bN^{D}`.

            Scalars are broadcasted to all dimensions.
        phi: tuple[callable]
            (D,) kernels :math:`\{ \phi_{1},\ldots,\phi_{D} \}`.


            If `inline_kernel` is False (default), then functions must be Numba-compiled funcs with the following signatures:

            * float32(float32)
            * float64(float64)

            If `inline_kernel` is True, then `phi` must be a ``PPoly`` instance. (See `inline_kernel`.)
        alpha: tuple[float]
            (D,) kernel scale factors :math:`\{ \alpha_{1},\ldots,\alpha_{D} \}`.
        max_cluster_size: int
            Maximum number of support points per sub-grid.
        max_window_ratio: tuple[float]
            (D,) maximum sub-grid support

            .. math::

               \{
                   \frac{2}{\alpha_{1}} r_{1},
                   \ldots,
                   \frac{2}{\alpha_{D}} r_{D},
               \}

            where :math:`r_{d} > 1` grows the window to some size larger than :math:`\psi_{d}`'s support :math:`\frac{2}\alpha_{d}}`.

            `max_window_ratio` defines the quantities :math:`\{ r_{1},\ldots,r_{D} \}`.
        nthreads: int
            Number of threads used to spread sub-grids. If 0, use all cores.
        inline_kernel: bool
            Inline kernel functions.

            This only works if `phi` is a ``PPoly`` instance shared across all dimensions.
            An error is raised if inlining fails.

        Notes
        -----
        Some guidelines to set these parameters:

        * `max_window_ratio` determines the maximum memory requirements per sub-grid.
        * `nthreads` sub-grids are processed in parallel.
          Due to the Python GIL, the speedup is not linear with the number of workers.
          Choosing a small value (ex 2-4) seems to offer the best parallel efficiency.
        * `max_cluster_size` should be chosen large enough for there to be meaningful work done by each thread.
          If chosen too small, then many sub-grids need to be written to the global grid, which may introduce
          overheads.
        * `max_window_ratio` should be chosen based on the point distribution.
          Set it to `inf` if only cluster size matters.
        """
        assert x.ndim in (1, 2)
        if x.ndim == 1:
            x = x[:, np.newaxis]
        M, D = x.shape

        z_spec["start"] = ftk_util.broadcast_seq(z_spec["start"], D, np.double)
        z_spec["step"] = ftk_util.broadcast_seq(z_spec["step"], D, np.double)
        z_spec["num"] = ftk_util.broadcast_seq(z_spec["num"], D, np.int64)
        assert np.all(z_spec["step"] > 0)
        assert np.all(z_spec["num"] > 0)

        phi = tuple(ftk_util.broadcast_seq(phi, D, None))
        assert all(map(callable, phi))
        if inline_kernel:
            # Expecting PPoly(support=1) instance shared among dimensions
            for d in range(D):
                assert isinstance(phi[d], ftk_kernel.PPoly)
                assert phi[d] is phi[0]
                assert np.isclose(phi[d].support(), 1)

            phi_low_level = (phi[0].low_level_callable(ufunc=False),) * D
        else:
            phi_low_level = phi

        alpha = ftk_util.broadcast_seq(alpha, D, np.double)
        assert np.all(alpha > 0)

        max_window_ratio = ftk_util.broadcast_seq(max_window_ratio, D, np.double)
        assert max_cluster_size > 0
        assert np.all(max_window_ratio >= 2)
        if nthreads == 0:
            nthreads = os.cpu_count()
        assert 1 <= nthreads <= os.cpu_count()

        cl_info = self._cluster_info(
            x=x,
            z0=z_spec["start"],
            dz=z_spec["step"],
            N=z_spec["num"],
            alpha=alpha,
            max_cluster_size=max_cluster_size,
            max_window_ratio=max_window_ratio,
        )
        self.cfg = ftk_util.as_namedtuple(
            x=x,
            M=M,
            D=D,
            # -------------------------
            z0=z_spec["start"],
            dz=z_spec["step"],
            N=z_spec["num"],
            # -------------------------
            phi=phi_low_level,
            alpha=alpha,
            # -------------------------
            max_cluster_size=max_cluster_size,
            max_window_ratio=max_window_ratio,
            nthreads=int(nthreads),
            # -------------------------
            **cl_info._asdict(),
        )

        subs_kwargs = dict(
            z_rank=D,
            z_shape=", ".join([f"len(z[{d}])" for d in range(D)]),
            support=", ".join([f"z_ub[{d}] - z_lb[{d}] + 1" for d in range(D)]),
            idx=", ".join([f"z_lb[{d}] + offset[{d}]" for d in range(D)]),
            range_D=tuple(range(D)),
            range_Dp1=tuple(range(1, D + 1)),
        )
        if inline_kernel:  # add extra substitutions
            _phi = phi[0]
            subs_kwargs.update(
                ppoly_sym=_phi._sym,
                ppoly_bin_count=_phi._weight.shape[0],
                ppoly_order=_phi._weight.shape[1] - 1,
                ppoly_pitch_rcp=1 / _phi._pitch,
                ppoly_support=_phi.support(),
                ppoly_weight=_phi._print_weights(),
            )

        pkg = generate_module(
            path=self.template_path[inline_kernel],
            subs=subs_kwargs,
        )
        self._spread = pkg.spread
        self._interpolate = pkg.interpolate

    def apply(self, w: np.ndarray) -> np.ndarray:
        r"""
        Parameters
        ----------
        w: ndarray[float/complex]
            (..., M) weights :math:`w_{m} \in \bC`.

        Returns
        -------
        g: ndarray[float/complex]
            (..., N1,...,ND) samples :math:`g(\bbz_{n}) \in \bC`.
        """
        translate = ftk_util.TranslateDType(w.dtype)
        fdtype = translate.to_float()

        # re-order/shape (x, w)
        sh = w.shape[:-1]  # (...,)
        Ns = int(np.prod(sh))
        x = ftk_util.cast_warn(self.cfg.x[self.cfg.x_idx], fdtype)
        w = w[..., self.cfg.x_idx].reshape(Ns, -1)  # (Ns, M)

        # lattice extractor
        z = self._lattice(fdtype)  # (N1,),...,(ND,)
        roi = lambda q: tuple(  # int -> tuple[slice]
            slice(n0, n0 + num)
            for (n0, num) in zip(
                self.cfg.z_anchor[q],
                self.cfg.z_num[q],
            )
        )
        z_roi = lambda q: tuple(  # int -> tuple[ndarray]
            _z[_roi]
            for (_z, _roi) in zip(
                z,
                roi(q),
            )
        )

        # kernel scale
        k_scale = self.cfg.alpha.astype(fdtype)  # (D,)

        # spread each cluster onto its own sub-grid
        Q = len(self.cfg.cl_bound) - 1
        fs, fs2cl = [None] * Q, dict()
        with cf.ThreadPoolExecutor(max_workers=self.cfg.nthreads) as executor:
            for q in range(Q):
                a, b = self.cfg.cl_bound[q : q + 2]

                future = executor.submit(
                    self._spread,
                    x=x[a:b, :],
                    w=w[:, a:b],
                    z=z_roi(q),
                    phi=self.cfg.phi,
                    a=k_scale,
                )
                fs[q], fs2cl[future] = future, q

        # update global grid
        g = np.zeros((*sh, *self.cfg.N), dtype=w.dtype)
        for future in cf.as_completed(fs):
            q = fs2cl[future]
            g_roi = future.result()

            g[..., *roi(q)] += g_roi.reshape(*sh, *self.cfg.z_num[q])
        return g

    def adjoint(self, g: np.ndarray) -> np.ndarray:
        r"""
        Parameters
        ----------
        g: ndarray[float/complex]
            (..., N1,...,ND) samples :math:`g(\bbz_{n}) \in \bC`.

        Returns
        -------
        w: ndarray[float/complex]
            (..., M) weights :math:`w_{m} \in \bC`.
        """
        translate = ftk_util.TranslateDType(g.dtype)
        fdtype = translate.to_float()

        # re-order/shape (x, g)
        sh = g.shape[: -self.cfg.D]  # (...,)
        Ns = int(np.prod(sh))
        x = ftk_util.cast_warn(self.cfg.x[self.cfg.x_idx], fdtype)
        g = g.reshape(Ns, *self.cfg.N)  # (Ns, N1,...,ND)

        # lattice extractor
        z = self._lattice(fdtype)  # (N1,),...,(ND,)
        roi = lambda q: tuple(  # int -> tuple[slice]
            slice(n0, n0 + num)
            for (n0, num) in zip(
                self.cfg.z_anchor[q],
                self.cfg.z_num[q],
            )
        )
        z_roi = lambda q: tuple(  # int -> tuple[ndarray]
            _z[_roi]
            for (_z, _roi) in zip(
                z,
                roi(q),
            )
        )

        # kernel scale
        k_scale = self.cfg.alpha.astype(fdtype)  # (D,)

        # interpolate each sub-grid onto support points within
        Q = len(self.cfg.cl_bound) - 1
        fs, fs2cl = [None] * Q, dict()
        with cf.ThreadPoolExecutor(max_workers=self.cfg.nthreads) as executor:
            for q in range(Q):
                a, b = self.cfg.cl_bound[q : q + 2]

                future = executor.submit(
                    self._interpolate,
                    x=x[a:b, :],
                    g=g[:, *roi(q)],
                    z=z_roi(q),
                    phi=self.cfg.phi,
                    a=k_scale,
                )
                fs[q], fs2cl[future] = future, q

        # update global support
        w = np.zeros((*sh, self.cfg.M), dtype=g.dtype)
        for future in cf.as_completed(fs):
            q = fs2cl[future]
            w_q = future.result()

            a, b = self.cfg.cl_bound[q : q + 2]
            idx = self.cfg.x_idx[a:b]
            w[..., idx] = w_q.reshape(*sh, b - a)
        return w

    # Helper routines (internal) ----------------------------------------------
    def _lattice(self, dtype: np.dtype) -> tuple[np.ndarray]:
        r"""
        Parameters
        ----------
        dtype: float/complex

        Returns
        -------
        z: tuple[ndarray]
            (N1,),...,(ND,) lattice coordinates :math:`\bbz_{n} \in \bR^{D}`.
        """
        translate = ftk_util.TranslateDType(dtype)
        fdtype = translate.to_float()

        z = [None] * self.cfg.D
        for d in range(self.cfg.D):
            _z = self.cfg.z0[d] + self.cfg.dz[d] * np.arange(self.cfg.N[d])

            z[d] = _z.astype(fdtype)
        return z

    @staticmethod
    def _cluster_info(
        x: np.ndarray,
        z0: np.ndarray,
        dz: np.ndarray,
        N: np.ndarray,
        alpha: np.ndarray,
        max_cluster_size: int,
        max_window_ratio: np.ndarray,
    ):
        """
        Build acceleration metadata.

        * Partitions the support points into Q clusters.
        * Identifies the sub-grids onto which each cluster is spread.

        See __init__() for parameter descriptions.

        Returns
        -------
        info: namedtuple
            (Q,) cluster metadata, with fields:

            * x_idx: ndarray[int]
                (M,) indices to re-order `x` s.t. points in each cluster are sequential.
                Its length may be smaller than M if points do not contribute to the lattice in any way.
            * cl_bound: ndarray[int]
                (Q+1,) indices into `x_idx` indicating where the q-th cluster's support points start/end.
                Cluster `q` contains support points `x[x_idx][cl_bound[q] : cl_bound[q+1]]`.
            * z_anchor: ndarray[int]
                (Q, D) lower-left coordinate of each sub-grid w.r.t. the global grid.
            * z_num: ndarray[int]
                (Q, D) sub-grid sizes.
        """
        idtype, fdtype = np.int64, x.dtype
        s = 1 / alpha  # kernel one-sided support.

        # Restrict clustering to support points which contribute to the lattice.
        active = ftk_numba.filter_to_bbox(  # (M,)
            x,
            z0.astype(fdtype, copy=False),
            (z0 + dz * N).astype(fdtype, copy=False),
            s.astype(fdtype, copy=False),
        )
        active2global = np.flatnonzero(active)
        x = x[active]

        # Quick exit if no support points.
        _, D = x.shape
        if len(x) == 0:
            zeros = lambda sh: np.zeros(sh, dtype=idtype)
            info = ftk_util.as_namedtuple(
                x_idx=zeros(0),
                cl_bound=zeros(1),
                z_anchor=zeros((0, D)),
                z_num=zeros((0, D)),
            )
        else:
            # Group support points into clusters to match max window size.
            bbox_dim = (2 * s) * max_window_ratio
            x_idx, cl_info = ftk_cluster.grid_cluster(x, bbox_dim)
            x_idx, cl_info = ftk_cluster.fuse_cluster(x, x_idx, cl_info, bbox_dim)

            # Split clusters to match max cluster size.
            cl_info = ftk_cluster.bisect_cluster(cl_info, max_cluster_size)

            # Compute off-grid lattice boundaries after spreading.
            cl_min, cl_max = ftk_numba.group_minmax(x, x_idx, cl_info)  # (Q, D)
            LL = cl_min - s  # lower-left lattice coordinate
            UR = cl_max + s  # upper-right lattice coordinate

            # Get gridded equivalents.
            LL_idx = np.floor((LL - z0) / dz)
            UR_idx = np.ceil((UR - z0) / dz)

            # Clip LL/UR to lattice boundaries.
            LL_idx = np.fmax(0, LL_idx).astype(idtype)
            UR_idx = np.fmin(UR_idx, N - 1).astype(idtype)

            info = ftk_util.as_namedtuple(
                x_idx=active2global[x_idx],  # indices w.r.t input `x`
                cl_bound=cl_info,  # (Q+1,)
                z_anchor=LL_idx,
                z_num=UR_idx - LL_idx + 1,
            )
        return info
