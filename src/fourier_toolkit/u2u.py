import os

import numpy as np

import fourier_toolkit.czt as ftk_czt
import fourier_toolkit.linalg as ftk_linalg
import fourier_toolkit.util as ftk_util

__all__ = [
    "Uniform2Uniform",
    "U2U",
]


class Uniform2Uniform:
    r"""
    Multi-dimensional Uniform-to-Uniform Fourier Transform.

    Given the Dirac stream

    .. math::

       f(\bbx) = \sum_{m} w_{m} \delta(\bbx - \bbx_{m}),

    computes samples of :math:`f^{F}`, i.e.,

    .. math::

       f^{F}(\bbv_{n}) = \bbz_{n} = \sum_{m} w_{m} \ee^{ -\cj 2\pi \innerProduct{\bbx_{m}}{\bbv_{n}} },

    where :math:`(\bbx_{m}, \bbv_{n})` lie on the regular lattice

    .. math::

       \begin{align}
           \bbx_{\bbm} &= \bbx_{0} + \Delta_{\bbx} \odot \bbm, & [\bbm]_{d} \in \{0,\ldots,M_{d}-1\}, \\
           \bbv_{\bbn} &= \bbv_{0} + \Delta_{\bbv} \odot \bbn, & [\bbn]_{d} \in \{0,\ldots,N_{d}-1\},
       \end{align}

    with :math:`M = \prod_{d} M_{d}` and :math:`N = \prod_{d} N_{d}`.
    """

    def __init__(
        self,
        x_spec: dict[str, np.ndarray],
        v_spec: dict[str, np.ndarray],
        *,
        isign: int = -1,
        nthreads: int = 0,
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
        v_spec: dict[str, ndarray]
            :math:`\bbv_{n}` lattice specifier, with keys:

            * `start`: (D,) values :math:`\bbv_{0} \in \bR^{D}`.
            * `step` : (D,) values :math:`\Delta_{\bbv} \in \bR^{D}`.
            * `num`  : (D,) values :math:`\{ N_{1},\ldots,N_{D} \} \in \bN^{D}`.
              This parameter fixes the dimensionality `D` of the transform.

            Scalars are broadcasted to all dimensions.
        isign: +1, -1
            Sign of the exponent.
        nthreads: int
            Number of threads to use. If 0, use all cores.
        """
        # Validate parameters =================================================
        x_spec["num"] = ftk_util.broadcast_seq(x_spec["num"], None, np.int64)
        Dx = len(x_spec["num"])
        x_spec["start"] = ftk_util.broadcast_seq(x_spec["start"], Dx, np.double)
        x_spec["step"] = ftk_util.broadcast_seq(x_spec["step"], Dx, np.double)
        assert np.all(x_spec["step"] > 0)
        assert np.all(x_spec["num"] > 0)

        v_spec["num"] = ftk_util.broadcast_seq(v_spec["num"], None, np.int64)
        Dv = len(v_spec["num"])
        v_spec["start"] = ftk_util.broadcast_seq(v_spec["start"], Dv, np.double)
        v_spec["step"] = ftk_util.broadcast_seq(v_spec["step"], Dv, np.double)
        assert np.all(v_spec["step"] > 0)
        assert np.all(v_spec["num"] > 0)

        assert Dx == Dv
        if nthreads == 0:
            nthreads = os.cpu_count()
        assert 1 <= nthreads <= os.cpu_count()
        # =====================================================================

        self.cfg = ftk_util.as_namedtuple(
            D=Dx,
            # -------------------------
            x0=x_spec["start"],
            dx=x_spec["step"],
            M=x_spec["num"],
            # -------------------------
            v0=v_spec["start"],
            dv=v_spec["step"],
            N=v_spec["num"],
            # -------------------------
            isign=int(isign / abs(isign)),
            # -------------------------
            nthreads=int(nthreads),
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
            (..., N1,...,ND) weights :math:`z_{n} \in \bC^{D}`.
        """
        czt, B = self._params(w.dtype)

        _w = czt.apply(w)
        z = ftk_linalg.hadamard_outer(_w, *B)
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
            (..., M1,...,MD) weights :math:`w_{m} \in \bC^{D}`.
        """
        czt, B = self._params(z.dtype)
        B = [_B.conj() for _B in B]

        _z = ftk_linalg.hadamard_outer(z, *B)
        w = czt.adjoint(_z)
        return w

    # Helper routines (internal) ----------------------------------------------
    def _params(self, dtype: np.dtype):
        """
        Parameters
        ----------
        dtype: float/complex

        Returns
        -------
        czt: CZT
            CZT(A,W,M,N) instance.
        B: ndarray
            (N1,),...,(ND,) post-CZT modulation vectors.
        """
        translate = ftk_util.TranslateDType(dtype)
        cdtype = translate.to_complex()

        # Build CZT operator
        A = np.exp(-1j * self.cfg.isign * 2 * np.pi * self.cfg.dx * self.cfg.v0)
        W = np.exp(+1j * self.cfg.isign * 2 * np.pi * self.cfg.dx * self.cfg.dv)
        czt = ftk_czt.CZT(
            N=self.cfg.M,
            M=self.cfg.N,
            A=A,
            W=W,
            nthreads=self.cfg.nthreads,
        )

        # Build modulation vector (B,).
        B = [None] * self.cfg.D
        for d in range(self.cfg.D):
            phase_scale = self.cfg.isign * 2 * np.pi * self.cfg.x0[d]
            v = self.cfg.v0[d] + self.cfg.dv[d] * np.arange(self.cfg.N[d])
            _B = np.exp(1j * phase_scale * v)

            B[d] = _B.astype(cdtype)

        return czt, B


U2U = Uniform2Uniform  # alias
