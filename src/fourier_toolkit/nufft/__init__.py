import math
from typing import Literal

import array_api_compat as aac

import fourier_toolkit.typing as ftkt
import fourier_toolkit.util as ftku

__all__ = [
    "nu2nu",
    "nu2u",
    "u2nu",
]

_eps_default: float = 1e-6
ExponentSign = Literal[-1, +1]
_eps_fp32_min = 1e-6
_eps_fp64_min = 1e-14


def nu2nu(
    x: ftkt.ArrayR,
    v: ftkt.ArrayR,
    w: ftkt.ArrayRC,
    isign: ExponentSign,
    eps: float = _eps_default,
    **kwargs,
) -> ftkt.ArrayC:
    r"""
    Multi-dimensional NonUniform-to-NonUniform Fourier Transform. (:math:`\tnunu`)

    Computes the Fourier sum

    .. math::

       \bbz_{n} = \sum_{m} w_{m} \ee^{ \pm \cj 2\pi \innerProduct{\bbx_{m}}{\bbv_{n}} },

    where :math:`(\bbx_{m}, \bbv_{n}) \in \bR^{D}`.

    Parameters
    ----------
    x: ArrayR
        (M, D) knots :math:`\bbx_{m} \in \bR^{D}`.
    v: ArrayR
        (N, D) knots :math:`\bbv_{n} \in \bR^{D}`.
    w: ArrayRC
        (..., M) weights :math:`w_{m} \in \bC`.
    isign: +1, -1
        Exponent sign.
    eps: float
        Target relative error :math:`\epsilon \in ]0, 1[`.
    kwargs: dict
        Extra parameters for :py:class:`finufft.Plan`.
        (For advanced users only.)

    Returns
    -------
    z: ArrayC
        (..., N) weights :math:`z_{n} \in \bC`.

    Notes
    -----
    * (x,v,w) must share the same (backend,device).

    * The :math:`\tnunu` transform will be computed at the least-precise dtype of (x,v,w).
    """
    fdtype, cdtype = _determine_precision(x, v, w)

    x = _canonicalize_knots(x, fdtype)
    v = _canonicalize_knots(v, fdtype)
    M, Dx = x.shape
    N, Dv = v.shape
    assert (D := Dx) == Dv
    assert D in (1, 2, 3)  # FINUFFT limitation
    w, sh = _canonicalize_weights(w, cdtype, M)

    assert (isign := int(isign)) in (-1, +1)

    assert 0 < (eps := float(eps)) < 1
    xp = aac.array_namespace(x)
    if cdtype == xp.complex64:
        eps = max(eps, _eps_fp32_min)
    else:  # xp.complex128
        eps = max(eps, _eps_fp64_min)

    if aac.is_array_api_obj(w):
        from . import generic as namespace
    else:
        raise NotImplementedError

    z = xp.reshape(
        namespace._nu2nu(x, v, w, isign, eps, **kwargs),
        (*sh, N),
    )
    return z


def nu2u(
    x: ftkt.ArrayR,
    v_spec: ftku.UniformSpec,
    w: ftkt.ArrayRC,
    isign: ExponentSign,
    eps: float = _eps_default,
    **kwargs,
) -> ftkt.ArrayC:
    r"""
    Multi-dimensional NonUniform-to-Uniform Fourier Transform. (:math:`\tnuu`)

    Computes the Fourier sum

    .. math::

       \bbz_{n} = \sum_{m} w_{m} \ee^{ \pm \cj 2\pi \innerProduct{\bbx_{m}}{\bbv_{n}} },

    where :math:`\bbx_{m} \in \bR^{D}`, and :math:`\bbv_{n}` lies on the regular lattice

    .. math::

       \bbv_{\bbn} = \bbv_{0} + \Delta_{\bbv} \odot \bbn,
       \qquad
       [\bbn]_{d} \in \discreteRange{0}{N_{d}-1},

    with :math:`N = \prod_{d} N_{d}`.

    Parameters
    ----------
    x: ArrayR
        (M, D) knots :math:`\bbx_{m} \in \bR^{D}`.
    v_spec: UniformSpec
        :math:`\bbv_{n}` lattice.
    w: ArrayRC
        (..., M) weights :math:`w_{m} \in \bC`.
    isign: +1, -1
        Exponent sign.
    eps: float
        Target relative error :math:`\epsilon \in ]0, 1[`.
    kwargs: dict
        Extra parameters for :py:class:`finufft.Plan`.
        (For advanced users only.)

    Returns
    -------
    z: ArrayC
        (..., N1,...,ND) weights :math:`z_{n} \in \bC`.

    Notes
    -----
    * (x,w) must share the same (backend,device).

    * (x,w) must have the same numerical precision:

      - (x,) float32, (w,) complex64
      - (x,) float64, (w,) complex128
    """
    raise NotImplementedError


def u2nu(
    x_spec: ftku.UniformSpec,
    v: ftkt.ArrayR,
    w: ftkt.ArrayRC,
    isign: ExponentSign,
    eps: float = _eps_default,
    **kwargs,
) -> ftkt.ArrayC:
    r"""
    Multi-dimensional Uniform-to-NonUniform Fourier Transform. (:math:`\tunu`)

    Computes the Fourier sum

    .. math::

       \bbz_{n} = \sum_{m} w_{m} \ee^{ \pm \cj 2\pi \innerProduct{\bbx_{m}}{\bbv_{n}} },

    where :math:`\bbv_{n} \in \bR^{D}`, and :math:`\bbx_{m}` lies on the regular lattice

    .. math::

       \bbx_{\bbm} = \bbx_{0} + \Delta_{\bbx} \odot \bbm,
       \qquad
       [\bbm]_{d} \in \discreteRange{0}{M_{d}-1},

    with :math:`M = \prod_{d} M_{d}`.

    Parameters
    ----------
    x_spec: UniformSpec
        :math:`\bbx_{m}` lattice.
    v: ArrayR
        (N, D) knots :math:`\bbv_{n} \in \bR^{D}`.
    w: ArrayRC
        (..., M1,...,MD) weights :math:`w_{m} \in \bC`.
    eps: float
        Target relative error :math:`\epsilon \in ]0, 1[`.
    isign: +1, -1
        Exponent sign.
    kwargs: dict
        Extra parameters for :py:class:`finufft.Plan`.
        (For advanced users only.)

    Returns
    -------
    z: ArrayC
        (..., N) weights :math:`z_{n} \in \bC`.

    Notes
    -----
    * (v,w) must share the same (backend,device).

    * (v,w) must have the same numerical precision:

      - (v,) float32, (w,) complex64
      - (v,) float64, (w,) complex128
    """
    raise NotImplementedError


# Helper routines (internal) ---------------------------------------------------
def _determine_precision(*args: tuple[ftkt.ArrayRC]) -> tuple[ftkt.DType, ftkt.DType]:
    """
    Determine the least precise floating/complex dtype among arrays.

    Returns
    -------
    fdtype, cdtype: DType
        floating/complex dtype
    """
    xp = aac.array_namespace(args[0])
    device = args[0].device
    for a in args:
        assert aac.array_namespace(a) == xp  # same backend
        assert a.device == device  # same device
        assert xp.finfo(a).bits in (32, 64)  # single/double precision (indirect check)

    n_bits = min(xp.finfo(a).bits for a in args)
    fdtype = getattr(xp, f"float{n_bits}")
    cdtype = getattr(xp, f"complex{2 * n_bits}")
    return fdtype, cdtype


def _canonicalize_knots(xv: ftkt.ArrayR, fdtype: ftkt.DType) -> ftkt.ArrayR:
    """
    Transform (x, v) knots to canonical form:
    - shape (M|N, D)
    - real-valued

    Parameters
    ----------
    xv: ArrayR
        (M|N, [D])

    Returns
    -------
    xv_c: ArrayR
        Canonical form of `xv`.
    """
    xp = aac.array_namespace(xv)

    assert xv.ndim in (1, 2)
    if xv.ndim == 1:
        xv = xp.reshape(xv, (-1, 1))

    xv_c = xp.asarray(xv, dtype=fdtype)
    return xv_c


def _canonicalize_weights(
    w: ftkt.ArrayRC,
    cdtype: ftkt.DType,
    M: int,
) -> tuple[ftkt.ArrayC, tuple]:
    """
    Transform (w,) weights to canonical form:
    - shape (n_trans, M)
    - complex-valued

    Parameters
    ----------
    w: ArrayRC
        (..., M)

    Returns
    -------
    w_c: ArrayC
        Canonical form of `w`.
    sh: tuple[int]
        Shape to transform the output of `plan.execute()` to.
        This is to respect the input's stack shape (...).
    """
    *sh, Mw = w.shape
    sh_flat = math.prod(sh)
    assert Mw == M

    xp = aac.array_namespace(w)
    w_c = xp.reshape(
        xp.asarray(w, dtype=cdtype),
        (sh_flat, M),
    )
    return w_c, sh
