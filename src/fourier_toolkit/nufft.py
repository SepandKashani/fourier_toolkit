import math
from typing import Literal

import finufft
import numpy as np

import fourier_toolkit.typing as ftkt
import fourier_toolkit.util as ftku

__all__ = [
    "nu2nu",
    "nu2u",
    "u2nu",
]

_eps_default: float = 1e-6
ExponentSign = Literal[-1, +1]


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
        (M, D) support points :math:`\bbx_{m} \in \bR^{D}`.
    v: ArrayR
        (N, D) frequencies :math:`\bbv_{n} \in \bR^{D}`.
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
    (x,v,w) must have the same numerical precision:

    - (x,v) float32, (w,) complex64
    - (x,v) float64, (w,) complex128
    """
    x = _canonicalize_knots(x)
    v = _canonicalize_knots(v)
    assert type(x) is type(v)  # common backend
    assert x.dtype == v.dtype  # common precision
    M, Dx = x.shape
    N, Dv = v.shape
    assert (D := Dx) == Dv
    assert D in (1, 2, 3)  # FINUFFT limitation

    w, sh = _canonicalize_weights(w, M)
    assert type(x) is type(w)  # common backend
    fdtype = ftku.TranslateDType(w.dtype).to_float()
    assert fdtype == x.dtype  # common precision

    assert 0 < (eps := float(eps)) < 1
    assert (isign := int(isign)) in (-1, +1)
    plan = _create_plan(x, v, w, isign, eps, **kwargs)

    z = plan.execute(w).reshape((*sh, N))
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
        (M, D) support points :math:`\bbx_{m} \in \bR^{D}`.
    v_spec: UniformSpec
        :math:`\bbv_{n}` lattice specifier.
    w: ArrayRC
        (..., M) weights :math:`w_{m} \in \bC`.
    isign: +1, -1
        Exponent sign.
    eps: float
        Target relative error :math:`\epsilon \in ]0, 1[`.
    kwargs: dict
        Extra parameters for :py:func:`~fourier_toolkit.nu2nu`.
        (For advanced users only.)

    Returns
    -------
    z: ArrayC
        (..., N1,...,ND) weights :math:`z_{n} \in \bC`.

    Notes
    -----
    * (x,w) must have the same numerical precision:

      - (x,) float32, (w,) complex64
      - (x,) float64, (w,) complex128

    * This implementation is a thin shell around an :math:`\tnunu` transform.
      This is because type-1 NUFFTs impose limits on domain(x, v_spec) that are a hindrance in practice.
      The price of relying on an :math:`\tnunu` transform is an extra interpolation step after the FFT, but its runtime is rarely the bottleneck.
      Note that the interpolation step can be done more efficiently via a low-rate filter bank, if needed.
    """
    x = _canonicalize_knots(x)
    M, Dx = x.shape
    N, Dv = v_spec.num, v_spec.ndim
    assert (D := Dx) == Dv
    assert D in (1, 2, 3)  # FINUFFT limitation

    sh = w.shape[:-1]
    z = nu2nu(
        x=x,
        v=np.require(
            a=v_spec.knots(x).reshape(math.prod(N), D),
            dtype=x.dtype,
            like=x,
        ),
        w=w,
        isign=isign,
        eps=eps,
        **kwargs,
    ).reshape((*sh, *N))
    return z


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
        :math:`\bbx_{m}` lattice specifier.
    v: ArrayR
        (N, D) support points :math:`\bbv_{n} \in \bR^{D}`.
    w: ArrayRC
        (..., M1,...,MD) weights :math:`w_{m} \in \bC`.
    eps: float
        Target relative error :math:`\epsilon \in ]0, 1[`.
    isign: +1, -1
        Exponent sign.
    kwargs: dict
        Extra parameters for :py:func:`~fourier_toolkit.nu2nu`.
        (For advanced users only.)

    Returns
    -------
    z: ArrayC
        (..., N) weights :math:`z_{n} \in \bC`.

    Notes
    -----
    * (v,w) must have the same numerical precision:

      - (v,) float32, (w,) complex64
      - (v,) float64, (w,) complex128

    * This implementation is a thin shell around an :math:`\tnunu` transform.
      This is because type-2 NUFFTs impose limits on domain(x_spec, v) that are a hindrance in practice.
      The price of relying on an :math:`\tnunu` transform is an extra interpolation step before the FFT, but its runtime is rarely the bottleneck.
      Note that the interpolation step can be done more efficiently via a low-rate filter bank, if needed.
    """
    M, Dx = x_spec.num, x_spec.ndim
    v = _canonicalize_knots(v)
    N, Dv = v.shape
    assert (D := Dx) == Dv
    assert D in (1, 2, 3)  # FINUFFT limitation

    sh = w.shape[:-D]
    z = nu2nu(
        x=np.require(
            a=x_spec.knots(v).reshape(math.prod(M), D),
            dtype=v.dtype,
            like=v,
        ),
        v=v,
        w=w.reshape((*sh, math.prod(M))),
        isign=isign,
        eps=eps,
        **kwargs,
    ).reshape((*sh, N))
    return z


# Helper routines (internal) ---------------------------------------------------
def _canonicalize_knots(xv: ftkt.ArrayR) -> ftkt.ArrayR:
    """
    Transform (x, v) knots to canonical form:
    - shape (M|N, D)
    - real-valued
    - F-ordered

    Parameters
    ----------
    xv: ArrayR

    Returns
    -------
    xv_c: ArrayR
        Canonical form of `xv`.
    """
    assert xv.ndim in (1, 2)
    if xv.ndim == 1:
        xv = xv[:, np.newaxis]
    _, D = xv.shape

    translate = ftku.TranslateDType(xv.dtype)
    fdtype = translate.to_float()

    xv_c = np.require(xv, fdtype, "F", like=xv)
    return xv_c


def _canonicalize_weights(w: ftkt.ArrayRC, M: int) -> tuple[ftkt.ArrayC, tuple]:
    """
    Transform (w,) weights to canonical form:
    - shape (n_trans, M)
    - complex-valued
    - C-ordered

    Parameters
    ----------
    w: ArrayRC

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

    translate = ftku.TranslateDType(w.dtype)
    cdtype = translate.to_complex()

    w_c = np.require(w, cdtype, "C", like=w)
    w_c = w_c.reshape((sh_flat, M))
    return w_c, sh


def _create_plan(
    x: ftkt.ArrayR,
    v: ftkt.ArrayR,
    w: ftkt.ArrayC,
    isign: ExponentSign,
    eps: float,
    **kwargs,
) -> finufft.Plan:
    r"""
    Parameters
    ----------
    x: ArrayR
        (M, D)
    v: ArrayR
        (N, D)
    w: ArrayC
        (n_trans, M)
    isign: +1, -1
    eps: ]0, 1[
    kwargs: dict

    Returns
    -------
    plan: finufft.Plan
    """
    _, D = x.shape
    n_trans, _ = w.shape

    plan = finufft.Plan(
        nufft_type=3,
        n_modes_or_dim=D,
        isign=isign,
        eps=eps,
        dtype=w.dtype,
        n_trans=n_trans,
        **kwargs,  # let FINUFFT parse them for correctness
    )

    scale = math.sqrt(2 * math.pi)
    setpts_kwargs = dict(
        zip(
            "xyz"[:D] + "stu"[:D],
            [*(scale * x.T), *(scale * v.T)],
        )
    )
    plan.setpts(**setpts_kwargs)

    return plan
