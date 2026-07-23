import math
import typing as typ

import array_api_compat as aac

import fourier_toolkit.typing as ftkt
import fourier_toolkit.util as ftku

__all__ = [
    "nu2nu",
    "nu2u",
    "u2nu",
]

_eps_default: float = 1e-6
ExponentSign = typ.Literal[-1, +1]
_eps_fp32_min = 1e-6
_eps_fp64_min = 1e-14

finufft_kwargsT = tuple[tuple[str, int], ...]


def nu2nu(
    x: ftkt.ArrayR,
    v: ftkt.ArrayR,
    w: ftkt.ArrayC,
    isign: ExponentSign,
    eps: float = _eps_default,
    finufft_kwargs: finufft_kwargsT = (),
) -> ftkt.ArrayC:
    r"""
    Multi-dimensional NonUniform-to-NonUniform Fourier Transform. (:math:`\tnunu`)

    Computes the Fourier sum

    .. math::

       z_{n} = \sum_{m} w_{m} \ee^{ \pm \cj 2\pi \innerProduct{\bbx_{m}}{\bbv_{n}} },

    where :math:`(\bbx_{m}, \bbv_{n}) \in \bR^{D}`.

    Parameters
    ----------
    x: ArrayR
        (M, D) knots :math:`\bbx_{m} \in \bR^{D}`.
    v: ArrayR
        (N, D) knots :math:`\bbv_{n} \in \bR^{D}`.
    w: ArrayC
        (..., M) weights :math:`w_{m} \in \bC`.
    isign: +1, -1
        Exponent sign.
    eps: float
        Target relative error :math:`\epsilon \in ]0, 1[`.
        (See :ref:`nu2nu-notes`.)
    finufft_kwargs: tuple[tuple[str,int], ...]
        Immutable sequence of (key,value) options for :py:class:`finufft.Plan`.
        Uses FINUFFT's default option settings if omitted.
        (For advanced users only.)

    Returns
    -------
    z: ArrayC
        (..., N) weights :math:`z_{n} \in \bC`.


    .. _nu2nu-notes:

    Notes
    -----
    * The :math:`\tnunu` transform is an approximate transform such that the computed output :math:`\bbz \in \bC^{N}` satisfies

      .. math::

         \norm{\bbz - \bbz^{GT}}{2} \lesssim \epsilon \norm{\bbz^{GT}}{2}.

    * (x,v,w) must share the same (backend,device).

    * The relative error :math:`\epsilon` is limited by the precision of (x,v,w):

      - any(x,v,w).dtype = float32 :math:`\rightarrow \epsilon_{\min}` = 1e-06
      - all(x,v,w).dtype = float64 :math:`\rightarrow \epsilon_{\min}` = 1e-14
    """
    assert ftku.is_complex(w)
    xp = aac.array_namespace(x)

    # standardize inputs -----------------------------------------------------
    fdtype, cdtype = _determine_precision(x, v, w)

    x = _canonicalize_knots(x, fdtype)
    v = _canonicalize_knots(v, fdtype)
    w = xp.astype(w, cdtype, copy=False)
    _, Dx = x.shape
    _, Dv = v.shape
    assert (D := Dx) == Dv
    assert D in (1, 2, 3)  # FINUFFT limitation

    assert (isign := int(isign)) in (-1, +1)

    assert 0 < (eps := float(eps)) < 1
    if cdtype == xp.complex64:
        eps = max(eps, _eps_fp32_min)
    elif cdtype == xp.complex128:
        eps = max(eps, _eps_fp64_min)
    else:
        raise ValueError(f"Unknown dtype: {cdtype}.")
    # ------------------------------------------------------------------------

    if aac.is_jax_array(w):
        from . import jax as namespace
    else:
        from . import generic as namespace

    z = namespace._nu2nu(x, v, w, isign, eps, finufft_kwargs)
    return z


def nu2u(
    x: ftkt.ArrayR,
    v_spec: ftku.UniformSpec,
    w: ftkt.ArrayC,
    isign: ExponentSign,
    eps: float = _eps_default,
    finufft_kwargs: finufft_kwargsT = (),
) -> ftkt.ArrayC:
    r"""
    Multi-dimensional NonUniform-to-Uniform Fourier Transform. (:math:`\tnuu`)

    Computes the Fourier sum

    .. math::

       z_{n} = \sum_{m} w_{m} \ee^{ \pm \cj 2\pi \innerProduct{\bbx_{m}}{\bbv_{n}} },

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
    w: ArrayC
        (..., M) weights :math:`w_{m} \in \bC`.
    isign: +1, -1
        Exponent sign.
    eps: float
        Target relative error :math:`\epsilon \in ]0, 1[`.
        (See :ref:`nu2u-notes`.)
    finufft_kwargs: tuple[tuple[str,int], ...]
        Immutable sequence of (key,value) options for :py:class:`finufft.Plan`.
        Uses FINUFFT's default option settings if omitted.
        (For advanced users only.)

    Returns
    -------
    z: ArrayC
        (..., N1,...,ND) weights :math:`z_{n} \in \bC`.


    .. _nu2u-notes:

    Notes
    -----
    * The :math:`\tnuu` transform is an approximate transform such that the computed output :math:`\bbz \in \bC^{N_{1} \times\cdots\times N_{D}}` satisfies

      .. math::

         \norm{\bbz - \bbz^{GT}}{2} \lesssim \epsilon \norm{\bbz^{GT}}{2}.

    * (x,w) must share the same (backend,device).

    * The relative error :math:`\epsilon` is limited by the precision of (x,w):

      - any(x,w).dtype = float32 :math:`\rightarrow \epsilon_{\min}` = 1e-06
      - all(x,w).dtype = float64 :math:`\rightarrow \epsilon_{\min}` = 1e-14
    """
    assert ftku.is_complex(w)
    xp = aac.array_namespace(x)
    v = xp.reshape(
        v_spec.knots(like=x),
        (-1, v_spec.ndim),
    )

    sh = w.shape[:-1]
    z = xp.reshape(
        nu2nu(x, v, w, isign, eps, finufft_kwargs),
        (*sh, *v_spec.num),
    )
    return z


def u2nu(
    x_spec: ftku.UniformSpec,
    v: ftkt.ArrayR,
    w: ftkt.ArrayC,
    isign: ExponentSign,
    eps: float = _eps_default,
    finufft_kwargs: finufft_kwargsT = (),
) -> ftkt.ArrayC:
    r"""
    Multi-dimensional Uniform-to-NonUniform Fourier Transform. (:math:`\tunu`)

    Computes the Fourier sum

    .. math::

       z_{n} = \sum_{m} w_{m} \ee^{ \pm \cj 2\pi \innerProduct{\bbx_{m}}{\bbv_{n}} },

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
    w: ArrayC
        (..., M1,...,MD) weights :math:`w_{m} \in \bC`.
    isign: +1, -1
        Exponent sign.
    eps: float
        Target relative error :math:`\epsilon \in ]0, 1[`.
        (See :ref:`u2nu-notes`.)
    finufft_kwargs: tuple[tuple[str,int], ...]
        Immutable sequence of (key,value) options for :py:class:`finufft.Plan`.
        Uses FINUFFT's default option settings if omitted.
        (For advanced users only.)

    Returns
    -------
    z: ArrayC
        (..., N) weights :math:`z_{n} \in \bC`.


    .. _u2nu-notes:

    Notes
    -----
    * The :math:`\tunu` transform is an approximate transform such that the computed output :math:`\bbz \in \bC^{N}` satisfies

      .. math::

         \norm{\bbz - \bbz^{GT}}{2} \lesssim \epsilon \norm{\bbz^{GT}}{2}.

    * (v,w) must share the same (backend,device).

    * The relative error :math:`\epsilon` is limited by the precision of (v,w):

      - any(v,w).dtype = float32 :math:`\rightarrow \epsilon_{\min}` = 1e-06
      - all(v,w).dtype = float64 :math:`\rightarrow \epsilon_{\min}` = 1e-14
    """
    assert ftku.is_complex(w)
    xp = aac.array_namespace(v)
    x = xp.reshape(
        x_spec.knots(like=v),
        (-1, x_spec.ndim),
    )

    sh = w.shape[: -x_spec.ndim]
    z = nu2nu(
        x,
        v,
        xp.reshape(w, (*sh, math.prod(x_spec.num))),
        isign,
        eps,
        finufft_kwargs,
    )
    return z


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
    # device = ftku.get_device(args[0])
    for a in args:
        assert aac.array_namespace(a) == xp  # same backend
        # assert ftku.get_device(a) == device  # same device
        #   [20260722] Deactivated since does not work with jax.vmap()
        #   Not as issue since device mis-matches will cause breakage downstream anyway.
        assert xp.finfo(a.dtype).bits in (32, 64)  # single/double precision

    n_bits = min(xp.finfo(a.dtype).bits for a in args)
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
