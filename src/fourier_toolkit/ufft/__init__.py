from typing import Literal

import array_api_compat as aac

import fourier_toolkit.typing as ftkt
import fourier_toolkit.util as ftku

__all__ = [
    "u2u",
]


ExponentSign = Literal[-1, +1]


def u2u(
    x_spec: ftku.UniformSpec,
    v_spec: ftku.UniformSpec,
    w: ftkt.ArrayRC,
    isign: ExponentSign,
) -> ftkt.ArrayC:
    r"""
    Multi-dimensional Uniform-to-Uniform Fourier Transform. (:math:`\tuu`)

    Computes the Fourier sum

    .. math::

       \bbz_{n} = \sum_{m} w_{m} \ee^{ \pm \cj 2\pi \innerProduct{\bbx_{m}}{\bbv_{n}} },

    where :math:`(\bbx_{m}, \bbv_{n})` lie on the regular lattice

    .. math::

       \bbx_{\bbm} &= \bbx_{0} + \Delta_{\bbx} \odot \bbm, \qquad [\bbm]_{d} \in \discreteRange{0}{M_{d}-1}, \\
       \bbv_{\bbn} &= \bbv_{0} + \Delta_{\bbv} \odot \bbn, \qquad [\bbn]_{d} \in \discreteRange{0}{N_{d}-1},

    with :math:`M = \prod_{d} M_{d}` and :math:`N = \prod_{d} N_{d}`.

    Parameters
    ----------
    x_spec: UniformSpec
        :math:`\bbx_{m}` lattice.
    v_spec: UniformSpec
        :math:`\bbv_{n}` lattice.
    isign: +1, -1
        Exponent sign.
    w: ArrayRC
        (..., M1,...,MD) weights :math:`w_{m} \in \bC`.

    Returns
    -------
    z: ArrayC
        (..., N1,...,ND) weights :math:`z_{n} \in \bC`.

    Notes
    -----
    * :math:`\tuu` transforms for arbitrary (x_spec,v_spec) can be implemented via
      the CZT algorithm (using 2 FFTs), but a single FFT can be used in some cases.
      This implementation chooses the (FFT, CZT) per axis to maximize efficiency.

    * Backend-specific notes:

      - [JAX] supports the following program transforms:

        + :py:func:`~jax.jit`
        + :py:func:`~jax.vmap`
        + :py:func:`~jax.jvp`
        + :py:func:`~jax.vjp`

      - [PyTorch] supports the following program transforms:

        + :py:func:`~torch.compile`, but it has limited support for complex-valued operators: do not expect huge performance gains.
        + :py:func:`~torch.func.vmap`, but it is currently broken when any transform axis uses the FFT algorithm instead of the CZT algorithm.
        + :py:func:`~torch.func.jvp`
        + :py:func:`~torch.func.vjp`
    """
    assert (isign := int(isign)) in (-1, +1)

    if aac.is_jax_array(w):
        from . import jax as namespace
    elif aac.is_torch_array(w):
        from . import torch as namespace
    else:
        from . import generic as namespace

    z = namespace._u2u(x_spec, v_spec, w, isign)
    return z
