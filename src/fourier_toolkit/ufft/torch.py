import fourier_toolkit.typing as ftkt
import fourier_toolkit.util as ftku

from . import ExponentSign


def _u2u(
    x_spec: ftku.UniformSpec,
    v_spec: ftku.UniformSpec,
    w: ftkt.ArrayRC,
    isign: ExponentSign,
) -> ftkt.ArrayC:
    from .generic import _u2u as generic_u2u

    z = generic_u2u(x_spec, v_spec, w, isign)
    return z
