import array_api_compat as aac
import array_api_extra as aae

import fourier_toolkit.typing as ftkt
import fourier_toolkit.util as ftku


def allclose(a: ftkt.Array, b: ftkt.Array, dtype: ftkt.DType) -> bool:
    xp = aac.array_namespace(a)

    fdtype = ftku.TranslateDType(
        xp.asarray([], dtype=dtype),
    ).to_float()

    if fdtype in (xp.float32, xp.complex64):
        atol = 1e-6
    elif fdtype in (xp.float64, xp.complex128):
        atol = 1e-12
    else:
        raise ValueError

    match = xp.all(aae.isclose(a, b, atol=atol, equal_nan=True))
    return bool(match)
