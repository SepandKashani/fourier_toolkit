import enum

import fourier_toolkit.typing as ftkt


class DLDeviceType(enum.IntEnum):
    """
    Subset of `DLPack Device Codes <https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__dlpack_device__.html#array_api.array.__dlpack_device__>`_.
    """

    CPU = 1
    CUDA = 2


def get_device(x: ftkt.Array):
    """
    Extract device on which array resides, with graceful default to `None`.

    This helper exists to circumvent issues with array-creating functions when used with :py:func:`jax.jit`.
    """
    return getattr(x, "device", None)
