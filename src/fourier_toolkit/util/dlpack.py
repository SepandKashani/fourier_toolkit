import enum


class DLDeviceType(enum.IntEnum):
    """
    Subset of `DLPack Device Codes <https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__dlpack_device__.html#array_api.array.__dlpack_device__>`_.
    """

    CPU = 1
    CUDA = 2
