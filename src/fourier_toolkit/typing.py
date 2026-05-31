# This module defines types used throughout the codebase.

from __future__ import annotations

import typing as typ

__all__ = [
    "ArrayC",
    "ArrayNameSpace",
    "ArrayR",
    "ArrayRC",
    "DType",
]


# Data Types ===================================================================
class DType:
    """
    Data type objects used as `dtype` specifiers in functions and methods.
    """

    pass


class int32(DType):
    pass


class int64(DType):
    pass


class float32(DType):
    pass


class float64(DType):
    pass


class complex64(DType):
    pass


class complex128(DType):
    pass


DTypeT = typ.TypeVar("DTypeT", bound=DType)


# Array NameSpace ==============================================================
@typ.runtime_checkable
class ArrayNameSpace(typ.Protocol):
    """
    Object that has all the array API functions in it.
    """

    # For typing purposes, we limit ourselves to a subset of the array API.

    def __array_namespace_info__(self) -> typ.Any: ...

    int32: type[int32]
    int64: type[int64]
    float32: type[float32]
    float64: type[float64]
    complex64: type[complex64]
    complex128: type[complex128]


@typ.runtime_checkable
class Array(typ.Protocol[DTypeT]):
    """
    Array-API-compatible array.
    """

    dtype: DTypeT
    device: typ.Any
    ndim: int
    shape: tuple[int, ...]
    size: int

    @property
    def T(self) -> Array: ...

    @property
    def mT(self) -> Array: ...

    def __array_namespace__(
        self,
        *,
        api_version: str | None = None,
    ) -> ArrayNameSpace: ...


ArrayR = typ.Union[
    Array[float32],
    Array[float64],
]
"""
Array-API-compatible array of dtype float[32,64].
"""

ArrayC = typ.Union[
    Array[complex64],
    Array[complex128],
]
"""
Array-API-compatible array of dtype complex[64,128].
"""

ArrayRC = typ.Union[
    ArrayR,
    ArrayC,
]
"""
Array-API-compatible array of dtype (float[32,64], complex[64,128]).
"""
