# This module defines array types used throughout the codebase.

from types import ModuleType
from typing import Union

import numpy as np
from array_api_typing import Array

__all__ = [
    "ArrayR",
    "ArrayC",
    "ArrayRC",
]

ArrayR = Union[
    Array[np.float32, ModuleType],
    Array[np.float64, ModuleType],
]
"""
Array-API-compatible array of dtype float[32,64].
"""

ArrayC = Union[
    Array[np.complex64, ModuleType],
    Array[np.complex128, ModuleType],
]
"""
Array-API-compatible array of dtype complex[64,128].
"""

ArrayRC = Union[
    ArrayR,
    ArrayC,
]
"""
Array-API-compatible array of dtype (float[32,64], complex[64,128]).
"""
