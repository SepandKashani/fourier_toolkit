# This module defines array types used throughout the codebase.

import typing

import numpy as np
import numpy.typing as npt

__all__ = [
    "ArrayR",
    "ArrayC",
    "ArrayRC",
]

# real-valued ndarrays
ArrayR = typing.Union[
    npt.NDArray[np.float32],
    npt.NDArray[np.float64],
]
"""
A (NumPy, CuPy) array of dtype float[32,64].
"""

# complex-valued ndarrays
ArrayC = typing.Union[
    npt.NDArray[np.complex64],
    npt.NDArray[np.complex128],
]
"""
A (NumPy, CuPy) array of dtype complex[64,128].
"""

# (real,complex)-valued ndarrays
ArrayRC = typing.Union[
    ArrayR,
    ArrayC,
]
"""
A (NumPy, CuPy) array of dtype (float[32,64], complex[64,128]).
"""
