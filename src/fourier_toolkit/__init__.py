import importlib.metadata

__version__ = importlib.metadata.version("fourier_toolkit")

# Internal API elements exported for external use -----------------------------

from .nufft import (
    nu2nu as nu2nu,
    nu2u as nu2u,
    u2nu as u2nu,
)

from .typing import (
    ArrayR as ArrayR,
    ArrayC as ArrayC,
    ArrayRC as ArrayRC,
)

from .ufft import (
    u2u as u2u,
)

from .util import (
    UniformSpec as UniformSpec,
)

# -----------------------------------------------------------------------------
