import pathlib as plib

import numpy as np
import scipy.special as sps

from fourier_toolkit.config import generate_module

__all__ = [
    "KaiserBessel",
    "KaiserBesselF",
]


class Kernel:
    r"""
    Essentially-finite-support function :math:`f: \bR \to \bR`.
    """

    def __init__(self):
        pass

    def support(self) -> float:
        r"""
        Function support.

        Returns
        -------
        s: float
            Value such that `f(x) \essentially 0` for all `x \notin [-s, s]`.
        """
        pass

    def low_level_callable(self) -> callable:
        """
        Numba-compiled ufunc.

        Useful to pass around explicitly to 3rd-party functions.
        """
        pass

    def __call__(self, x):
        """
        Call a Numba-compiled ufunc to compute `f(x)`.

        Must accept FP32/FP64 inputs.
        """
        func = self.low_level_callable()
        return func(x)


class KaiserBessel(Kernel):
    r"""
    Kaiser-Bessel pulse.

    f(x) = I0(\beta * \sqrt[ 1 - x**2 ]) / I0(\beta)
           1_{[-1,1]}(x)
    """

    template_path = plib.Path(__file__).parent / "_kb_template.txt"

    def __init__(self, beta: float):
        assert beta > 0
        self._beta = float(beta)

        self._pkg = generate_module(
            path=self.template_path,
            subs=dict(
                beta=self._beta,
                i0_beta=sps.i0(self._beta),
            ),
        )

    def support(self) -> float:
        return 1.0

    def low_level_callable(self) -> callable:
        return self._pkg.apply

    @classmethod
    def beta_from_eps(cls, eps: float) -> float:
        r"""
        Set \beta parameter given relative energy \epsilon at cut-off frequency.

            ``beta = a log10(eps) + b``

        See `data/kb_beta_estimation/generate.py:fit_model()` for (a,b) computation.
        """
        assert 0 < eps < 1
        a = -1.2192969471866881
        b = +1.196087484290671
        beta = a * np.log10(eps) + b
        return float(beta)

    @classmethod
    def from_eps(cls, eps: float):
        """
        Create instance from relative energy at cut-off frequency.
        """
        beta = cls.beta_from_eps(eps)
        return cls(beta)


class KaiserBesselF(KaiserBessel):
    r"""
    Kaiser-Bessel Fourier Transform.

    f^{F}(v) = (2 / I0(\beta)) \sinh[a] / a
           a = \sqrt[ \beta**2 - (2\pi v)**2 ]
    """

    def support(self) -> float:
        return self._beta / (2 * np.pi)

    def low_level_callable(self) -> callable:
        return self._pkg.applyF
