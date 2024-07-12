import pathlib as plib

import numpy as np
import scipy.linalg as spl
import scipy.special as sps

from fourier_toolkit.config import generate_module

__all__ = [
    "KaiserBessel",
    "KaiserBesselF",
    "PPoly",
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

    def low_level_callable(self, ufunc: bool) -> callable:
        """
        Numba-compiled function, in scalar or ufunc version.

        Useful to pass around explicitly to 3rd-party functions.
        """
        pass

    def __call__(self, x):
        """
        Call a Numba-compiled ufunc to compute `f(x)`.

        Must accept FP32/FP64 inputs.
        """
        func = self.low_level_callable(ufunc=True)
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

    def low_level_callable(self, ufunc: bool) -> callable:
        if ufunc:
            return self._pkg.v_apply
        else:
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

    def low_level_callable(self, ufunc: bool) -> callable:
        if ufunc:
            return self._pkg.v_applyF
        else:
            return self._pkg.applyF


class PPoly(Kernel):
    r"""
    Piecewise-Polynomial pulse.

    For non-symmetric functions f: [-s, s] -> R:
        f(x) = \sum_{n=0...N} w[b, n] x**n
           b = int(x + s) / pitch  # bin index
           all bins lie in [-s, s]
    For symmetric functions f: [-s, s] -> R:
        f(x) = \sum_{n=0...N} w[b, n] |x|**n
           b = int(|x| / pitch)  # bin index
           all bins lie in [0, s].

    w = (B, N+1) polynomial coefficients. (B bins, order-N segments)
    """

    template_path = plib.Path(__file__).parent / "_ppoly_template.txt"

    def __init__(
        self,
        weight: np.ndarray,
        pitch: float,
        sym: bool,
    ):
        """
        Parameters
        ----------
        weight: ndarray
            (B, N+1) coefficients encoding a B-piecewise polynomial of order N.

            Coefficients are ordered in decreasing powers (aN,...,a0).
        pitch: float
            Width of each bin.
        sym: bool
            Symmetric or non-symmetric parameterization provided.
        """
        self._weight = weight
        assert pitch > 0
        self._pitch = float(pitch)
        self._sym = bool(sym)

        self._pkg = generate_module(
            path=self.template_path,
            subs=dict(
                sym=self._sym,
                bin_count=self._weight.shape[0],
                poly_order=self._weight.shape[1] - 1,
                pitch_rcp=1 / self._pitch,
                support=self.support(),
                weight=self._print_weights(),
            ),
        )

    def support(self) -> float:
        B = self._weight.shape[0]
        if self._sym:
            s = B * self._pitch
        else:
            s = B * self._pitch / 2
        return float(s)

    def low_level_callable(self, ufunc: bool) -> callable:
        if ufunc:
            return self._pkg.v_apply
        else:
            return self._pkg.apply

    @classmethod
    def fit_kernel(
        cls,
        kern: Kernel,
        B: int,
        N: int,
        sym: bool,
    ):
        """
        Find (weight, pitch) pair which best approximates a given kernel.

        Parameters
        ----------
        kern: Kernel
            Function to approximate.
        B: int
            Number of piecewise bins.
        N: int
            Polynomial order.
        sym: bool
            Assume kernel is symmetric.

        Returns
        -------
        weight: ndarray
            (B, N+1) coefficients encoding a B-piecewise polynomial of order N.

            Coefficients are ordered in decreasing powers (aN,...,a0).
        pitch: float
            Width of each bin.
        """
        assert B >= 1
        assert N >= 0

        if sym:
            pitch = kern.support() / B
            bin_bound = np.linspace(0, kern.support(), B + 1)
        else:
            pitch = 2 * kern.support() / B
            bin_bound = np.linspace(-kern.support(), kern.support(), B + 1)

        offset = np.linspace(0, pitch, N + 1)
        weight = np.zeros((B, N + 1))
        for b in range(B):
            A = (bin_bound[b] + offset.reshape(-1, 1)) ** np.arange(N, -1, -1)  # (N+1, N+1)
            y = kern(bin_bound[b] + offset)
            weight[b], *_ = spl.lstsq(A, y)

        return weight, pitch

    @classmethod
    def from_kernel(
        cls,
        kern: Kernel,
        B: int,
        N: int,
        sym: bool,
    ):
        """
        Create instance by approximating a kernel.

        Parameters
        ----------
        kern: Kernel
            Function to approximate.
        B: int
            Number of piecewise bins.
        N: int
            Polynomial order.
        sym: bool
            Assume kernel is symmetric.
        """
        w, p = cls.fit_kernel(kern, B, N, sym)
        return cls(w, p, sym)

    def _print_weights(self) -> str:
        # Print each bin weight on seperable line.
        B = self._weight.shape[0]
        N = self._weight.shape[1] - 1

        s = ""
        for b in range(B):
            s += "    ("
            for n in range(N + 1):
                s += f"{self._weight[b, n]},"
            s += "),\n"

        s = s[:-1]  # drop final \n
        return s
