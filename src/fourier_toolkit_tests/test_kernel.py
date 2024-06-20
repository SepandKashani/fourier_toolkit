import conftest as ct
import numpy as np
import pytest
import scipy.integrate as spi
import scipy.special as sps

import fourier_toolkit.kernel as ftk_kernel


class TestKaiserBessel:
    @pytest.mark.parametrize("beta", [5, 10])
    def test_value(self, beta):
        # output values match ground truth.
        rng = np.random.default_rng()

        op = ftk_kernel.KaiserBessel(beta)
        assert ct.allclose(op(0), kb(0, beta), np.double)  # scalar in/out ok
        x = rng.uniform(-op.support(), op.support(), 10)
        assert ct.allclose(op(x), np.array([kb(_x, beta) for _x in x]), np.double)  # vector in/out ok

        opF = ftk_kernel.KaiserBesselF(beta)
        assert ct.allclose(opF(0), kbF(0, beta), np.double)  # scalar in/out ok
        v = rng.uniform(-opF.support(), opF.support(), 10)
        assert ct.allclose(opF(v), np.array([kbF(_v, beta) for _v in v]), np.double)  # vector in/out ok

    @pytest.mark.parametrize(
        "dtype",
        [
            np.single,
            np.double,
        ],
    )
    def test_prec(self, dtype):
        # output dtype matches input dtype.
        rng = np.random.default_rng()

        op = ftk_kernel.KaiserBessel(beta=10)
        assert op(dtype(0)).dtype == dtype  # scalar in/out ok
        x = rng.uniform(-op.support(), op.support(), 10).astype(dtype)
        assert op(x).dtype == dtype  # vector in/out ok

        opF = ftk_kernel.KaiserBesselF(beta=10)
        assert opF(dtype(0)).dtype == dtype  # scalar in/out ok
        v = rng.uniform(-opF.support(), opF.support(), 10).astype(dtype)
        assert opF(v).dtype == dtype  # vector in/out ok

    def test_math_symmetric(self):
        # KaiserBessel[F] are symmetric around 0
        rng = np.random.default_rng()

        op = ftk_kernel.KaiserBessel(beta=10)
        x = rng.uniform(0, op.support(), 10)
        assert ct.allclose(op(x), op(-x), x.dtype)

        opF = ftk_kernel.KaiserBesselF(beta=10)
        v = rng.uniform(0, opF.support(), 10)
        assert ct.allclose(opF(v), opF(-v), v.dtype)

    def test_math_fourier(self):
        # kb(0) = \int_{-sF}^{sF} kbF(v) dv  [ if eps chosen tiny ]
        # kbF(0) = \int_{-s}^{s} kb(x) dx    [ if eps chosen tiny ]
        op = ftk_kernel.KaiserBessel.from_eps(1e-16)
        opF = ftk_kernel.KaiserBesselF.from_eps(1e-16)

        kb0, kbF0 = op(0), opF(0)
        kb_int, *_ = spi.quad(op.low_level_callable(), -op.support(), op.support())
        kbF_int, *_ = spi.quad(opF.low_level_callable(), -opF.support(), opF.support())

        assert ct.allclose(kb0, kbF_int, np.single)  # low-precision sufficient
        assert ct.allclose(kbF0, kb_int, np.single)

    @pytest.mark.parametrize(
        "eps",
        [
            1e-3,
            1e-8,
            1e-11,
            1e-16,
        ],
    )
    def test_math_parseval(self, eps):
        # energy(kb) (1-eps) == energy(kbF) [ within passband ]
        op = ftk_kernel.KaiserBessel.from_eps(eps)
        opF = ftk_kernel.KaiserBesselF.from_eps(eps)

        E_kb, *_ = spi.quad(lambda _: op(_) ** 2, -op.support(), op.support())
        E_kbF, *_ = spi.quad(lambda _: opF(_) ** 2, -opF.support(), opF.support())
        assert ct.allclose(E_kb * (1 - eps), E_kbF, np.single)  # low-precision sufficient


class TestPPoly:
    @pytest.mark.parametrize("sym", [True, False])
    def test_value(self, sym):
        # output values match ground truth
        kbF = ftk_kernel.KaiserBesselF(beta=10)
        ppoly = ftk_kernel.PPoly.from_kernel(kbF, B=100, N=20, sym=sym)  # an extreme fit

        x = np.linspace(-kbF.support(), kbF.support(), 73)
        y_gt = kbF(x)
        y = ppoly(x)
        assert ct.allclose(y, y_gt, y_gt.dtype)

    @pytest.mark.parametrize(
        "dtype",
        [
            np.single,
            np.double,
        ],
    )
    def test_prec(self, dtype):
        # output dtype matches input dtype.
        kbF = ftk_kernel.KaiserBesselF(beta=10)
        ppoly = ftk_kernel.PPoly.from_kernel(kbF, B=100, N=20, sym=False)  # an extreme fit

        assert ppoly(dtype(0)).dtype == dtype  # scalar in/out ok
        v = np.linspace(-ppoly.support(), ppoly.support(), 11).astype(dtype)
        assert ppoly(v).dtype == dtype  # vector in/out ok

    @pytest.mark.parametrize("sym", [True, False])
    def test_fit_kernel(self, sym):
        # fit_kernel() produces right (B,N) values
        kbF = ftk_kernel.KaiserBesselF(beta=10)
        B, N = 13, 21
        w, p = ftk_kernel.PPoly.fit_kernel(kbF, B, N, sym)
        assert w.shape == (B, N + 1)
        if sym:
            assert np.allclose(p, kbF.support() / B)
        else:
            assert np.allclose(p, 2 * kbF.support() / B)


# Ground truth re-implementations of KB pulses for testing purposes -----------
def kb(x: float, beta: float) -> float:
    # f(x) = \frac{
    #     I_{0}(\beta \sqrt{1 - x^{2}})
    # }{
    #     I_{0}(\beta)
    # } 1_{[-1, 1]}(x)
    assert beta > 0

    if x <= 1:
        y = sps.i0(beta * np.sqrt(1 - (x**2)))
        y /= sps.i0(beta)
    else:
        y = 0
    return y


def kbF(v: float, beta: float) -> float:
    # f^{\mathcal{F}}(v) =
    #     \frac{2}{I_{0}(\beta)}
    #     \frac{\sinh \gamma}{\gamma}
    # with \gamma = \sqrt{\beta^{2} - (2 \pi v)^{2}}
    assert beta > 0

    a = beta**2 - (2 * np.pi * v) ** 2
    before_cutoff = a >= 0
    a = np.sqrt(np.fabs(a))

    if before_cutoff:
        z = np.sinh(a) / a
    else:
        z = np.sinc(a / np.pi)

    z *= 2 / sps.i0(beta)
    return z
