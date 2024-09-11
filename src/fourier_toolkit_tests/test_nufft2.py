import numpy as np
import pytest

import fourier_toolkit.nufft1 as ftk_nufft1
import fourier_toolkit.nufft2 as ftk_nufft2
import fourier_toolkit_tests.conftest as ct

# spread/interpolate() take tuple[func] as parameters -> known experimental feature.
pytestmark = pytest.mark.filterwarnings("ignore::numba.NumbaExperimentalFeatureWarning")


class TestNUFFT2:
    # NUFFT2 is just NUFFT1's adjoint with a sign flip.
    # We therefore assume de-facto that it works if NUFFT1 works.
    # All we check here is that the outputs are as expected, i.e. we didn't make a mistake in flipping the sign/etc.

    @pytest.mark.parametrize("D", [1, 2, 3])
    def test_coherent_with_nufft1(self, D):
        rng = np.random.default_rng()

        nu_spec = rng.uniform(-1, 1, (51, D))
        nu_d0 = np.ptp(nu_spec, axis=0)
        u_spec = dict(
            start=rng.uniform(-1, 1, D),
            step=rng.uniform(1e-1, 1, D) / nu_d0,
            num=rng.integers(15, 20, D),
        )
        M = len(nu_spec)  # int
        N = u_spec["num"]  # (D,)

        A = ftk_nufft1.NUFFT1(
            x=nu_spec,
            v_spec=u_spec,
            isign=1,
        )
        B = ftk_nufft2.NUFFT2(
            x_spec=u_spec,
            v=nu_spec,
            isign=-1,
        )

        sh = (2, 1, 3)
        w = 1j * rng.standard_normal((*sh, M))
        w += rng.standard_normal(w.shape)
        z = 1j * rng.standard_normal((*sh, *N))
        z += rng.standard_normal(z.shape)

        assert ct.relclose(A.apply(w), B.adjoint(w), D=D, eps=1e-6)
        assert ct.relclose(A.adjoint(z), B.apply(z), D=1, eps=1e-6)
