import numpy as np
import pytest

import fourier_toolkit.nu2u as ftk_nu2u
import fourier_toolkit.u2nu as ftk_u2nu
import fourier_toolkit_tests.conftest as ct

# spread/interpolate() take tuple[func] as parameters -> known experimental feature.
pytestmark = pytest.mark.filterwarnings("ignore::numba.NumbaExperimentalFeatureWarning")


class TestU2NU:
    # U2NU is just NU2U's adjoint with a sign flip.
    # We therefore assume de-facto that it works if NU2U works.
    # All we check here is that the outputs are as expected, i.e. we didn't make a mistake in flipping the sign/etc.

    @pytest.mark.parametrize("D", [1, 2, 3])
    def test_coherent_with_nu2u(self, D):
        rng = np.random.default_rng()

        nu_spec = rng.uniform(-1, 1, (51, D))
        u_spec = dict(
            start=rng.uniform(-1, 1, D),
            step=rng.uniform(1e-1, 2, D),
            num=rng.integers(15, 20, D),
        )
        M = len(nu_spec)  # int
        N = u_spec["num"]  # (D,)

        A = ftk_nu2u.NU2U(
            x=nu_spec,
            v_spec=u_spec,
            isign=1,
        )
        B = ftk_u2nu.U2NU(
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
