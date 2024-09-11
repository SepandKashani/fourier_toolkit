import numpy as np
import pytest

import fourier_toolkit.nufft1 as ftk_nufft1
import fourier_toolkit_tests.conftest as ct
import fourier_toolkit_tests.test_nu2u as ct_nu2u

# spread/interpolate() take tuple[func] as parameters -> known experimental feature.
pytestmark = pytest.mark.filterwarnings("ignore::numba.NumbaExperimentalFeatureWarning")

# We piggy-back onto tests defined in TestNU2U since the interface is near-identical.
# _generate_NU2U_inputs() is replaced with _generate_NUFFT1_inputs() since chunking is not supported (yet).


def _generate_NUFFT1_inputs() -> list:
    # Create (D, x_m, v_spec) pairs used to initialize fixtures
    data = []

    # D=1 -------------------------------------------------
    D = 1
    rng = np.random.default_rng(0)
    T = rng.uniform(1, 10, size=D)
    x_m = ct.generate_point_cloud(
        N_point=1_00,
        D=D,
        bbox_dim=T,
        N_blk=100,
        sparsity_ratio=(5e-2) ** D,
        rng=rng,
    )
    v_spec = dict(
        start=rng.uniform(-1, 1, D),
        step=1 / T,
        num=2_00 + np.arange(D),
    )
    data.append((D, x_m, v_spec))

    # D=2 -------------------------------------------------
    D = 2
    rng = np.random.default_rng(0)
    T = rng.uniform(1, 10, size=D)
    x_m = ct.generate_point_cloud(
        N_point=1_00,
        D=D,
        bbox_dim=T,
        N_blk=100,
        sparsity_ratio=(5e-2) ** D,
        rng=rng,
    )
    v_spec = dict(
        start=rng.uniform(-1, 1, D),
        step=1 / T,
        num=2_00 + np.arange(D),
    )
    data.append((D, x_m, v_spec))

    return data


class TestNUFFT1(ct_nu2u.TestNU2U):
    # Fixtures ----------------------------------------------------------------
    @pytest.fixture(params=_generate_NUFFT1_inputs())
    def D_xM_vspec(self, request) -> tuple:
        D, x_m, v_spec = request.param
        return D, x_m, v_spec

    @pytest.fixture
    def space_dim(self, D_xM_vspec) -> int:
        D, x_m, v_spec = D_xM_vspec
        return D

    @pytest.fixture
    def x_m(self, D_xM_vspec) -> np.ndarray:
        D, x_m, v_spec = D_xM_vspec
        return x_m

    @pytest.fixture
    def v_spec(self, D_xM_vspec) -> dict[str, np.ndarray]:
        D, x_m, v_spec = D_xM_vspec
        return v_spec

    @pytest.fixture
    def op(  # we only test settings which affect accuracy, not runtime
        self,
        x_m,
        v_spec,
        *,
        isign,
        # Accuracy-related ------------
        eps,
        upsampfac,
        # Runtime behavior ------------
        # fft_nthreads,
        # spread_nthreads,
        # max_cluster_size,
        # max_window_ratio,
    ) -> ftk_nufft1.NUFFT1:
        op = ftk_nufft1.NUFFT1(
            x=x_m,
            v_spec=v_spec,
            isign=isign,
            eps=eps,
            upsampfac=upsampfac,
        )
        return op

    # Helper functions --------------------------------------------------------
