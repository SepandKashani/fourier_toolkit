import numpy as np
import pytest

import fourier_toolkit.nu2nu as ftk_nu2nu
import fourier_toolkit.util as ftk_util
import fourier_toolkit_tests.conftest as ct

# spread/interpolate() take tuple[func] as parameters -> known experimental feature.
pytestmark = pytest.mark.filterwarnings("ignore::numba.NumbaExperimentalFeatureWarning")


class TestNU2NU:
    @pytest.mark.parametrize("real", [True, False])
    @pytest.mark.parametrize("stack_shape", [(), (1,), (5, 3, 4)])
    def test_value_apply(self, op, x_m, v_n, isign, dtype, real, stack_shape):
        # output value matches ground truth.
        translate = ftk_util.TranslateDType(dtype)
        fdtype = translate.to_float()
        cdtype = translate.to_complex()

        # Generate NU2NU input
        rng = np.random.default_rng()
        if real:
            w = rng.standard_normal((*stack_shape, op.cfg.M))
            w = w.astype(fdtype)
        else:
            w = 1j * rng.standard_normal((*stack_shape, op.cfg.M))
            w += rng.standard_normal((*stack_shape, op.cfg.M))
            w = w.astype(cdtype)

        # Generate NU2NU output ground-truth
        z_gt = np.zeros((*stack_shape, op.cfg.N), dtype=cdtype)
        A = self._generate_A(x_m, v_n, isign)  # (N, M)
        for idx in np.ndindex(stack_shape):
            z_gt[idx] = ct.inner_product(w[idx], A, 1)  # (N,)

        # Test NU2NU compliance
        z = op.apply(w)
        assert z.shape == z_gt.shape
        assert ct.relclose(z, z_gt, D=1, eps=1e-3)  # todo: progressive eps?

    @pytest.mark.parametrize("real", [True, False])
    @pytest.mark.parametrize("direction", ["apply", "adjoint"])
    def test_prec(self, op, dtype, real, direction):
        # output precision (not dtype!) matches input precision.
        translate = ftk_util.TranslateDType(dtype)
        fdtype = translate.to_float()
        cdtype = translate.to_complex()

        rng = np.random.default_rng()
        size = op.cfg.M if (direction == "apply") else op.cfg.N
        if real:
            x = rng.standard_normal(size)
            x = x.astype(fdtype)
        else:
            x = 1j * rng.standard_normal(size)
            x += rng.standard_normal(size)
            x = x.astype(cdtype)

        f = getattr(op, direction)
        y = f(x)
        assert y.dtype == cdtype

    def test_math_adjoint(self, op, dtype):
        # <A x, y> == <x, A^H y>
        translate = ftk_util.TranslateDType(dtype)
        cdtype = translate.to_complex()

        sh = (5, 3, 4)
        rng = np.random.default_rng()
        x = 1j * rng.standard_normal((*sh, op.cfg.M))
        x += rng.standard_normal((*sh, op.cfg.M))
        x = x.astype(cdtype)
        y = 1j * rng.standard_normal((*sh, op.cfg.N))
        y += rng.standard_normal((*sh, op.cfg.N))
        y = y.astype(cdtype)

        lhs = ct.inner_product(op.apply(x), y, D=1)
        rhs = ct.inner_product(x, op.adjoint(y), D=1)
        assert ct.relclose(lhs, rhs, D=1, eps=1e-3)  # todo: progressive eps?

    # Fixtures ----------------------------------------------------------------
    @pytest.fixture(params=[1, 2])  # we don't do 3D since [unchunked] Heisenberg volume may be large.
    def space_dim(self, request) -> int:
        return request.param

    @pytest.fixture
    def x_m(self, space_dim) -> np.ndarray:
        rng = np.random.default_rng()
        x = ct.generate_point_cloud(
            N_point=50,
            D=space_dim,
            bbox_dim=np.arange(space_dim) + 10,
            N_blk=np.arange(space_dim) + 20,
            sparsity_ratio=0.1,
            rng=rng,
        )
        return x

    @pytest.fixture
    def v_n(self, space_dim) -> np.ndarray:
        rng = np.random.default_rng()
        v = ct.generate_point_cloud(
            N_point=51,
            D=space_dim,
            bbox_dim=np.arange(space_dim) + 7,
            N_blk=np.arange(space_dim) + 10,
            sparsity_ratio=0.1,
            rng=rng,
        )
        return v

    @pytest.fixture(params=[-1, 1])
    def isign(self, request) -> int:
        return request.param

    @pytest.fixture(params=[1e-16])
    def eps(self, request) -> float:
        return request.param

    @pytest.fixture(params=[2])
    def upsampfac(self, request) -> float:
        return request.param

    @pytest.fixture(params=[0.5])
    def upsampfac_ratio(self, request) -> float:
        return request.param

    @pytest.fixture(
        params=[
            (False, "xv"),  # domain doesn't matter
            (True, "x"),
            (True, "v"),
            (True, "xv"),
        ]
    )
    def chunked_domain(self, request) -> tuple[bool, str]:
        return request.param

    @pytest.fixture
    def chunked(self, chunked_domain) -> bool:
        return chunked_domain[0]

    @pytest.fixture
    def domain(self, chunked_domain) -> str:
        return chunked_domain[1]

    @pytest.fixture(
        params=[
            np.float32,
            np.float64,
        ]
    )
    def dtype(self, request) -> np.dtype:
        # FP precisions to test implementation.
        # This is NOT the dtype of inputs, just sets the precision.
        return np.dtype(request.param)

    @pytest.fixture
    def op(  # we only test settings which affect accuracy, not runtime
        self,
        x_m,
        v_n,
        *,
        isign,
        # Accuracy-related ------------
        eps,
        upsampfac,
        upsampfac_ratio,
        # Runtime behavior ------------
        # fft_nthreads,
        # spread_nthreads,
        # max_cluster_size,
        # max_window_ratio,
        # Chunking behavior -----------
        chunked,
        domain,
        # max_bbox_ratio,
        # max_bbox_anisotropy,
    ) -> ftk_nu2nu.NU2NU:
        return ftk_nu2nu.NU2NU(
            x=x_m,
            v=v_n,
            isign=isign,
            eps=eps,
            upsampfac=upsampfac,
            upsampfac_ratio=upsampfac_ratio,
            chunked=chunked,
            domain=domain,
        )

    # Helper functions --------------------------------------------------------
    @staticmethod
    def _generate_A(x_m, v_n, isign) -> np.ndarray:
        # (N, M) tensor which, when inner-produced with `w(M,)`, gives `z(N,)`.
        phase = np.tensordot(v_n, x_m, axes=[[-1], [-1]])  # (N, M)
        A = np.exp(-1j * isign * 2 * np.pi * phase)
        return A
