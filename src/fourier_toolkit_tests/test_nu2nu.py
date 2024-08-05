import numpy as np
import pytest

import fourier_toolkit.nu2nu as ftk_nu2nu
import fourier_toolkit.util as ftk_util
import fourier_toolkit_tests.conftest as ct

# spread/interpolate() take tuple[func] as parameters -> known experimental feature.
pytestmark = pytest.mark.filterwarnings("ignore::numba.NumbaExperimentalFeatureWarning")


def _generate_NU2NU_inputs() -> list:
    # Create (D, x_m, v_n, chunked, domain) pairs used to initialize fixtures
    data = []

    # D=1, no chunk ---------------------------------------
    D, chunked, domain = 1, False, "xv"
    rng = np.random.default_rng(0)
    x_m = ct.generate_point_cloud(
        N_point=250,
        D=D,
        bbox_dim=2,
        N_blk=100,
        sparsity_ratio=5e-2,
        rng=rng,
    )
    v_n = ct.generate_point_cloud(
        N_point=251,
        D=D,
        bbox_dim=2.1,
        N_blk=100,
        sparsity_ratio=5e-2,
        rng=rng,
    )
    data.append((D, x_m, v_n, chunked, domain))

    # D=1, chunk x ----------------------------------------
    D, chunked, domain = 1, True, "x"
    rng = np.random.default_rng(0)
    x_m = ct.generate_point_cloud(
        N_point=250,
        D=D,
        bbox_dim=200,
        N_blk=100,
        sparsity_ratio=5e-2,
        rng=rng,
    )
    v_n = ct.generate_point_cloud(
        N_point=251,
        D=D,
        bbox_dim=21,
        N_blk=100,
        sparsity_ratio=5e-2,
        rng=rng,
    )
    data.append((D, x_m, v_n, chunked, domain))

    # D=1, chunk v ----------------------------------------
    D, chunked, domain = 1, True, "v"
    rng = np.random.default_rng(0)
    x_m = ct.generate_point_cloud(
        N_point=250,
        D=D,
        bbox_dim=200,
        N_blk=100,
        sparsity_ratio=5e-2,
        rng=rng,
    )
    v_n = ct.generate_point_cloud(
        N_point=251,
        D=D,
        bbox_dim=21,
        N_blk=100,
        sparsity_ratio=5e-2,
        rng=rng,
    )
    data.append((D, x_m, v_n, chunked, domain))

    # D=1, chunk xv ---------------------------------------
    D, chunked, domain = 1, True, "xv"
    rng = np.random.default_rng(0)
    x_m = ct.generate_point_cloud(
        N_point=250,
        D=D,
        bbox_dim=200,
        N_blk=100,
        sparsity_ratio=5e-2,
        rng=rng,
    )
    v_n = ct.generate_point_cloud(
        N_point=251,
        D=D,
        bbox_dim=201,
        N_blk=100,
        sparsity_ratio=5e-2,
        rng=rng,
    )
    data.append((D, x_m, v_n, chunked, domain))

    # D=2, no chunk ---------------------------------------
    D, chunked, domain = 2, False, "xv"
    rng = np.random.default_rng(0)
    x_m = ct.generate_point_cloud(
        N_point=250,
        D=D,
        bbox_dim=2,
        N_blk=100,
        sparsity_ratio=(5e-2) ** D,
        rng=rng,
    )
    v_n = ct.generate_point_cloud(
        N_point=251,
        D=D,
        bbox_dim=2.1,
        N_blk=100,
        sparsity_ratio=(5e-2) ** D,
        rng=rng,
    )
    data.append((D, x_m, v_n, chunked, domain))

    # D=2, chunk x ----------------------------------------
    D, chunked, domain = 2, True, "x"
    rng = np.random.default_rng(0)
    x_m = ct.generate_point_cloud(
        N_point=250,
        D=D,
        bbox_dim=[250, 251],
        N_blk=100,
        sparsity_ratio=(5e-2) ** D,
        rng=rng,
    )
    v_n = ct.generate_point_cloud(
        N_point=251,
        D=D,
        bbox_dim=1,
        N_blk=100,
        sparsity_ratio=(5e-2) ** D,
        rng=rng,
    )
    data.append((D, x_m, v_n, chunked, domain))

    # D=2, chunk v ----------------------------------------
    D, chunked, domain = 2, True, "v"
    rng = np.random.default_rng(0)
    x_m = ct.generate_point_cloud(
        N_point=250,
        D=D,
        bbox_dim=[250, 251],
        N_blk=100,
        sparsity_ratio=(5e-2) ** D,
        rng=rng,
    )
    v_n = ct.generate_point_cloud(
        N_point=251,
        D=D,
        bbox_dim=1,
        N_blk=100,
        sparsity_ratio=(5e-2) ** D,
        rng=rng,
    )
    data.append((D, x_m, v_n, chunked, domain))

    # D=2, chunk xv ---------------------------------------
    D, chunked, domain = 2, True, "xv"
    rng = np.random.default_rng(0)
    x_m = ct.generate_point_cloud(
        N_point=250,
        D=D,
        bbox_dim=[10, 11],
        N_blk=100,
        sparsity_ratio=(5e-2) ** D,
        rng=rng,
    )
    v_n = ct.generate_point_cloud(
        N_point=251,
        D=D,
        bbox_dim=[0.1, 130],
        N_blk=100,
        sparsity_ratio=(5e-2) ** D,
        rng=rng,
    )
    data.append((D, x_m, v_n, chunked, domain))

    return data


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
            np.float32,
            np.float64,
        ]
    )
    def dtype(self, request) -> np.dtype:
        # FP precisions to test implementation.
        # This is NOT the dtype of inputs, just sets the precision.
        return np.dtype(request.param)

    @pytest.fixture(params=_generate_NU2NU_inputs())
    def D_xM_vN_chunked_domain(self, request) -> tuple:
        D, x_m, v_n, chunked, domain = request.param
        return (D, x_m, v_n, chunked, domain)

    @pytest.fixture
    def space_dim(self, D_xM_vN_chunked_domain) -> int:
        D, x_m, v_n, chunked, domain = D_xM_vN_chunked_domain
        return D

    @pytest.fixture
    def x_m(self, D_xM_vN_chunked_domain) -> int:
        D, x_m, v_n, chunked, domain = D_xM_vN_chunked_domain
        return x_m

    @pytest.fixture
    def v_n(self, D_xM_vN_chunked_domain) -> int:
        D, x_m, v_n, chunked, domain = D_xM_vN_chunked_domain
        return v_n

    @pytest.fixture
    def chunked(self, D_xM_vN_chunked_domain) -> int:
        D, x_m, v_n, chunked, domain = D_xM_vN_chunked_domain
        return chunked

    @pytest.fixture
    def domain(self, D_xM_vN_chunked_domain) -> int:
        D, x_m, v_n, chunked, domain = D_xM_vN_chunked_domain
        return domain

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
        op = ftk_nu2nu.NU2NU(
            x=x_m,
            v=v_n,
            isign=isign,
            eps=eps,
            upsampfac=upsampfac,
            upsampfac_ratio=upsampfac_ratio,
            chunked=chunked,
            domain=domain,
        )

        # Make sure chunking is doing what is expected
        if chunked:
            if domain == "x":
                assert op.cfg.Px > 1
                assert op.cfg.Pv == 1
            elif domain == "v":
                assert op.cfg.Px == 1
                assert op.cfg.Pv > 1
            elif domain == "xv":
                assert op.cfg.Px > 1
                assert op.cfg.Pv > 1
            else:
                assert False

        return op

    # Helper functions --------------------------------------------------------
    @staticmethod
    def _generate_A(x_m, v_n, isign) -> np.ndarray:
        # (N, M) tensor which, when inner-produced with `w(M,)`, gives `z(N,)`.
        phase = np.tensordot(v_n, x_m, axes=[[-1], [-1]])  # (N, M)
        A = np.exp(-1j * isign * 2 * np.pi * phase)
        return A
