import numpy as np
import pytest

import fourier_toolkit.nu2u as ftk_nu2u
import fourier_toolkit.util as ftk_util
import fourier_toolkit_tests.conftest as ct

# spread/interpolate() take tuple[func] as parameters -> known experimental feature.
pytestmark = pytest.mark.filterwarnings("ignore::numba.NumbaExperimentalFeatureWarning")


def _generate_NU2U_inputs() -> list:
    # Create (D, x_m, v_spec, chunked) pairs used to initialize fixtures
    data = []

    # D=1, no over/undersampling --------------------------
    D, chunked = 1, False
    rng = np.random.default_rng(0)
    x_m = ct.generate_point_cloud(
        N_point=1_00,
        D=D,
        bbox_dim=10_000 + np.arange(D),
        N_blk=100,
        sparsity_ratio=(5e-2) ** D,
        rng=rng,
    )
    v_spec = dict(
        start=rng.uniform(-1, 1, D),
        step=0.6 * (1 + np.arange(D)) / np.ptp(x_m, axis=0),  # chunk=True
        num=2_00 + np.arange(D),  # chunk=True
    )
    data.append((D, x_m, v_spec, chunked))

    # D=1, oversampling case ------------------------------
    D, chunked = 1, False
    rng = np.random.default_rng(0)
    x_m = ct.generate_point_cloud(
        N_point=1_00,
        D=D,
        bbox_dim=10_000 + np.arange(D),
        N_blk=100,
        sparsity_ratio=(5e-2) ** D,
        rng=rng,
    )
    v_spec = dict(
        start=rng.uniform(-1, 1, D),
        step=1e-1 * (1 + np.arange(D)) / np.ptp(x_m, axis=0),
        num=1_00 + np.arange(D),
    )
    data.append((D, x_m, v_spec, chunked))

    # D=1, undersampling case -----------------------------
    D, chunked = 1, False
    rng = np.random.default_rng(0)
    x_m = ct.generate_point_cloud(
        N_point=1_00,
        D=D,
        bbox_dim=10_000 + np.arange(D),
        N_blk=100,
        sparsity_ratio=(5e-2) ** D,
        rng=rng,
    )
    v_spec = dict(
        start=rng.uniform(-1, 1, D),
        step=2 * (1 + np.arange(D)) / np.ptp(x_m, axis=0),
        num=50 + (1 + np.arange(D)),
    )
    data.append((D, x_m, v_spec, chunked))

    # D=2, over/under-sampled mix -------------------------
    D, chunked = 2, False
    rng = np.random.default_rng(0)
    x_m = ct.generate_point_cloud(
        N_point=1_00,
        D=D,
        bbox_dim=10_000 + np.arange(D),
        N_blk=100,
        sparsity_ratio=(5e-2) ** D,
        rng=rng,
    )
    v_spec = dict(
        start=rng.uniform(-1, 1, D),
        step=np.r_[1e-1, 1.1] * (1 + np.arange(D)) / np.ptp(x_m, axis=0),
        num=50 + (1 + np.arange(D)),
    )
    data.append((D, x_m, v_spec, chunked))

    # D=2, chunked (over/under doesn't matter here) -------
    D, chunked = 2, True
    rng = np.random.default_rng(0)
    x_m = ct.generate_point_cloud(
        N_point=1_00,
        D=D,
        bbox_dim=10_000 + np.arange(D),
        N_blk=100,
        sparsity_ratio=(5e-2) ** D,
        rng=rng,
    )
    v_spec = dict(
        start=rng.uniform(-1, 1, D),
        step=np.r_[0.75, 1.1] * (1 + np.arange(D)) / np.ptp(x_m, axis=0),
        num=2_00 + np.arange(D),
    )
    data.append((D, x_m, v_spec, chunked))

    return data


def _mesh_from_spec(spec: dict[str, np.ndarray]) -> np.ndarray:
    # Given {start(D,), step(D,), num(D,)}, create associated (*num, D) mesh
    p0 = spec["start"]
    dp = spec["step"]
    N = spec["num"]

    mesh_axial = [_p0 + _dp * np.arange(_N) for (_p0, _dp, _N) in zip(p0, dp, N)]
    mesh = np.stack(np.meshgrid(*mesh_axial, indexing="ij"), axis=-1)  # (N1,...,ND, D)
    return mesh


class TestNU2U:
    @pytest.mark.parametrize("real", [True, False])
    @pytest.mark.parametrize("stack_shape", [(), (1,), (5, 3, 4)])
    def test_value_apply(self, op, x_m, v_spec, isign, dtype, real, stack_shape):
        # output value matches ground truth.
        translate = ftk_util.TranslateDType(dtype)
        fdtype = translate.to_float()
        cdtype = translate.to_complex()

        # Generate NU2U input
        rng = np.random.default_rng()
        if real:
            w = rng.standard_normal((*stack_shape, op.cfg.M))
            w = w.astype(fdtype)
        else:
            w = 1j * rng.standard_normal((*stack_shape, op.cfg.M))
            w += rng.standard_normal((*stack_shape, op.cfg.M))
            w = w.astype(cdtype)

        # Generate NU2U output ground-truth
        z_gt = np.zeros((*stack_shape, *op.cfg.N), dtype=cdtype)
        A = self._generate_A(x_m, v_spec, isign)  # (N1,...,ND, M)
        for idx in np.ndindex(stack_shape):
            z_gt[idx] = ct.inner_product(w[idx], A, 1)  # (N1,...,ND)

        # Test NU2U compliance
        z = op.apply(w)
        assert z.shape == z_gt.shape
        assert ct.relclose(z, z_gt, D=op.cfg.D, eps=1e-3)  # todo: progressive eps?

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
        y = 1j * rng.standard_normal((*sh, *op.cfg.N))
        y += rng.standard_normal((*sh, *op.cfg.N))
        y = y.astype(cdtype)

        lhs = ct.inner_product(op.apply(x), y, D=op.cfg.D)
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

    @pytest.fixture(params=_generate_NU2U_inputs())
    def D_xM_vspec_chunked(self, request) -> tuple:
        D, x_m, v_spec, chunked = request.param
        return D, x_m, v_spec, chunked

    @pytest.fixture
    def space_dim(self, D_xM_vspec_chunked) -> int:
        D, x_m, v_spec, chunked = D_xM_vspec_chunked
        return D

    @pytest.fixture
    def x_m(self, D_xM_vspec_chunked) -> np.ndarray:
        D, x_m, v_spec, chunked = D_xM_vspec_chunked
        return x_m

    @pytest.fixture
    def v_spec(self, D_xM_vspec_chunked) -> dict[str, np.ndarray]:
        D, x_m, v_spec, chunked = D_xM_vspec_chunked
        return v_spec

    @pytest.fixture
    def chunked(self, D_xM_vspec_chunked) -> bool:
        D, x_m, v_spec, chunked = D_xM_vspec_chunked
        return chunked

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
        upsampfac_ratio,
        # Runtime behavior ------------
        # fft_nthreads,
        # spread_nthreads,
        # max_cluster_size,
        # max_window_ratio,
        # Chunking behavior -----------
        chunked,
        # max_bbox_ratio,
        # max_bbox_anisotropy,
    ) -> ftk_nu2u.NU2U:
        op = ftk_nu2u.NU2U(
            x=x_m,
            v_spec=v_spec,
            isign=isign,
            eps=eps,
            upsampfac=upsampfac,
            upsampfac_ratio=upsampfac_ratio,
            chunked=chunked,
        )

        # Make sure chunking is doing what is expected
        if chunked:
            assert op.cfg.Px > 1
            assert op.cfg.Pv == 1

        return op

    # Helper functions --------------------------------------------------------
    @staticmethod
    def _generate_A(x_m, v_spec, isign) -> np.ndarray:
        # (N1,...,ND, M) tensor which, when inner-produced with `w(M,)`, gives `z(N1,...,ND,)`.
        v_n = _mesh_from_spec(v_spec)  # (N1,...,ND, D)

        phase = np.tensordot(v_n, x_m, axes=[[-1], [-1]])  # (N1,...,ND, M)
        A = np.exp(-1j * isign * 2 * np.pi * phase)
        return A
