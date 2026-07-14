import array_api_compat as aac
import numpy as np
import pytest

import fourier_toolkit.typing as ftkt
import fourier_toolkit.util as ftku
from fourier_toolkit import nu2nu, nu2u, u2nu  # test as exposed to user

from . import conftest as ct
from . import helper

parametrize_real = pytest.mark.parametrize("real", [True, False])
parametrize_stack = pytest.mark.parametrize(
    "stack_shape",
    [(), (1,), (5, 3, 4)],
)


class TestNU2NU:
    @parametrize_real
    @parametrize_stack
    def test_apply(
        self,
        array_backend,
        x_m,
        v_n,
        isign,
        finufft_kwargs,
        dtype,
        real,
        stack_shape,
    ):
        # output value matches ground truth.
        w, z_gt = self._generate_io(x_m, v_n, isign, dtype, real, stack_shape)

        # Test nu2nu() compliance
        xp = array_backend.xp
        w = ct.to_backend(w, array_backend)
        z_gt = ct.to_backend(z_gt, array_backend)
        fdtype = ftku.TranslateDType(w).to_float()
        x_m = xp.astype(ct.to_backend(x_m, array_backend), fdtype)
        v_n = xp.astype(ct.to_backend(v_n, array_backend), fdtype)
        z = nu2nu(
            x=x_m,
            v=v_n,
            w=w,
            isign=isign,
            **finufft_kwargs,
        )
        assert z.shape == z_gt.shape
        assert helper.similar(z, z_gt)
        assert_areclose(z, z_gt, 1)

    # Fixtures ----------------------------------------------------------------
    @pytest.fixture(params=[1, 2, 3])
    def space_dim(self, request) -> int:
        return request.param

    @pytest.fixture
    def x_m(self, space_dim) -> np.ndarray:
        rng = np.random.default_rng()
        M = 50
        x = rng.uniform(0, 5, (M, space_dim))
        x *= rng.uniform(size=space_dim)  # to have different spreads in each direction
        return x

    @pytest.fixture
    def v_n(self, space_dim) -> np.ndarray:
        rng = np.random.default_rng()
        N = 51
        v = rng.uniform(-3, -2, (N, space_dim))
        v *= rng.uniform(size=space_dim)  # to have different spreads in each direction
        return v

    @pytest.fixture(params=[+1, -1])
    def isign(self, request) -> int:
        return request.param

    @pytest.fixture
    def finufft_kwargs(self) -> dict:
        return dict(
            eps=1e-14,  # actual precision assessed by assert_areclose()
            upsampfac=2,  # to actually reach `eps` precision for type-3
        )

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

    # Helper functions --------------------------------------------------------
    @staticmethod
    def _generate_A(x_m, v_n, isign) -> np.ndarray:
        # (N, M) tensor which, when inner-produced with `w(M,)`, gives `z(N,)`.
        phase = np.tensordot(v_n, x_m, axes=[[-1], [-1]])  # (N, M)
        A = np.exp(-isign * 1j * 2 * np.pi * phase)
        return A

    @classmethod
    def _generate_io(
        cls,
        x_m: np.ndarray,
        v_n: np.ndarray,
        isign: int,
        dtype: np.dtype,
        real: bool,
        stack_shape: tuple[int, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        # (w, z_gt) arrays used for tests
        translate = ftku.TranslateDType(np.array([], dtype=dtype))
        fdtype = translate.to_float()
        cdtype = translate.to_complex()

        # Generate nu2nu() input
        M, _ = x_m.shape
        rng = np.random.default_rng()
        if real:
            w = rng.standard_normal((*stack_shape, M))
            w = w.astype(fdtype)
        else:
            w = 1j * rng.standard_normal((*stack_shape, M))
            w += rng.standard_normal((*stack_shape, M))
            w = w.astype(cdtype)

        # Generate nu2nu() output ground-truth
        N, _ = v_n.shape
        z_gt = np.zeros((*stack_shape, N), dtype=cdtype)
        A = cls._generate_A(x_m, v_n, isign).astype(cdtype)  # (N, M)
        for idx in np.ndindex(stack_shape):
            z_gt[idx] = helper.inner_product(w[idx], A, 1)  # (N,)

        return (w, z_gt)


class TestNU2U:
    @parametrize_real
    @parametrize_stack
    def test_apply(
        self,
        array_backend,
        x_m,
        v_spec,
        isign,
        finufft_kwargs,
        dtype,
        real,
        stack_shape,
    ):
        # output value matches ground truth.
        w, z_gt = self._generate_io(x_m, v_spec, isign, dtype, real, stack_shape)

        # Test nu2u() compliance
        xp = array_backend.xp
        w = ct.to_backend(w, array_backend)
        z_gt = ct.to_backend(z_gt, array_backend)
        fdtype = ftku.TranslateDType(w).to_float()
        x_m = xp.astype(ct.to_backend(x_m, array_backend), fdtype)
        z = nu2u(
            x=x_m,
            v_spec=v_spec,
            w=w,
            isign=isign,
            **finufft_kwargs,
        )
        assert z.shape == z_gt.shape
        assert helper.similar(z, z_gt)
        assert_areclose(z, z_gt, v_spec.ndim)

    # Fixtures ----------------------------------------------------------------
    @pytest.fixture(params=[1, 2, 3])
    def space_dim(self, request) -> int:
        return request.param

    @pytest.fixture
    def x_m(self, space_dim) -> np.ndarray:
        rng = np.random.default_rng()
        M = 50
        x = rng.uniform(0, 5, (M, space_dim))
        x *= rng.uniform(size=space_dim)  # to have different spreads in each direction
        return x

    @pytest.fixture
    def v_spec(self, space_dim) -> ftku.UniformSpec:
        rng = np.random.default_rng()
        v0 = rng.uniform(-3, -2, space_dim)
        dv = rng.uniform(1e-2, 1.5, space_dim)
        dv *= rng.choice([-1, +1], space_dim)
        N = rng.integers(3, 12, space_dim)
        return ftku.UniformSpec(start=v0, step=dv, num=N)

    @pytest.fixture(params=[+1, -1])
    def isign(self, request) -> int:
        return request.param

    @pytest.fixture
    def finufft_kwargs(self) -> dict:
        return dict(
            eps=1e-14,  # actual precision assessed by assert_areclose()
            upsampfac=2,  # to actually reach `eps` precision for type-3
        )

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

    # Helper functions --------------------------------------------------------
    @staticmethod
    def _generate_A(x_m, v_spec, isign) -> np.ndarray:
        # (N1,...,ND, M) tensor which, when inner-produced with `w(M,)`, gives `z(N1,...,ND)`.
        phase = np.tensordot(v_spec.knots(), x_m, axes=[[-1], [-1]])  # (N1,...,ND, M)
        A = np.exp(-isign * 1j * 2 * np.pi * phase)
        return A

    @classmethod
    def _generate_io(
        cls,
        x_m: np.ndarray,
        v_spec: ftku.UniformSpec,
        isign: int,
        dtype: np.dtype,
        real: bool,
        stack_shape: tuple[int, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        # (w, z_gt) arrays used for tests
        translate = ftku.TranslateDType(np.array([], dtype=dtype))
        fdtype = translate.to_float()
        cdtype = translate.to_complex()

        # Generate nu2u() input
        M, _ = x_m.shape
        rng = np.random.default_rng()
        if real:
            w = rng.standard_normal((*stack_shape, M))
            w = w.astype(fdtype)
        else:
            w = 1j * rng.standard_normal((*stack_shape, M))
            w += rng.standard_normal((*stack_shape, M))
            w = w.astype(cdtype)

        # Generate nu2u() output ground-truth
        z_gt = np.zeros((*stack_shape, *v_spec.num), dtype=cdtype)
        A = cls._generate_A(x_m, v_spec, isign).astype(cdtype)  # (N1,...,ND, M)
        for idx in np.ndindex(stack_shape):
            z_gt[idx] = helper.inner_product(w[idx], A, 1)  # (N1,...,ND)

        return (w, z_gt)


class TestU2NU:
    @parametrize_real
    @parametrize_stack
    def test_apply(
        self,
        array_backend,
        x_spec,
        v_n,
        isign,
        finufft_kwargs,
        dtype,
        real,
        stack_shape,
    ):
        # output value matches ground truth.
        w, z_gt = self._generate_io(x_spec, v_n, isign, dtype, real, stack_shape)

        # Test u2nu() compliance
        xp = array_backend.xp
        w = ct.to_backend(w, array_backend)
        z_gt = ct.to_backend(z_gt, array_backend)
        fdtype = ftku.TranslateDType(w).to_float()
        v_n = xp.astype(ct.to_backend(v_n, array_backend), fdtype)
        z = u2nu(
            x_spec=x_spec,
            v=v_n,
            w=w,
            isign=isign,
            **finufft_kwargs,
        )
        assert z.shape == z_gt.shape
        assert helper.similar(z, z_gt)
        assert_areclose(z, z_gt, 1)

    # Fixtures ----------------------------------------------------------------
    @pytest.fixture(params=[1, 2, 3])
    def space_dim(self, request) -> int:
        return request.param

    @pytest.fixture
    def x_spec(self, space_dim) -> ftku.UniformSpec:
        rng = np.random.default_rng()
        x0 = rng.uniform(-3, -2, space_dim)
        dx = rng.uniform(1e-2, 1.5, space_dim)
        dx *= rng.choice([-1, +1], space_dim)
        M = rng.integers(3, 12, space_dim)
        return ftku.UniformSpec(start=x0, step=dx, num=M)

    @pytest.fixture
    def v_n(self, space_dim) -> np.ndarray:
        rng = np.random.default_rng()
        N = 50
        v = rng.uniform(0, 5, (N, space_dim))
        v *= rng.uniform(size=space_dim)  # to have different spreads in each direction
        return v

    @pytest.fixture(params=[+1, -1])
    def isign(self, request) -> int:
        return request.param

    @pytest.fixture
    def finufft_kwargs(self) -> dict:
        return dict(
            eps=1e-14,  # actual precision assessed by assert_areclose()
            upsampfac=2,  # to actually reach `eps` precision for type-3
        )

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

    # Helper functions --------------------------------------------------------
    @staticmethod
    def _generate_A(x_spec, v_n, isign) -> np.ndarray:
        # (N, M1,...,MD) tensor which, when inner-produced with `w(M1,...,MD)`, gives `z(N,)`.
        phase = np.tensordot(v_n, x_spec.knots(), axes=[[-1], [-1]])  # (N, M1,...,MD)
        A = np.exp(-isign * 1j * 2 * np.pi * phase)
        return A

    @classmethod
    def _generate_io(
        cls,
        x_spec: ftku.UniformSpec,
        v_n: np.ndarray,
        isign: int,
        dtype: np.dtype,
        real: bool,
        stack_shape: tuple[int, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        # (w, z_gt) arrays used for tests
        translate = ftku.TranslateDType(np.array([], dtype=dtype))
        fdtype = translate.to_float()
        cdtype = translate.to_complex()

        # Generate u2nu() input
        rng = np.random.default_rng()
        if real:
            w = rng.standard_normal((*stack_shape, *x_spec.num))
            w = w.astype(fdtype)
        else:
            w = 1j * rng.standard_normal((*stack_shape, *x_spec.num))
            w += rng.standard_normal((*stack_shape, *x_spec.num))
            w = w.astype(cdtype)

        # Generate u2nu() output ground-truth
        N, _ = v_n.shape
        z_gt = np.zeros((*stack_shape, N), dtype=cdtype)
        A = cls._generate_A(x_spec, v_n, isign).astype(cdtype)  # (N, M1,...,MD)
        for idx in np.ndindex(stack_shape):
            z_gt[idx] = helper.inner_product(w[idx], A, x_spec.ndim)  # (N,)

        return (w, z_gt)


def assert_areclose(VAL: ftkt.Array, GT: ftkt.Array, D: int):
    """
    Assess (soft) closeness of (`VAL`, `GT`).

    [Context] NUFFTs are `eps`-approximate transforms. This makes `allclose()` ill-suited to compare (`VAL`, `GT`).

    `assert_areclose()` therefore uses a lax-er notion of closeness:
    ``areclose(VAL,GT) == rel_L2_close(VAL,GT)``

    Note that we deliberately do not check correctness at the desired NUFFT precision `eps` since quite brittle. As far as we are concerned in FTK, numerical accuracy as defined in `fp_atol` is sufficient.
    """
    xp = aac.array_namespace(GT)
    info = xp.__array_namespace_info__()
    dtypes = {v: k for (k, v) in info.dtypes().items()}
    fdtype_name = dtypes[GT.dtype]

    if helper.rel_l2_close(VAL, GT, D, eps=helper.fp_atol[fdtype_name]):
        assert True
    else:
        # fail test, showing statistics
        diff = xp.abs(VAL - GT)
        linf = helper.rel_linf_distance(VAL, GT, D)
        l2 = helper.rel_l2_distance(VAL, GT, D)

        stat_funcs = (xp.min, xp.mean, xp.max, xp.std)
        stats_diff = [float(f(diff)) for f in stat_funcs]
        stats_linf = [float(f(linf)) for f in stat_funcs]
        stats_l2 = [float(f(l2)) for f in stat_funcs]

        diagnostic_msg = "\n".join(
            [
                "Failed rel_l2_distance() test.",
                "[min,mean,max,std] statistics (over stack dimensions):",
                f"- abs(VAL - GT):              ({', '.join(map(str, stats_diff))})",
                f"- rel_linf_distance(VAL, GT): ({', '.join(map(str, stats_linf))})",
                f"- rel_l2_distance(VAL, GT):   ({', '.join(map(str, stats_l2))})",
                f"- eps(rel_l2_close):           {helper.fp_atol[fdtype_name]}",
            ]
        )

        assert False, diagnostic_msg
