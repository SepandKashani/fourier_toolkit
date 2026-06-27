import functools
import numpy as np
import pytest
import scipy.signal as sps

import array_api_compat as aac

import fourier_toolkit.typing as ftkt
import fourier_toolkit.util as ftku
from fourier_toolkit import u2u  # test as exposed to user
from fourier_toolkit.ufft import _CZT

from . import conftest as ct
from . import helper

parametrize_real = pytest.mark.parametrize("real", [True, False])
parametrize_stack = pytest.mark.parametrize(
    "stack_shape",
    [(), (1,), (5, 3, 4)],
)


class TestCZT:
    @parametrize_real
    @parametrize_stack
    def test_apply(self, array_backend, op, dtype, real, stack_shape):
        # output value matches ground truth.
        translate = ftku.TranslateDType(np.array([], dtype=dtype))
        fdtype = translate.to_float()
        cdtype = translate.to_complex()

        # Generate _CZT.apply() input
        rng = np.random.default_rng()
        if real:
            x = rng.standard_normal((*stack_shape, *op.cfg.N))
            x = x.astype(fdtype)
        else:
            x = 1j * rng.standard_normal((*stack_shape, *op.cfg.N))
            x += rng.standard_normal((*stack_shape, *op.cfg.N))
            x = x.astype(cdtype)

        # Generate _CZT.apply() output ground-truth
        y_gt = np.zeros((*stack_shape, *op.cfg.M), dtype=cdtype)
        C = self._generate_C(op.cfg.N, op.cfg.M, op.cfg.A, op.cfg.W).astype(cdtype)
        for idx in np.ndindex(stack_shape):
            y_gt[idx] = helper.inner_product(x[idx], C, op.cfg.D)  # (M1,...,MD)

        # Test CZT compliance
        x = ct.to_backend(x, array_backend)
        y_gt = ct.to_backend(y_gt, array_backend)
        y = op.apply(x)
        assert y.shape == y_gt.shape
        assert helper.similar(y, y_gt)
        assert_areclose(y, y_gt, op.cfg.D)

    # Fixtures ----------------------------------------------------------------
    @pytest.fixture(params=[1, 2, 3])
    def space_dim(self, request) -> int:
        return request.param

    @pytest.fixture(params=[True, False])
    def M(self, space_dim, request) -> tuple[int]:
        rng = np.random.default_rng()
        M = rng.integers(2, 7, space_dim)

        force_one = request.param
        if force_one:  # assess limiting case when one dimension is singleton.
            M[0] = 1
        return tuple(map(int, M))  # (M1,...,MD)

    @pytest.fixture(params=[True, False])
    def N(self, space_dim, request) -> tuple[int]:
        rng = np.random.default_rng()
        N = rng.integers(7, 14, space_dim)

        force_one = request.param
        if force_one:  # assess limiting case when one dimension is singleton.
            N[0] = 1
        return tuple(map(int, N))  # (N1,...,ND)

    @pytest.fixture
    def A(self, space_dim) -> tuple[complex]:
        rng = np.random.default_rng()
        A = rng.standard_normal(2 * space_dim).view(np.cdouble)
        A /= np.abs(A)
        return tuple(map(complex, A))  # (D,)

    @pytest.fixture
    def W(self, space_dim) -> tuple[complex]:
        rng = np.random.default_rng()
        W = rng.standard_normal(2 * space_dim).view(np.cdouble)
        W /= np.abs(W)
        return tuple(map(complex, W))  # (D,)

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
    def op(self, N, M, A, W) -> _CZT:
        return _CZT(N, M, A, W)

    # Helper functions --------------------------------------------------------
    @staticmethod
    def _generate_C(N, M, A, W) -> np.ndarray:
        # (M1,...,MD,N1,...,ND) tensor which, when inner-product-ed with `x(N1,...,ND)`, gives `y(M1,...,MD)`.
        C = np.zeros((*M, *N), dtype=np.cdouble)
        for n in np.ndindex(N):
            _C = np.zeros(N)
            _C[n] = 1

            D = len(N)
            for d in range(D):
                _C = sps.czt(
                    _C,
                    m=M[d],
                    w=W[d],
                    a=A[d],
                    axis=d,
                )

            C[..., *n] = _C
        return C.conj()


class TestU2U:
    # General U2U case where (x_spec, v_spec) are arbitrary.

    @parametrize_real
    @parametrize_stack
    def test_apply(
        self,
        array_backend,
        x_spec,
        v_spec,
        isign,
        dtype,
        real,
        stack_shape,
    ):
        # output value matches ground truth.
        w, z_gt = self._generate_io(x_spec, v_spec, isign, dtype, real, stack_shape)

        # Test u2u() compliance
        w = ct.to_backend(w, array_backend)
        z_gt = ct.to_backend(z_gt, array_backend)
        z = u2u(x_spec, v_spec, w, isign)
        assert z.shape == z_gt.shape
        assert helper.similar(z, z_gt)
        assert_areclose(z, z_gt, v_spec.ndim)

    @parametrize_real
    @parametrize_stack
    def test_jit(
        self,
        array_backend,
        x_spec,
        v_spec,
        isign,
        dtype,
        real,
        stack_shape,
    ):
        # can JIT u2u() calls.
        is_jax = array_backend.name.startswith("jax-")
        is_torch = array_backend.name.startswith("torch-")
        if not (is_jax or is_torch):
            pytest.skip()

        w, z_gt = self._generate_io(x_spec, v_spec, isign, dtype, real, stack_shape)

        # Test u2u() compliance
        w = ct.to_backend(w, array_backend)
        z_gt = ct.to_backend(z_gt, array_backend)

        if is_jax:
            import jax

            u2u_jit = jax.jit(u2u, static_argnums=(0, 1, 3))
        elif is_torch:
            import torch

            u2u_jit = torch.compile(u2u)
            # we don't use any torch.compile() options on purpose: user can tune those if/when needed. (If JAX tests work, then we expect the same of PyTorch.)
        else:
            raise ValueError("Unknown case encountered.")

        z = u2u_jit(x_spec, v_spec, w, isign)
        assert z.shape == z_gt.shape
        assert helper.similar(z, z_gt)
        assert_areclose(z, z_gt, v_spec.ndim)

    @parametrize_real
    def test_vmap(
        self,
        array_backend,
        x_spec,
        v_spec,
        isign,
        dtype,
        real,
    ):
        # can VMAP u2u() calls.
        is_jax = array_backend.name.startswith("jax-")
        is_torch = array_backend.name.startswith("torch-")
        if not (is_jax or is_torch):
            pytest.skip()

        stack_size = 7
        w, z_gt = self._generate_io(
            x_spec, v_spec, isign, dtype, real, stack_shape=(stack_size,)
        )

        # Test u2u() compliance
        xp = array_backend.xp
        for idx_in in range(1 + x_spec.ndim):
            for idx_out in range(1 + v_spec.ndim):
                _w = xp.moveaxis(ct.to_backend(w, array_backend), 0, idx_in)
                _z_gt = xp.moveaxis(ct.to_backend(z_gt, array_backend), 0, idx_out)

                if is_jax:
                    import jax

                    u2u_vmap = jax.vmap(
                        u2u,
                        in_axes=(None, None, idx_in, None),
                        out_axes=idx_out,
                    )
                elif is_torch:
                    import torch

                    u2u_vmap = torch.vmap(
                        u2u,
                        in_dims=(None, None, idx_in, None),
                        out_dims=idx_out,
                    )
                else:
                    raise ValueError("Unknown case encountered.")

                z = u2u_vmap(x_spec, v_spec, _w, isign)
                assert z.shape == _z_gt.shape
                assert helper.similar(z, _z_gt)
                assert_areclose(  # permutations needed to have core-dims at tail
                    xp.moveaxis(z, idx_out, 0),
                    xp.moveaxis(_z_gt, idx_out, 0),
                    v_spec.ndim,
                )

    @parametrize_real
    @parametrize_stack
    def test_jvp(
        self,
        array_backend,
        x_spec,
        v_spec,
        isign,
        dtype,
        real,
        stack_shape,
    ):
        # can JVP u2u() calls.

        # Since u2u() is linear in `w` -> jvp(u2u,w,v) == u2u(v) [w does not matter]

        is_jax = array_backend.name.startswith("jax-")
        is_torch = array_backend.name.startswith("torch-")
        if not (is_jax or is_torch):
            pytest.skip()

        primal, primal_out_gt = self._generate_io(
            x_spec, v_spec, isign, dtype, real, stack_shape
        )

        tangent, tangent_out_gt = self._generate_io(
            x_spec, v_spec, isign, dtype, real, stack_shape
        )

        # Test u2u() compliance
        primal = ct.to_backend(primal, array_backend)
        primal_out_gt = ct.to_backend(primal_out_gt, array_backend)
        tangent = ct.to_backend(tangent, array_backend)
        tangent_out_gt = ct.to_backend(tangent_out_gt, array_backend)

        _u2u = functools.partial(u2u, x_spec, v_spec, isign=isign)
        if is_jax:
            import jax

            primal_out, tangent_out = jax.jvp(_u2u, (primal,), (tangent,))
        elif is_torch:
            import torch

            primal_out, tangent_out = torch.func.jvp(_u2u, (primal,), (tangent,))
        else:
            raise ValueError("Unknown case encountered.")

        assert primal_out.shape == primal_out_gt.shape
        assert helper.similar(primal_out, primal_out_gt)
        assert_areclose(primal_out, primal_out_gt, v_spec.ndim)
        assert tangent_out.shape == tangent_out_gt.shape
        assert helper.similar(tangent_out, tangent_out_gt)
        assert_areclose(tangent_out, tangent_out_gt, v_spec.ndim)

    # Fixtures ----------------------------------------------------------------
    @pytest.fixture(params=[1, 2, 3])
    def space_dim(self, request) -> int:
        return request.param

    @pytest.fixture
    def x_spec(self, space_dim) -> ftku.UniformSpec:
        rng = np.random.default_rng()
        x0 = rng.standard_normal(space_dim)
        dx = rng.standard_normal(space_dim)
        M = rng.integers(1, 7, space_dim)
        return ftku.UniformSpec(start=x0, step=dx, num=M)

    @pytest.fixture
    def v_spec(self, space_dim) -> ftku.UniformSpec:
        rng = np.random.default_rng()
        v0 = rng.standard_normal(space_dim)
        dv = rng.standard_normal(space_dim)
        N = rng.integers(3, 12, space_dim)
        return ftku.UniformSpec(start=v0, step=dv, num=N)

    @pytest.fixture(params=[+1, -1])
    def isign(self, request) -> int:
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

    # Helper functions --------------------------------------------------------
    @staticmethod
    def _generate_A(x_spec, v_spec, isign) -> np.ndarray:
        # (N1,...,ND,M1,...,MD) tensor which, when inner-product-ed with `w(M1,...,MD)`, gives `z(N1,...,ND)`.
        x_m = x_spec.knots()  # (M1,...,MD,D)
        v_n = v_spec.knots()  # (N1,...,ND,D)
        phase = np.tensordot(v_n, x_m, axes=[[-1], [-1]])  # (N1,...,ND,M1,...,MD)
        A = np.exp(-isign * 1j * 2 * np.pi * phase)
        return A

    @classmethod
    def _generate_io(
        cls,
        x_spec: ftku.UniformSpec,
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

        # Generate u2u() input
        rng = np.random.default_rng()
        if real:
            w = rng.standard_normal((*stack_shape, *x_spec.num))
            w = w.astype(fdtype)
        else:
            w = 1j * rng.standard_normal((*stack_shape, *x_spec.num))
            w += rng.standard_normal((*stack_shape, *x_spec.num))
            w = w.astype(cdtype)

        # Generate u2u() output ground-truth
        z_gt = np.zeros((*stack_shape, *v_spec.num), dtype=cdtype)
        A = cls._generate_A(x_spec, v_spec, isign).astype(cdtype)
        for idx in np.ndindex(stack_shape):
            z_gt[idx] = helper.inner_product(w[idx], A, x_spec.ndim)  # (N1,...,ND)

        return (w, z_gt)


class TestU2USpecialCase(TestU2U):
    # Special U2U case where all (x_spec, v_spec) axes permit single-FFT algorithm.

    @pytest.fixture
    def xv_spec(self, space_dim) -> tuple[ftku.UniformSpec]:
        rng = np.random.default_rng()
        x0 = rng.standard_normal(space_dim)
        v0 = rng.standard_normal(space_dim)
        M = rng.integers(1, 7, space_dim)
        N = M.copy()
        dx = rng.uniform(1e-3, 2, space_dim)
        dv = 1 / (N * dx)
        dx *= rng.choice([-1, +1], space_dim)
        dv *= rng.choice([-1, +1], space_dim)

        x_spec = ftku.UniformSpec(start=x0, step=dx, num=M)
        v_spec = ftku.UniformSpec(start=v0, step=dv, num=N)
        return x_spec, v_spec

    @pytest.fixture
    def x_spec(self, xv_spec) -> ftku.UniformSpec:
        return xv_spec[0]

    @pytest.fixture
    def v_spec(self, xv_spec) -> ftku.UniformSpec:
        return xv_spec[1]


class TestU2UMixCase(TestU2U):
    # Special U2U case where some (x_spec, v_spec) axes permit single-FFT algorithm, and others require CZT algorithm.

    @pytest.fixture(params=[2, 3])
    def space_dim(self, request) -> int:
        return request.param

    @pytest.fixture
    def xv_spec(self, space_dim) -> tuple[ftku.UniformSpec]:
        rng = np.random.default_rng()

        # all axes work with single-FFT
        x0 = rng.standard_normal(space_dim)
        v0 = rng.standard_normal(space_dim)
        M = rng.integers(1, 7, space_dim)
        N = M.copy()
        dx = rng.uniform(1e-3, 2, space_dim)
        dv = 1 / (N * dx)
        dx *= rng.choice([-1, +1], space_dim)
        dv *= rng.choice([-1, +1], space_dim)

        # now replace some axes (dv,N) at random to use CZT
        axes = rng.integers(space_dim, size=max(1, space_dim - 1))
        for ax in axes:
            dv[ax] = rng.uniform(5, 5.5)
            N[ax] += rng.integers(2)  # randomly increase length, or not

        x_spec = ftku.UniformSpec(start=x0, step=dx, num=M)
        v_spec = ftku.UniformSpec(start=v0, step=dv, num=N)
        return x_spec, v_spec

    @pytest.fixture
    def x_spec(self, xv_spec) -> ftku.UniformSpec:
        return xv_spec[0]

    @pytest.fixture
    def v_spec(self, xv_spec) -> ftku.UniformSpec:
        return xv_spec[1]


def assert_areclose(VAL: ftkt.Array, GT: ftkt.Array, D: int):
    """
    Assess (soft) closeness of (`VAL`, `GT`).

    [Context] array backends use different math libraries. This makes `allclose()` ill-suited to compare (`VAL`, `GT`) when the code paths to generate them differ.

    `assert_areclose()` therefore uses a lax-er notion of closeness:
    ``areclose(VAL,GT) == allclose(VAL,GT) || rel_L2_close(VAL,GT)``
    """

    if helper.allclose(VAL, GT, GT.dtype):
        assert True
    else:
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
                    "Failed allclose() and rel_l2_distance() test.",
                    "[min,mean,max,std] statistics (over stack dimensions):",
                    f"- abs(VAL - GT):              ({', '.join(map(str, stats_diff))})",
                    f"- rel_linf_distance(VAL, GT): ({', '.join(map(str, stats_linf))})",
                    f"- rel_l2_distance(VAL, GT):   ({', '.join(map(str, stats_l2))})",
                    f"- eps(rel_l2_close):           {helper.fp_atol[fdtype_name]}",
                ]
            )

            assert False, diagnostic_msg
