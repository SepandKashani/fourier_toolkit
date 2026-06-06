import numpy as np
import pytest
import scipy.signal as sps

import fourier_toolkit.typing as ftkt
import fourier_toolkit.util as ftku
from fourier_toolkit import u2u  # test as exposed to user
from fourier_toolkit.ufft import _CZT

from . import conftest as ct
from . import helper


class TestCZT:
    @staticmethod
    def to_backend(x: ftkt.Array, array_backend: ct.ArrayBackend) -> ftkt.Array:
        return array_backend.xp.asarray(
            x,
            device=array_backend.device,
        )

    @pytest.mark.parametrize("real", [True, False])
    @pytest.mark.parametrize("stack_shape", [(), (1,), (5, 3, 4)])
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
        x = self.to_backend(x, array_backend)
        y_gt = self.to_backend(y_gt, array_backend)
        y = op.apply(x)
        assert y.shape == y_gt.shape
        assert helper.similar(y, y_gt)
        try:
            assert helper.allclose(y, y_gt, y_gt.dtype)
        except AssertionError:
            xp = array_backend.xp
            info = xp.__array_namespace_info__()
            dtypes = {v: k for (k, v) in info.dtypes().items()}
            fdtype_name = dtypes[y_gt.dtype]

            if helper.rel_l2_close(y, y_gt, op.cfg.D, eps=helper.fp_atol[fdtype_name]):
                assert True
            else:
                # fail test, showing statistics
                diff = xp.abs(y - y_gt)
                linf = helper.rel_linf_distance(y, y_gt, op.cfg.D)
                l2 = helper.rel_l2_distance(y, y_gt, op.cfg.D)

                stat_funcs = (xp.min, xp.mean, xp.max, xp.std)
                stats_diff = [float(f(diff)) for f in stat_funcs]
                stats_linf = [float(f(linf)) for f in stat_funcs]
                stats_l2 = [float(f(l2)) for f in stat_funcs]

                diagnostic_msg = "\n".join(
                    [
                        "Failed allclose() and rel_l2_distance() test.",
                        "[min,mean,max,std] statistics (over stack dimensions):",
                        f"- abs(y - y_gt):              ({', '.join(map(str, stats_diff))})",
                        f"- rel_linf_distance(y, y_gt): ({', '.join(map(str, stats_linf))})",
                        f"- rel_l2_distance(y, y_gt):   ({', '.join(map(str, stats_l2))})",
                        f"- eps(rel_l2_close):           {helper.fp_atol[fdtype_name]}",
                    ]
                )

                assert False, diagnostic_msg

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

    @pytest.mark.parametrize("real", [True, False])
    @pytest.mark.parametrize("stack_shape", [(), (1,), (5, 3, 4)])
    def test_value_apply(self, x_spec, v_spec, isign, dtype, real, stack_shape):
        # output value matches ground truth.
        translate = ftku.TranslateDType(dtype)
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
        A = self._generate_A(x_spec, v_spec, isign)  # (N1,...,ND,M1,...,MD)
        for idx in np.ndindex(stack_shape):
            z_gt[idx] = ct.inner_product(w[idx], A, x_spec.ndim)  # (N1,...,ND)

        # Test u2u() compliance
        z = u2u(x_spec, v_spec, w, isign)
        assert z.shape == z_gt.shape
        assert ct.allclose(z, z_gt, fdtype)

    @pytest.mark.parametrize("real", [True, False])
    def test_prec(self, x_spec, v_spec, isign, dtype, real):
        # output precision (not dtype!) matches input precision.
        translate = ftku.TranslateDType(dtype)
        fdtype = translate.to_float()
        cdtype = translate.to_complex()

        rng = np.random.default_rng()
        if real:
            w = rng.standard_normal(x_spec.num)
            w = w.astype(fdtype)
        else:
            w = 1j * rng.standard_normal(x_spec.num)
            w += rng.standard_normal(x_spec.num)
            w = w.astype(cdtype)

        z = u2u(x_spec, v_spec, w, isign)
        assert z.dtype == cdtype

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

        x_spec = ftku.UniformSpec(x0, dx, M)
        v_spec = ftku.UniformSpec(v0, dv, N)
        return x_spec, v_spec

    @pytest.fixture
    def x_spec(self, xv_spec) -> ftku.UniformSpec:
        return xv_spec[0]

    @pytest.fixture
    def v_spec(self, xv_spec) -> ftku.UniformSpec:
        return xv_spec[1]


class TestU2UMixCase(TestU2U):
    # Special U2U case where some (x_spec, v_spec) axes permit single-FFT algorithm, and others require CZT algorithm.

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

        x_spec = ftku.UniformSpec(x0, dx, M)
        v_spec = ftku.UniformSpec(v0, dv, N)
        return x_spec, v_spec

    @pytest.fixture
    def x_spec(self, xv_spec) -> ftku.UniformSpec:
        return xv_spec[0]

    @pytest.fixture
    def v_spec(self, xv_spec) -> ftku.UniformSpec:
        return xv_spec[1]
