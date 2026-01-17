import numpy as np
import pytest
import scipy.signal as sps

import fourier_toolkit.util as ftku
from fourier_toolkit import u2u  # test as exposed to user
from fourier_toolkit.ufft import CZT, DFT

from . import conftest as ct


class TestDFT:
    @pytest.mark.parametrize("real", [True, False])
    @pytest.mark.parametrize("stack_shape", [(), (1,), (5, 3, 4)])
    def test_value_apply(self, op, dtype, real, stack_shape):
        # output value matches ground truth.
        translate = ftku.TranslateDType(dtype)
        fdtype = translate.to_float()
        cdtype = translate.to_complex()

        # Generate DFT input
        rng = np.random.default_rng()
        size = rng.integers(3, 17, op.cfg.D)
        if real:
            x = rng.standard_normal((*stack_shape, *size))
            x = x.astype(fdtype)
        else:
            x = 1j * rng.standard_normal((*stack_shape, *size))
            x += rng.standard_normal((*stack_shape, *size))
            x = x.astype(cdtype)

        # Generate DFT output ground-truth
        y_gt = x.copy()
        for d in range(op.cfg.D):
            y_gt = np.fft.fft(
                y_gt,
                axis=len(stack_shape) + d,
            )

        # Test DFT compliance
        y = op.apply(x)
        assert y.shape == y_gt.shape
        assert ct.allclose(y, y_gt, fdtype)

    @pytest.mark.parametrize("real", [True, False])
    @pytest.mark.parametrize("direction", ["apply", "adjoint", "inverse"])
    def test_prec(self, op, dtype, real, direction):
        # output precision (not dtype!) matches input precision.
        translate = ftku.TranslateDType(dtype)
        fdtype = translate.to_float()
        cdtype = translate.to_complex()

        rng = np.random.default_rng()
        size = rng.integers(3, 17, op.cfg.D)
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
        translate = ftku.TranslateDType(dtype)
        cdtype = translate.to_complex()

        sh = (5, 3, 4)
        rng = np.random.default_rng()
        size = rng.integers(3, 17, op.cfg.D)
        x = 1j * rng.standard_normal((*sh, *size))
        x += rng.standard_normal((*sh, *size))
        x = x.astype(cdtype)
        y = 1j * rng.standard_normal((*sh, *size))
        y += rng.standard_normal((*sh, *size))
        y = y.astype(cdtype)

        lhs = ct.inner_product(op.apply(x), y, op.cfg.D)
        rhs = ct.inner_product(x, op.adjoint(y), op.cfg.D)
        # apply/adjoint use different code paths, so comparing wrt rel-error
        assert ct.relclose(lhs, rhs, len(sh), 1e-5)

    def test_math_inverse(self, op, dtype):
        # A^{-1} A x == x
        translate = ftku.TranslateDType(dtype)
        cdtype = translate.to_complex()

        sh = (5, 3, 4)
        rng = np.random.default_rng()
        size = rng.integers(3, 17, op.cfg.D)
        x = 1j * rng.standard_normal((*sh, *size))
        x += rng.standard_normal((*sh, *size))
        x = x.astype(cdtype)

        y1 = op.inverse(op.apply(x))
        y2 = op.apply(op.inverse(x))
        assert ct.relclose(y1, x, op.cfg.D, 1e-5)
        assert ct.relclose(y2, x, op.cfg.D, eps=1e-5)

    # Fixtures ----------------------------------------------------------------
    @pytest.fixture(params=[1, 2, 3])
    def space_dim(self, request) -> int:
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

    @pytest.fixture
    def op(self, space_dim) -> DFT:
        return DFT(space_dim)

    # Helper functions --------------------------------------------------------


class TestCZT:
    @pytest.mark.parametrize("real", [True, False])
    @pytest.mark.parametrize("stack_shape", [(), (1,), (5, 3, 4)])
    def test_value_apply(self, op, dtype, real, stack_shape):
        # output value matches ground truth.
        translate = ftku.TranslateDType(dtype)
        fdtype = translate.to_float()
        cdtype = translate.to_complex()

        # Generate CZT input
        rng = np.random.default_rng()
        if real:
            x = rng.standard_normal((*stack_shape, *op.cfg.N))
            x = x.astype(fdtype)
        else:
            x = 1j * rng.standard_normal((*stack_shape, *op.cfg.N))
            x += rng.standard_normal((*stack_shape, *op.cfg.N))
            x = x.astype(cdtype)

        # Generate CZT output ground-truth
        y_gt = x.copy()
        for d in range(op.cfg.D):
            y_gt = sps.czt(
                y_gt,
                m=op.cfg.M[d],
                w=op.cfg.W[d],
                a=op.cfg.A[d],
                axis=len(stack_shape) + d,
            )

        # Test CZT compliance
        y = op.apply(x)
        assert y.shape == y_gt.shape
        assert ct.allclose(y, y_gt, fdtype)

    @pytest.mark.parametrize("real", [True, False])
    @pytest.mark.parametrize("direction", ["apply", "adjoint"])
    def test_prec(self, op, dtype, real, direction):
        # output precision (not dtype!) matches input precision.
        translate = ftku.TranslateDType(dtype)
        fdtype = translate.to_float()
        cdtype = translate.to_complex()

        rng = np.random.default_rng()
        size = op.cfg.N if (direction == "apply") else op.cfg.M
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
        translate = ftku.TranslateDType(dtype)
        cdtype = translate.to_complex()

        sh = (5, 3, 4)
        rng = np.random.default_rng()
        x = 1j * rng.standard_normal((*sh, *op.cfg.N))
        x += rng.standard_normal((*sh, *op.cfg.N))
        x = x.astype(cdtype)
        y = 1j * rng.standard_normal((*sh, *op.cfg.M))
        y += rng.standard_normal((*sh, *op.cfg.M))
        y = y.astype(cdtype)

        lhs = ct.inner_product(op.apply(x), y, op.cfg.D)
        rhs = ct.inner_product(x, op.adjoint(y), op.cfg.D)
        # apply/adjoint use different code paths, so comparing wrt rel-error
        assert ct.relclose(lhs, rhs, len(sh), 1e-5)

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
    def op(self, N, M, A, W) -> CZT:
        return CZT(N, M, A, W)

    # Helper functions --------------------------------------------------------


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
        dx = rng.uniform(1e-3, 2, space_dim)
        M = rng.integers(1, 7, space_dim)
        return ftku.UniformSpec(start=x0, step=dx, num=M)

    @pytest.fixture
    def v_spec(self, space_dim) -> ftku.UniformSpec:
        rng = np.random.default_rng()
        v0 = rng.standard_normal(space_dim)
        dv = rng.uniform(1e-2, 5, space_dim)
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
