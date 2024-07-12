import numpy as np
import pytest
import scipy.signal as sps

import fourier_toolkit.czt as ftk_czt
import fourier_toolkit.util as ftk_util
import fourier_toolkit_tests.conftest as ct


class TestCZT:
    @pytest.mark.parametrize("real", [True, False])
    @pytest.mark.parametrize("stack_shape", [(), (1,), (5, 3, 4)])
    def test_value_apply(self, op, dtype, real, stack_shape):
        # output value matches ground truth.
        translate = ftk_util.TranslateDType(dtype)
        fdtype = translate.to_float()
        cdtype = translate.to_complex()

        # Generate CZT input
        rng = np.random.default_rng()
        if real:
            x = rng.standard_normal((*stack_shape, *op._N))
            x = x.astype(fdtype)
        else:
            x = 1j * rng.standard_normal((*stack_shape, *op._N))
            x += rng.standard_normal((*stack_shape, *op._N))
            x = x.astype(cdtype)

        # Generate CZT output ground-truth
        y_gt = x.copy()
        for d in range(op._D):
            y_gt = sps.czt(
                y_gt,
                m=op._M[d],
                w=op._W[d],
                a=op._A[d],
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
        translate = ftk_util.TranslateDType(dtype)
        fdtype = translate.to_float()
        cdtype = translate.to_complex()

        rng = np.random.default_rng()
        size = op._N if (direction == "apply") else op._M
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
        fdtype = translate.to_float()
        cdtype = translate.to_complex()

        sh = (5, 3, 4)
        rng = np.random.default_rng()
        x = 1j * rng.standard_normal((*sh, *op._N))
        x += rng.standard_normal((*sh, *op._N))
        x = x.astype(cdtype)
        y = 1j * rng.standard_normal((*sh, *op._M))
        y += rng.standard_normal((*sh, *op._M))
        y = y.astype(cdtype)

        lhs = ct.inner_product(op.apply(x), y, op._D)
        rhs = ct.inner_product(x, op.adjoint(y), op._D)
        assert ct.allclose(lhs, rhs, fdtype)

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
    def op(self, N, M, A, W) -> ftk_czt.CZT:
        return ftk_czt.CZT(N, M, A, W)

    # Helper functions --------------------------------------------------------
