import functools

import numpy as np
import pytest

import fourier_toolkit.linalg as ftk_linalg
import fourier_toolkit_tests.conftest as ct


class TestHadamardOuter:
    def test_value(self, x, args, y_gt):
        # output value matches ground truth.
        y = ftk_linalg.hadamard_outer(x, *args)
        assert x.shape == y_gt.shape
        assert ct.allclose(y, y_gt, y_gt.dtype)

    def test_prec(self, x, args):
        # output precision (not dtype!) matches input precision.
        assert all(x.dtype == A.dtype for A in args)
        y = ftk_linalg.hadamard_outer(x, *args)
        assert y.dtype == x.dtype

    # Fixtures ----------------------------------------------------------------
    @pytest.fixture(params=[1, 2, 3, 4])
    def space_dim(self, request) -> int:
        return request.param

    @pytest.fixture
    def space_shape(self, space_dim) -> tuple[int]:
        rng = np.random.default_rng()
        N = rng.integers(5, 11, space_dim)
        return tuple(map(int, N))  # (N1,...,ND)

    @pytest.fixture(
        params=[
            (),
            (1,),
            (5, 3, 4),
        ]
    )
    def stack_shape(self, request) -> tuple[int]:
        return request.param  # (...)

    @pytest.fixture(
        params=[
            np.int32,
            np.int64,
            np.float32,
            np.float64,
            np.complex64,
            np.complex128,
        ]
    )
    def dtype(self, request) -> np.dtype:
        return np.dtype(request.param)

    @pytest.fixture
    def x(self, space_shape, stack_shape, dtype) -> np.ndarray:
        size = np.prod(stack_shape, dtype=int)
        size *= np.prod(space_shape, dtype=int)
        x = np.linspace(-5, 5, size)
        return x.reshape(*stack_shape, *space_shape).astype(dtype)

    @pytest.fixture
    def args(self, space_dim, space_shape, dtype) -> list[np.ndarray]:
        args = [None] * space_dim
        for d in range(space_dim):
            A = np.linspace(-40, 5, space_shape[d])
            args[d] = A.astype(dtype)
        return args

    @pytest.fixture
    def y_gt(self, x, args, dtype) -> np.ndarray:
        A = functools.reduce(np.multiply.outer, args)  # (N1,...,ND)
        y = x * A  # (..., N1,...,ND)
        return y.astype(dtype)


class TestHadamardOuter2:
    def test_value(self, x, args, y_gt):
        # output value matches ground truth.
        y = ftk_linalg.hadamard_outer2(x, *args)
        assert ct.allclose(y, y_gt, y_gt.dtype)

    def test_prec(self, x, args):
        # output precision (not dtype!) matches input precision.
        assert all(x.dtype == A.dtype for A in args)
        y = ftk_linalg.hadamard_outer2(x, *args)
        assert y.dtype == x.dtype

    # Fixtures ----------------------------------------------------------------
    @pytest.fixture(params=[1, 2, 3, 4])
    def space_dim(self, request) -> int:
        return request.param

    @pytest.fixture(params=[1, 3])
    def p_dim(self, request) -> int:
        return request.param

    @pytest.fixture
    def space_shape(self, space_dim) -> tuple[int]:
        rng = np.random.default_rng()
        N = rng.integers(5, 11, space_dim)
        return tuple(map(int, N))  # (N1,...,ND)

    @pytest.fixture(
        params=[
            (),
            (1,),
            (5, 3, 4),
        ]
    )
    def stack_shape(self, request) -> tuple[int]:
        return request.param  # (...)

    @pytest.fixture(
        params=[
            np.int32,
            np.int64,
            np.float32,
            np.float64,
            np.complex64,
            np.complex128,
        ]
    )
    def dtype(self, request) -> np.dtype:
        return np.dtype(request.param)

    @pytest.fixture
    def x(self, p_dim, space_shape, stack_shape, dtype) -> np.ndarray:
        size = p_dim
        size *= np.prod(stack_shape, dtype=int)
        size *= np.prod(space_shape, dtype=int)
        x = np.linspace(-5, 5, size)
        return x.reshape(p_dim, *stack_shape, *space_shape).astype(dtype)  # (P, *sh, N1,...,ND)

    @pytest.fixture
    def args(self, p_dim, space_dim, space_shape, dtype) -> list[np.ndarray]:
        args = [None] * space_dim
        for d in range(space_dim):
            A = np.linspace(-40, 5, p_dim * space_shape[d])
            A = A.reshape(p_dim, space_shape[d])
            args[d] = A.astype(dtype)
        return args

    @pytest.fixture
    def y_gt(self, x, args, dtype) -> np.ndarray:
        D = len(args)
        P = x.shape[0]
        N = x.shape[-D:]  # (N1,...,ND)
        Ns = len(x.shape[1:-D])  # (...,)

        A = [None] * D
        for d, _A in enumerate(args):
            sh = [1] * (1 + Ns + D)
            sh[0] = P
            sh[1 + Ns + d] = N[d]

            A[d] = _A.reshape(*sh)
        A = functools.reduce(np.multiply, A)  # (P, 1...1, N1,...,ND)

        y = np.sum(x * A, axis=0)  # (..., N1,...,ND)
        return y.astype(dtype)
