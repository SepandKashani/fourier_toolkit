import functools

import numpy as np
import pytest

import fourier_toolkit.ffs as ftk_ffs
import fourier_toolkit.util as ftk_util
import fourier_toolkit_tests.conftest as ct


class TestFFS:
    @pytest.mark.parametrize("real", [True, False])
    @pytest.mark.parametrize("stack_shape", [(), (1,), (5, 3, 4)])
    def test_value_apply(self, op, dtype, real, stack_shape):
        # output value matches ground truth.
        translate = ftk_util.TranslateDType(dtype)
        fdtype = translate.to_float()
        cdtype = translate.to_complex()

        # Generate ground-truth and FFS test inputs
        N_stack = int(np.prod(stack_shape))
        G_FS_gt = np.zeros((N_stack, *op._L), dtype=cdtype)
        G = np.zeros((N_stack, *op._L), dtype=fdtype if real else cdtype)
        for i in range(N_stack):
            gFS, g = self.generate_data(op._T, op._K, real)

            for d in range(op._D):
                _2Kp1 = 2 * op._K[d] + 1
                gFS[d] = np.pad(gFS[d], pad_width=(0, op._L[d] - _2Kp1))
            G_FS_gt[i] = functools.reduce(np.multiply.outer, gFS)  # (L1,...,LD)

            x = np.meshgrid(*op.sample_points(fdtype), indexing="ij", sparse=True)
            G[i] = g(*x)
        G_FS_gt = G_FS_gt.reshape((*stack_shape, *op._L))
        G = G.reshape((*stack_shape, *op._L))

        # Test FFS compliance
        G_FS = op.apply(G)
        assert G_FS.shape == G_FS_gt.shape
        assert ct.allclose(G_FS, G_FS_gt, fdtype)

    @pytest.mark.parametrize("real", [True, False])
    @pytest.mark.parametrize("direction", ["apply", "adjoint"])
    def test_prec(self, op, dtype, real, direction):
        # output precision (not dtype!) matches input precision.
        translate = ftk_util.TranslateDType(dtype)
        fdtype = translate.to_float()
        cdtype = translate.to_complex()

        rng = np.random.default_rng()
        if real:
            x = rng.standard_normal(size=op._L)
            x = x.astype(fdtype)
        else:
            x = 1j * rng.standard_normal(size=op._L)
            x += rng.standard_normal(size=op._L)
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
        x = 1j * rng.standard_normal((*sh, *op._L))
        x += rng.standard_normal((*sh, *op._L))
        x = x.astype(cdtype)
        y = 1j * rng.standard_normal((*sh, *op._L))
        y += rng.standard_normal((*sh, *op._L))
        y = y.astype(cdtype)

        lhs = ct.inner_product(op.apply(x), y, op._D)
        rhs = ct.inner_product(x, op.adjoint(y), op._D)
        assert ct.allclose(lhs, rhs, fdtype)

    # Fixtures ----------------------------------------------------------------
    @pytest.fixture(params=[1, 2, 3])
    def space_dim(self, request) -> int:
        return request.param

    @pytest.fixture
    def T(self, space_dim) -> tuple[float]:
        rng = np.random.default_rng()
        T = rng.uniform(0.1, 1.3, space_dim)
        return tuple(map(float, T))  # (T1,...,TD)

    @pytest.fixture(params=[True, False])
    def K(self, space_dim, request) -> tuple[int]:
        rng = np.random.default_rng()
        K = rng.integers(0, 5, space_dim)

        force_zero = request.param
        if force_zero:  # assess limiting case when one dimension is DC-only.
            K[0] = 0
        return tuple(map(int, K))  # (K1,...,KD)

    @pytest.fixture(params=[True, False])
    def L(self, space_dim, K, request) -> tuple[int]:
        rng = np.random.default_rng()
        L = np.zeros_like(K)

        tight_fit = request.param
        for d in range(space_dim):  # assess Q=0 case explicitly
            if tight_fit:
                Q = 0
            else:
                Q = rng.integers(2, 6)
            L[d] = 2 * K[d] + 1 + Q

        if (not tight_fit) and (space_dim > 1):  # make dimensions have even/odd lengths
            L[0] += (L[0] + 0) % 2  # L[0] will always be even
            L[1] += (L[1] + 1) % 2  # L[1] will always be odd

        return tuple(map(int, L))

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
    def op(self, T, K, L) -> ftk_ffs.FFS:
        return ftk_ffs.FFS(T, K, L)

    # Helper functions --------------------------------------------------------
    @staticmethod
    def generate_data(T: tuple[float], K: tuple[int], real: bool):
        # Create a seperable, real/complex, T-periodic, K-bandlimited function \tilde{g}.
        # Return its FS coefficients (in seperable form), and a ufunc to compute its space samples.
        #
        # gFS: [(2K1+1,),...,(2KD+1,)]
        # g: func(x1,...,xD) -> R/C
        #    x1,...,xD must be broadcast adequately
        rng = np.random.default_rng()

        D = len(T)
        gFS = [None] * D
        for d in range(D):
            _gFS = 1j * rng.uniform(-1, 1, 2 * K[d] + 1)
            _gFS += rng.uniform(-1, 1, 2 * K[d] + 1)
            if real:
                # conj-symmetrize spectrum
                _gFS = _gFS + _gFS.conj()[::-1]
            gFS[d] = _gFS

        # Create callable time-domain function
        def g(*x):
            y = [None] * D
            for d in range(D):
                x_span = x[d][..., np.newaxis]
                K_range = np.arange(-K[d], K[d] + 1)
                scale = 2 * np.pi / T[d]
                F = np.exp(1j * scale * x_span * K_range)
                y[d] = (F * gFS[d]).sum(axis=-1)
            y = functools.reduce(np.multiply, y)
            return y.real if real else y

        return gFS.copy(), g
