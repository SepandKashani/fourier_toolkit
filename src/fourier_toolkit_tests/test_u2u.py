import numpy as np
import pytest

import fourier_toolkit.u2u as ftk_u2u
import fourier_toolkit.util as ftk_util
import fourier_toolkit_tests.conftest as ct


class TestU2U:
    @pytest.mark.parametrize("real", [True, False])
    @pytest.mark.parametrize("stack_shape", [(), (1,), (5, 3, 4)])
    def test_value_apply(self, op, dtype, real, stack_shape):
        # output value matches ground truth.
        translate = ftk_util.TranslateDType(dtype)
        fdtype = translate.to_float()
        cdtype = translate.to_complex()

        # Generate U2U input
        rng = np.random.default_rng()
        if real:
            w = rng.standard_normal((*stack_shape, *op.cfg.M))
            w = w.astype(fdtype)
        else:
            w = 1j * rng.standard_normal((*stack_shape, *op.cfg.M))
            w += rng.standard_normal((*stack_shape, *op.cfg.M))
            w = w.astype(cdtype)

        # Generate U2U output ground-truth
        z_gt = np.zeros((*stack_shape, *op.cfg.N), dtype=cdtype)
        A = self._generate_A(  # (N1,...,ND,M1,...,MD)
            *(op.cfg.x0, op.cfg.dx, op.cfg.M),
            *(op.cfg.v0, op.cfg.dv, op.cfg.N),
            op.cfg.isign,
        )
        for idx in np.ndindex(stack_shape):
            z_gt[idx] = ct.inner_product(w[idx], A, op.cfg.D)  # (N1,...,ND)

        # Test U2U compliance
        z = op.apply(w)
        assert z.shape == z_gt.shape
        assert ct.allclose(z, z_gt, fdtype)

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
        fdtype = translate.to_float()
        cdtype = translate.to_complex()

        sh = (5, 3, 4)
        rng = np.random.default_rng()
        x = 1j * rng.standard_normal((*sh, *op.cfg.M))
        x += rng.standard_normal((*sh, *op.cfg.M))
        x = x.astype(cdtype)
        y = 1j * rng.standard_normal((*sh, *op.cfg.N))
        y += rng.standard_normal((*sh, *op.cfg.N))
        y = y.astype(cdtype)

        lhs = ct.inner_product(op.apply(x), y, op.cfg.D)
        rhs = ct.inner_product(x, op.adjoint(y), op.cfg.D)
        assert ct.allclose(lhs, rhs, fdtype)

    # Fixtures ----------------------------------------------------------------
    @pytest.fixture(params=[1, 2, 3])
    def space_dim(self, request) -> int:
        return request.param

    @pytest.fixture
    def x_spec(self, space_dim) -> dict:
        rng = np.random.default_rng()
        x0 = rng.standard_normal(space_dim)
        dx = rng.uniform(1e-3, 2, space_dim)
        M = rng.integers(1, 7, space_dim)
        return dict(start=x0, step=dx, num=M)

    @pytest.fixture
    def v_spec(self, space_dim) -> dict:
        rng = np.random.default_rng()
        v0 = rng.standard_normal(space_dim)
        dv = rng.uniform(1e-2, 5, space_dim)
        N = rng.integers(3, 12, space_dim)
        return dict(start=v0, step=dv, num=N)

    @pytest.fixture(params=[-1, 1])
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

    @pytest.fixture
    def op(self, x_spec, v_spec, isign) -> ftk_u2u.U2U:
        return ftk_u2u.U2U(x_spec, v_spec, isign=isign)

    # Helper functions --------------------------------------------------------
    @staticmethod
    def _generate_A(x0, dx, M, v0, dv, N, isign) -> np.ndarray:
        # (N1,...,ND,M1,...,MD) tensor which, when inner-produced with `w(M1,...,MD)`, gives `z(N1,...,ND)`.

        x_m = [_x0 + _dx * np.arange(_m) for (_x0, _dx, _m) in zip(x0, dx, M)]
        x_m = np.stack(np.meshgrid(*x_m, indexing="ij"), axis=-1)
        v_n = [_v0 + _dv * np.arange(_n) for (_v0, _dv, _n) in zip(v0, dv, N)]
        v_n = np.stack(np.meshgrid(*v_n, indexing="ij"), axis=-1)

        phase = np.tensordot(v_n, x_m, axes=[[-1], [-1]])  # (N1,...,ND,M1,...,MD)
        A = np.exp(-1j * isign * 2 * np.pi * phase)
        return A
