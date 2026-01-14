import numpy as np
import pytest

import fourier_toolkit.util as ftku
from fourier_toolkit import nu2nu, nu2u, u2nu  # test as exposed to user

from . import conftest as ct


class TestNU2NU:
    @pytest.mark.parametrize("real", [True, False])
    @pytest.mark.parametrize("stack_shape", [(), (1,), (5, 3, 4)])
    def test_value_apply(self, x_m, v_n, eps, finufft_kwargs, dtype, real, stack_shape):
        # output value matches ground truth.
        translate = ftku.TranslateDType(dtype)
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
        A = self._generate_A(x_m, v_n)  # (N, M)
        for idx in np.ndindex(stack_shape):
            z_gt[idx] = ct.inner_product(w[idx], A, 1)  # (N,)

        # Test nu2nu() compliance
        z = nu2nu(
            x=x_m.astype(fdtype),
            v=v_n.astype(fdtype),
            w=w,
            eps=eps,
            **finufft_kwargs,
        )
        assert z.shape == z_gt.shape
        assert ct.rel_error(z, z_gt, D=1).mean() <= eps * 10

    @pytest.mark.parametrize("real", [True, False])
    def test_prec(self, x_m, v_n, eps, finufft_kwargs, dtype, real):
        # output precision (not dtype!) matches input precision.
        translate = ftku.TranslateDType(dtype)
        fdtype = translate.to_float()
        cdtype = translate.to_complex()

        M, _ = x_m.shape
        rng = np.random.default_rng()
        if real:
            w = rng.standard_normal(M)
            w = w.astype(fdtype)
        else:
            w = 1j * rng.standard_normal(M)
            w += rng.standard_normal(M)
            w = w.astype(cdtype)

        z = nu2nu(
            x=x_m.astype(fdtype),
            v=v_n.astype(fdtype),
            w=w,
            eps=eps,
            **finufft_kwargs,
        )
        assert z.dtype == cdtype

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

    @pytest.fixture
    def eps(self) -> float:
        # This order-of-magnitude rel-error should be attainable by both (FP32,FP64) if `upsampfac=2`.
        return 1e-6

    @pytest.fixture
    def finufft_kwargs(self) -> dict:
        return dict(
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
    def _generate_A(x_m, v_n) -> np.ndarray:
        # (N, M) tensor which, when inner-produced with `w(M,)`, gives `z(N,)`.
        phase = np.tensordot(v_n, x_m, axes=[[-1], [-1]])  # (N, M)
        A = np.exp(+1j * 2 * np.pi * phase)
        return A


class TestNU2U:
    @pytest.mark.parametrize("real", [True, False])
    @pytest.mark.parametrize("stack_shape", [(), (1,), (5, 3, 4)])
    def test_value_apply(
        self, x_m, v_spec, eps, finufft_kwargs, dtype, real, stack_shape
    ):
        # output value matches ground truth.
        translate = ftku.TranslateDType(dtype)
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
        N = v_spec.num
        z_gt = np.zeros((*stack_shape, *N), dtype=cdtype)
        A = self._generate_A(x_m, v_spec)  # (N1,...,ND, M)
        for idx in np.ndindex(stack_shape):
            z_gt[idx] = ct.inner_product(w[idx], A, 1)  # (N1,...,ND)

        # Test nu2u() compliance
        z = nu2u(
            x=x_m.astype(fdtype),
            v_spec=v_spec,
            w=w,
            eps=eps,
            **finufft_kwargs,
        )
        assert z.shape == z_gt.shape
        assert ct.rel_error(z, z_gt, D=v_spec.ndim).mean() <= eps * 10

    @pytest.mark.parametrize("real", [True, False])
    def test_prec(self, x_m, v_spec, eps, finufft_kwargs, dtype, real):
        # output precision (not dtype!) matches input precision.
        translate = ftku.TranslateDType(dtype)
        fdtype = translate.to_float()
        cdtype = translate.to_complex()

        M, _ = x_m.shape
        rng = np.random.default_rng()
        if real:
            w = rng.standard_normal(M)
            w = w.astype(fdtype)
        else:
            w = 1j * rng.standard_normal(M)
            w += rng.standard_normal(M)
            w = w.astype(cdtype)

        z = nu2u(
            x=x_m.astype(fdtype),
            v_spec=v_spec,
            w=w,
            eps=eps,
            **finufft_kwargs,
        )
        assert z.dtype == cdtype

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
        N = rng.integers(3, 12, space_dim)
        return ftku.UniformSpec(start=v0, step=dv, num=N)

    @pytest.fixture
    def eps(self) -> float:
        # This order-of-magnitude rel-error should be attainable by both (FP32,FP64) if `upsampfac=2`.
        return 1e-6

    @pytest.fixture
    def finufft_kwargs(self) -> dict:
        return dict(
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
    def _generate_A(x_m, v_spec) -> np.ndarray:
        # (N1,...,ND, M) tensor which, when inner-produced with `w(M,)`, gives `z(N1,...,ND)`.
        phase = np.tensordot(v_spec.knots(np), x_m, axes=[[-1], [-1]])  # (N1,...,ND, M)
        A = np.exp(+1j * 2 * np.pi * phase)
        return A


class TestU2NU:
    @pytest.mark.parametrize("real", [True, False])
    @pytest.mark.parametrize("stack_shape", [(), (1,), (5, 3, 4)])
    def test_value_apply(
        self, x_spec, v_n, eps, finufft_kwargs, dtype, real, stack_shape
    ):
        # output value matches ground truth.
        translate = ftku.TranslateDType(dtype)
        fdtype = translate.to_float()
        cdtype = translate.to_complex()

        # Generate u2nu() input
        M = x_spec.num
        rng = np.random.default_rng()
        if real:
            w = rng.standard_normal((*stack_shape, *M))
            w = w.astype(fdtype)
        else:
            w = 1j * rng.standard_normal((*stack_shape, *M))
            w += rng.standard_normal((*stack_shape, *M))
            w = w.astype(cdtype)

        # Generate u2nu() output ground-truth
        N, D = v_n.shape
        z_gt = np.zeros((*stack_shape, N), dtype=cdtype)
        A = self._generate_A(x_spec, v_n)  # (N, M1,...,MD)
        for idx in np.ndindex(stack_shape):
            z_gt[idx] = ct.inner_product(w[idx], A, D)  # (N,)

        # Test u2nu() compliance
        z = u2nu(
            x_spec=x_spec,
            v=v_n.astype(fdtype),
            w=w,
            eps=eps,
            **finufft_kwargs,
        )
        assert z.shape == z_gt.shape
        assert ct.rel_error(z, z_gt, D=1).mean() <= eps * 10

    @pytest.mark.parametrize("real", [True, False])
    def test_prec(self, x_spec, v_n, eps, finufft_kwargs, dtype, real):
        # output precision (not dtype!) matches input precision.
        translate = ftku.TranslateDType(dtype)
        fdtype = translate.to_float()
        cdtype = translate.to_complex()

        M = x_spec.num
        rng = np.random.default_rng()
        if real:
            w = rng.standard_normal(M)
            w = w.astype(fdtype)
        else:
            w = 1j * rng.standard_normal(M)
            w += rng.standard_normal(M)
            w = w.astype(cdtype)

        z = u2nu(
            x_spec=x_spec,
            v=v_n.astype(fdtype),
            w=w,
            eps=eps,
            **finufft_kwargs,
        )
        assert z.dtype == cdtype

    # Fixtures ----------------------------------------------------------------
    @pytest.fixture(params=[1, 2, 3])
    def space_dim(self, request) -> int:
        return request.param

    @pytest.fixture
    def x_spec(self, space_dim) -> ftku.UniformSpec:
        rng = np.random.default_rng()
        x0 = rng.uniform(-3, -2, space_dim)
        dx = rng.uniform(1e-2, 1.5, space_dim)
        M = rng.integers(3, 12, space_dim)
        return ftku.UniformSpec(start=x0, step=dx, num=M)

    @pytest.fixture
    def v_n(self, space_dim) -> np.ndarray:
        rng = np.random.default_rng()
        N = 50
        v = rng.uniform(0, 5, (N, space_dim))
        v *= rng.uniform(size=space_dim)  # to have different spreads in each direction
        return v

    @pytest.fixture
    def eps(self) -> float:
        # This order-of-magnitude rel-error should be attainable by both (FP32,FP64) if `upsampfac=2`.
        return 1e-6

    @pytest.fixture
    def finufft_kwargs(self) -> dict:
        return dict(
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
    def _generate_A(x_spec, v_n) -> np.ndarray:
        # (N, M1,...,MD) tensor which, when inner-produced with `w(M1,...,MD)`, gives `z(N,)`.
        phase = np.tensordot(v_n, x_spec.knots(np), axes=[[-1], [-1]])  # (N, M1,...,MD)
        A = np.exp(+1j * 2 * np.pi * phase)
        return A
