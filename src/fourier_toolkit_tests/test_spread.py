import numba as nb
import numba.types as nbt
import numpy as np
import pytest

import fourier_toolkit.kernel as ftk_kernel
import fourier_toolkit.spread as ftk_spread
import fourier_toolkit.util as ftk_util
import fourier_toolkit_tests.conftest as ct

# spread/interpolate() take tuple[func] as parameters -> known experimental feature.
pytestmark = pytest.mark.filterwarnings("ignore::numba.NumbaExperimentalFeatureWarning")


class TestUniformSpread:
    # Idea: evaluate
    #     g(z) = \sum_{m} w_{m} \psi(z - x_{m})
    # on the regular lattice
    #     z_{n} = z0 + dz * n
    # where
    #     \psi(z) = \phi(\alpha z)
    #     \phi(x) = (1-|x|^2) 1_{-1 <= x <  0}
    #             + (1-|x|)   1_{ 0 <= x <= 1}
    #             OR
    #             = (1-|x|^2) 1_{-1 <= x <= 1}
    # \alpha chosen s.t.
    #     (2 / dz) (k_max / N-1) \le \alpha \le (2 / dz)
    # which guarantees \psi support in [dz, k_max * dz], i.e. [1--k_max] samples.
    #
    # Points x_{m} are chosen to lie inside & outside the lattice.

    @pytest.mark.parametrize("real", [True, False])
    @pytest.mark.parametrize("stack_shape", [(), (1,), (5, 3, 4)])
    def test_value_apply(self, op, dtype, real, stack_shape):
        # output value matches ground truth.
        translate = ftk_util.TranslateDType(dtype)
        fdtype = translate.to_float()
        cdtype = translate.to_complex()

        # Generate UniformSpread input
        rng = np.random.default_rng()
        if real:
            w = rng.standard_normal((*stack_shape, op.cfg.M))
            w = w.astype(fdtype)
        else:
            w = 1j * rng.standard_normal((*stack_shape, op.cfg.M))
            w += rng.standard_normal((*stack_shape, op.cfg.M))
            w = w.astype(cdtype)

        # Generate UniformSpread output ground-truth
        A = self._generate_A(  # (N1,...,ND, M)
            op.cfg.x,
            op.cfg.z0,
            op.cfg.dz,
            op.cfg.N,
            op.cfg.phi,
            op.cfg.alpha,
        )
        z_gt = np.tensordot(w, A, axes=[[-1], [-1]])  # (*sh, N1,...,ND)

        # Test UniformSpread compliance
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
            w = rng.standard_normal(size)
            w = w.astype(fdtype)
            odtype = fdtype
        else:
            w = 1j * rng.standard_normal(size)
            w += rng.standard_normal(size)
            w = w.astype(cdtype)
            odtype = cdtype

        f = getattr(op, direction)
        z = f(w)
        assert z.dtype == odtype

    @pytest.mark.parametrize("real", [True, False])
    def test_math_adjoint(self, op, dtype, real):
        # <A x, y> == <x, A^H y>
        translate = ftk_util.TranslateDType(dtype)
        fdtype = translate.to_float()
        cdtype = translate.to_complex()

        sh = (5, 3, 4)
        rng = np.random.default_rng()
        if real:
            w = rng.standard_normal((*sh, op.cfg.M))
            w = w.astype(fdtype)
            z = rng.standard_normal((*sh, *op.cfg.N))
            z = z.astype(fdtype)
        else:
            w = 1j * rng.standard_normal((*sh, op.cfg.M))
            w += rng.standard_normal((*sh, op.cfg.M))
            w = w.astype(cdtype)
            z = 1j * rng.standard_normal((*sh, *op.cfg.N))
            z += rng.standard_normal((*sh, *op.cfg.N))
            z = z.astype(cdtype)

        lhs = ct.inner_product(op.apply(w), z, op.cfg.D)
        rhs = ct.inner_product(w, op.adjoint(z), 1)
        assert ct.allclose(lhs, rhs, fdtype)

    # Fixtures ----------------------------------------------------------------
    @pytest.fixture(params=[1, 2])  # we don't do 3D since GT creation takes a long time.
    def space_dim(self, request) -> int:
        return request.param

    @pytest.fixture
    def z_spec(self, space_dim) -> dict[str, np.ndarray]:
        rng = np.random.default_rng()
        z0 = rng.uniform(-1, 1, space_dim)
        dz = rng.uniform(0.1, 0.3, space_dim)
        N = 50 + np.arange(space_dim)
        return dict(start=z0, step=dz, num=N)

    @pytest.fixture
    def x(self, space_dim, z_spec) -> np.ndarray:
        z0 = z_spec["start"]
        dz = z_spec["step"]
        N = z_spec["num"]

        rng = np.random.default_rng()
        M = 500

        # Points inside lattice
        x_i = rng.uniform(0, 1, (M, space_dim))
        x_i = z0 + x_i * dz * (N - 1)

        # Points at lattice boundary
        x_b = rng.uniform(0, 1, (3, space_dim))
        x_b = z0 - x_b * dz * 1  # 1 sample outside lattice -> interface region

        # Points outside lattice
        x_o = rng.uniform(0, 1, (3, space_dim))
        x_o = z0 - x_o * dz * 10  # 10 samples outside lattice -> external region

        x_m = np.concatenate([x_b, x_i, x_o], axis=0)  # (M + 6, D)
        return x_m

    @pytest.fixture(params=[True, False])
    def inline_kernel(self, request) -> bool:
        return request.param

    @pytest.fixture(params=[True, False])
    def symmetric_kernel(self, request) -> bool:
        return request.param

    @pytest.fixture
    def phi(self, space_dim, symmetric_kernel, inline_kernel) -> tuple[callable]:
        if inline_kernel:
            if symmetric_kernel:
                kernel = ftk_kernel.PPoly(
                    weight=np.array(
                        [
                            [-1, 0, 1],  # (-1)x**2 + (+0)x + (+1)
                        ]
                    ),
                    pitch=1,
                    sym=True,
                )
            else:
                kernel = ftk_kernel.PPoly(
                    weight=np.array(
                        [
                            [-1, 0, 1],  # (-1)x**2 + (+0)x + (+1)
                            [0, -1, 1],  # (+0)x**2 + (-1)x + (+1)
                        ]
                    ),
                    pitch=1,
                    sym=False,
                )
        else:
            nb_flags = dict(
                nopython=True,
                nogil=True,
                cache=False,  # don't cache during tests
                forceobj=False,
                parallel=False,
                error_model="numpy",
                fastmath=True,
                locals={},
                boundscheck=False,
            )
            jit = nb.jit(
                [
                    nbt.float32(nbt.float32),
                    nbt.float64(nbt.float64),
                ],
                **nb_flags,
            )

            if symmetric_kernel:

                @jit
                def kernel(x):
                    if -1 <= x <= 1:
                        y = 1 - (x**2)
                    else:
                        y = 0
                    return y

            else:

                @jit
                def kernel(x):
                    if 0 <= x <= 1:
                        y = 1 - x
                    elif -1 <= x < 0:
                        y = 1 - (x**2)
                    else:
                        y = 0
                    return y

        return (kernel,) * space_dim

    @pytest.fixture
    def alpha(self, space_dim, z_spec) -> np.ndarray:
        rng = np.random.default_rng()

        k_max = 5
        dz = z_spec["step"]
        N = z_spec["num"]

        a = np.zeros(space_dim)
        for d in range(space_dim):
            lhs = (2 / dz[d]) * (k_max / (N[d] - 1))
            rhs = 2 / dz[d]
            a[d] = rng.uniform(lhs, rhs)
        return a

    @pytest.fixture(
        params=[
            {"max_cluster_size": 50, "max_window_ratio": 3, "nthreads": 0},
            {"max_cluster_size": 250, "max_window_ratio": 3, "nthreads": 1},
            {"max_cluster_size": 450, "max_window_ratio": 10, "nthreads": 0},
        ]
    )
    def kwargs(self, inline_kernel, request) -> dict:
        data = request.param.copy()
        data.update(inline_kernel=inline_kernel)
        return data

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
    def op(self, x, z_spec, phi, alpha, kwargs) -> ftk_spread.UniformSpread:
        return ftk_spread.UniformSpread(x, z_spec, phi, alpha, **kwargs)

    # Helper functions --------------------------------------------------------
    @staticmethod
    def _generate_A(x, z0, dz, N, phi, alpha) -> np.ndarray:
        # (N1,...,ND, M) matrix form of the operator.
        M, D = x.shape

        z = [None] * D
        for d in range(D):
            z[d] = z0[d] + dz[d] * np.arange(N[d])
        z = np.stack(np.meshgrid(*z, indexing="ij"), axis=-1)  # (N1,...,ND,D)

        # vectorized variant of phi for convenience.
        phiv = [None] * D
        for d in range(D):
            phiv[d] = np.vectorize(phi[d])

        offset = z.reshape(*N, 1, D) - x  # (N1,...,ND,M,D)
        A = np.ones((*N, M), dtype=np.double)
        for d in range(D):
            phi_args = alpha[d] * offset[..., d]
            mask = abs(phi_args) <= 1
            A[mask] *= phiv[d](phi_args[mask])
            A[~mask] = 0
        return A
