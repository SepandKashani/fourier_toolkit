import numpy as np
import pytest

import fourier_toolkit.typing as ftkt

from . import helper

parametrize_stack = pytest.mark.parametrize(
    "sh_stack",
    [
        (),
        (1,),
        (2, 3),
        (2, 1),
    ],
)
parametrize_dim = pytest.mark.parametrize("D", [1, 2, 3])


class TestAllClose:
    @pytest.mark.parametrize(
        "ab_dtype",
        [
            "int32",
            "int64",
            "float32",
            "float64",
            "complex64",
            "complex128",
        ],
    )
    def test_accepted_numerical_types(self, array_backend, ab_dtype):
        # any `ab_dtype` should be comparable with allclose()
        rng = np.random.default_rng()

        sh = (5, 3, 4)
        a = rng.standard_normal(sh)
        b = a.copy()

        xp = array_backend.xp
        kwargs = dict(
            dtype=getattr(xp, ab_dtype),
            device=array_backend.device,
        )
        a = xp.asarray(a, **kwargs)
        b = xp.asarray(b, **kwargs)

        helper.allclose(a, b, a.dtype)

    @pytest.mark.parametrize("input_dtype", ["float64", "complex128"])
    @pytest.mark.parametrize("precision", ["low", "high"])
    def test_value(self, array_backend, input_dtype, precision):
        # allclose()-ness depends on requested precision
        rng = np.random.default_rng()

        sh = (5, 3, 4)
        a = rng.standard_normal(sh)
        b = a.copy()

        # Add jitter to `b` based on precision.
        # If changing `atol` thresholds in allclose(), update bounds here accordingly.
        if precision == "low":
            lrhs = 1e-6
        elif precision == "high":
            lrhs = 1e-12
        else:
            raise ValueError
        b += rng.uniform(-lrhs, +lrhs, size=b.shape)

        xp = array_backend.xp
        kwargs = dict(
            dtype=getattr(xp, input_dtype),
            device=array_backend.device,
        )
        a = xp.asarray(a, **kwargs)
        b = xp.asarray(b, **kwargs)
        if precision == "low":
            comparison_dtype = xp.float32
        elif precision == "high":
            comparison_dtype = xp.float64
        else:
            raise ValueError

        assert helper.allclose(a, b, comparison_dtype)


class DistanceMixin:
    @parametrize_stack
    @parametrize_dim
    def test_value(self, array_backend, sh_stack, D):
        rng = np.random.default_rng()

        # We test with (a=complex,b=real) to ensure mixed-dtype works
        sh_core = tuple(rng.integers(2, 6, size=D))
        a = rng.standard_normal(sh_stack + sh_core)
        a = a + 1j * rng.standard_normal(sh_stack + sh_core)
        b = rng.standard_normal(sh_stack + sh_core)
        c_gt = self.compute_distance_gt(a, b, D)

        xp = array_backend.xp
        kwargs = dict(
            device=array_backend.device,
        )
        a = xp.asarray(a, **kwargs)
        b = xp.asarray(b, **kwargs)
        c_gt = xp.asarray(c_gt, **kwargs)

        c = self.compute_distance(a, b, D)
        assert c.shape == sh_stack
        assert helper.allclose(c, c_gt, a.dtype)


class TestRelL2Distance(DistanceMixin):
    @classmethod
    def compute_distance(cls, a: ftkt.Array, b: ftkt.Array, D: int) -> ftkt.Array:
        return helper.rel_l2_distance(a, b, D)

    @classmethod
    def compute_distance_gt(cls, a: np.ndarray, b: np.ndarray, D: int) -> np.ndarray:
        axis = tuple(range(-D, 0))
        return np.sqrt(
            np.sum((a - b) * (a - b).conj(), axis=axis).real
            / np.sum(b * b.conj(), axis=axis).real
        )


class TestRelLInfDistance(DistanceMixin):
    @classmethod
    def compute_distance(cls, a: ftkt.Array, b: ftkt.Array, D: int) -> ftkt.Array:
        return helper.rel_linf_distance(a, b, D)

    @classmethod
    def compute_distance_gt(cls, a: np.ndarray, b: np.ndarray, D: int) -> np.ndarray:
        axis = tuple(range(-D, 0))
        return np.max(np.abs(a - b), axis=axis) / np.max(np.abs(b), axis=axis)


class TestRelL2Close:
    @parametrize_stack
    @parametrize_dim
    def test_value(self, array_backend, sh_stack, D):
        # we verify 2 special cases:
        # (1) rel_l2_distance(a=0,b=random,eps=1) == True
        # (2) rel_l2_distance(a=b,b=random,eps=0) == True
        rng = np.random.default_rng()

        sh_core = tuple(rng.integers(2, 6, size=D))
        b = rng.standard_normal(sh_stack + sh_core)
        b = b + 1j * rng.standard_normal(sh_stack + sh_core)
        a1 = np.zeros(sh_stack + sh_core)
        a2 = b.copy()

        xp = array_backend.xp
        kwargs = dict(
            dtype=xp.complex128,
            device=array_backend.device,
        )
        a1 = xp.asarray(a1, **kwargs)
        a2 = xp.asarray(a2, **kwargs)
        b = xp.asarray(b, **kwargs)

        assert helper.rel_l2_close(a1, b, D, eps=1)
        assert helper.rel_l2_close(a2, b, D, eps=0)


class TestInnerProduct:
    @parametrize_stack
    @parametrize_dim
    def test_value(self, array_backend, sh_stack, D):
        rng = np.random.default_rng()

        sh_core = tuple(rng.integers(2, 6, size=D))
        a = rng.standard_normal(sh_stack + sh_core)
        a = a + 1j * rng.standard_normal(sh_stack + sh_core)
        b = rng.standard_normal(sh_stack + sh_core)
        b = b + 1j * rng.standard_normal(sh_stack + sh_core)
        c_gt = np.zeros(sh_stack, dtype=np.complex128)
        for idx in np.ndindex(sh_stack):
            c_gt[idx] = np.sum(a[idx] * b[idx].conj())

        xp = array_backend.xp
        kwargs = dict(
            dtype=xp.complex128,
            device=array_backend.device,
        )
        a = xp.asarray(a, **kwargs)
        b = xp.asarray(b, **kwargs)
        c_gt = xp.asarray(c_gt, **kwargs)

        assert helper.allclose(
            helper.inner_product(a, b, D),
            c_gt,
            xp.complex128,
        )
