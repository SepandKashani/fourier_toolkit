import numpy as np
import pytest

from . import helper


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
    def test_same_prec_equivalence(self, array_backend, input_dtype, precision):
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

    def test_broadcasted_inputs(self, array_backend):
        rng = np.random.default_rng()
        a = rng.standard_normal((1, 5, 3))
        b = np.broadcast_to(a, (6, 5, 3)).copy()

        xp = array_backend.xp
        kwargs = dict(
            dtype=xp.float64,
            device=array_backend.device,
        )
        a = xp.asarray(a, **kwargs)
        b = xp.asarray(b, **kwargs)

        assert helper.allclose(a, b, a.dtype)
