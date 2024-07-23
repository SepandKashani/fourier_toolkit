import numpy as np
import pytest

import fourier_toolkit.complex as ftk_complex
import fourier_toolkit_tests.conftest as ct


class TestCExp:
    @pytest.mark.parametrize("ndim", [1, 2, 3, 4])
    def test_value(self, ndim, dtype):
        # output value matches ground truth.
        rng = np.random.default_rng()

        x = rng.standard_normal((5, 3, 4, 5, 6)[:ndim], dtype=dtype)
        y_gt = np.exp(1j * x)

        y = ftk_complex.cexp(x)
        assert x.shape == y_gt.shape
        assert ct.allclose(y, y_gt, y_gt.dtype)

    @pytest.mark.parametrize("ndim", [1, 2, 3, 4])
    def test_prec(self, ndim, dtype):
        # output precision (not dtype!) matches input precision.
        rng = np.random.default_rng()

        x = rng.standard_normal((5, 3, 4, 5, 6)[:ndim], dtype=dtype)
        y = ftk_complex.cexp(x)
        assert y.real.dtype == x.dtype
        assert y.imag.dtype == x.dtype

    # Fixtures ----------------------------------------------------------------
    @pytest.fixture(
        params=[
            np.float32,
            np.float64,
        ]
    )
    def dtype(self, request) -> np.dtype:
        return np.dtype(request.param)
