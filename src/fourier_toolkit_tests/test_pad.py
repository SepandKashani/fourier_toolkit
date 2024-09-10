import numpy as np
import pytest

import fourier_toolkit.pad as ftk_pad
import fourier_toolkit.util as ftk_util
import fourier_toolkit_tests.conftest as ct

Pad_config = [
    # (dim_shape, pad_width, mode) configs to test.
    # * `param[0]` corresponds to raw inputs users provide to Pad().
    # * `param[1]` corresponds to their ground-truth canonical parameterization.
    #
    #
    # 1D, uni-mode -------------------------------------
    (
        ((5,), 2, "constant"),
        ((5,), ((2, 2),), ("constant",)),
    ),
    (
        ((5,), 2, "wrap"),
        ((5,), ((2, 2),), ("wrap",)),
    ),
    (
        ((5,), 2, "reflect"),
        ((5,), ((2, 2),), ("reflect",)),
    ),
    (
        ((5,), 2, "symmetric"),
        ((5,), ((2, 2),), ("symmetric",)),
    ),
    (
        ((5,), 2, "edge"),
        ((5,), ((2, 2),), ("edge",)),
    ),
    # ND, uni-mode -------------------------------------
    (
        ((5, 3, 4), 2, "constant"),
        ((5, 3, 4), ((2, 2), (2, 2), (2, 2)), ("constant", "constant", "constant")),
    ),
    (
        ((5, 3, 4), 2, "wrap"),
        ((5, 3, 4), ((2, 2), (2, 2), (2, 2)), ("wrap", "wrap", "wrap")),
    ),
    (
        ((5, 3, 4), 2, "reflect"),
        ((5, 3, 4), ((2, 2), (2, 2), (2, 2)), ("reflect", "reflect", "reflect")),
    ),
    (
        ((5, 3, 4), 2, "symmetric"),
        ((5, 3, 4), ((2, 2), (2, 2), (2, 2)), ("symmetric", "symmetric", "symmetric")),
    ),
    (
        ((5, 3, 4), 2, "edge"),
        ((5, 3, 4), ((2, 2), (2, 2), (2, 2)), ("edge", "edge", "edge")),
    ),
    # ND, multi-mode -----------------------------------
    (
        ((5, 3, 4), (2, 1, 3), ("constant", "edge", "wrap")),
        ((5, 3, 4), ((2, 2), (1, 1), (3, 3)), ("constant", "edge", "wrap")),
    ),
    (
        ((5, 3, 4), ((0, 2), (1, 3), (3, 2)), ("constant", "edge", "wrap")),
        ((5, 3, 4), ((0, 2), (1, 3), (3, 2)), ("constant", "edge", "wrap")),
    ),
    # Special case of padding with zeros ---------------
    (
        ((5,), 0, "constant"),
        ((5,), ((0, 0),), ("constant",)),
    ),
    (
        ((5,), 0, "wrap"),
        ((5,), ((0, 0),), ("wrap",)),
    ),
    (
        ((5,), 0, "reflect"),
        ((5,), ((0, 0),), ("reflect",)),
    ),
    (
        ((5,), 0, "symmetric"),
        ((5,), ((0, 0),), ("symmetric",)),
    ),
    (
        ((5,), 0, "edge"),
        ((5,), ((0, 0),), ("edge",)),
    ),
]


class TestPad:
    @pytest.mark.parametrize("real", [True, False])
    @pytest.mark.parametrize("stack_shape", [(), (1,), (5, 3, 4)])
    def test_value_apply(
        self,
        op,
        dtype,
        c_dimShape_padWidth_mode,
        real,
        stack_shape,
    ):
        # output value matches ground truth.
        translate = ftk_util.TranslateDType(dtype)
        fdtype = translate.to_float()
        cdtype = translate.to_complex()

        # Generate Pad input
        dim_shape, pad_width, mode = c_dimShape_padWidth_mode
        rng = np.random.default_rng()
        if real:
            x = rng.standard_normal((*stack_shape, *dim_shape))
            x = x.astype(fdtype)
        else:
            x = 1j * rng.standard_normal((*stack_shape, *dim_shape))
            x += rng.standard_normal((*stack_shape, *dim_shape))
            x = x.astype(cdtype)

        # Generate Pad output ground-truth
        N_stack = len(stack_shape)
        if len(set(mode)) == 1:  # uni-mode
            p = (((0, 0),) * N_stack) + pad_width
            y_gt = np.pad(
                array=x,
                pad_width=p,
                mode=mode[0],
            )
        else:  # multi-mode
            N_dim = len(dim_shape)
            y_gt = x.copy()
            for i in range(N_dim):
                p = [
                    (0, 0),
                ] * (N_stack + N_dim)
                p[N_stack + i] = pad_width[i]
                y_gt = np.pad(
                    array=y_gt,
                    pad_width=p,
                    mode=mode[i],
                )

        # Test Pad compliance
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
        size = op._dim_shape if (direction == "apply") else op._codim_shape
        if real:
            x = rng.standard_normal(size)
            x = x.astype(fdtype)
            odtype = fdtype
        else:
            x = 1j * rng.standard_normal(size)
            x += rng.standard_normal(size)
            x = x.astype(cdtype)
            odtype = cdtype

        f = getattr(op, direction)
        y = f(x)
        assert y.dtype == odtype

    @pytest.mark.parametrize("real", [True, False])
    def test_math_adjoint(self, op, dtype, real):
        # <A x, y> == <x, A^H y>
        translate = ftk_util.TranslateDType(dtype)
        fdtype = translate.to_float()
        cdtype = translate.to_complex()

        sh = (5, 3, 4)
        rng = np.random.default_rng()
        if real:
            x = rng.standard_normal((*sh, *op._dim_shape))
            x = x.astype(fdtype)
            y = rng.standard_normal((*sh, *op._codim_shape))
            y = y.astype(fdtype)
        else:
            x = 1j * rng.standard_normal((*sh, *op._dim_shape))
            x += rng.standard_normal((*sh, *op._dim_shape))
            x = x.astype(cdtype)
            y = 1j * rng.standard_normal((*sh, *op._codim_shape))
            y += rng.standard_normal((*sh, *op._codim_shape))
            y = y.astype(cdtype)

        lhs = ct.inner_product(op.apply(x), y, op._codim_rank)
        rhs = ct.inner_product(x, op.adjoint(y), op._dim_rank)
        assert ct.allclose(lhs, rhs, fdtype)

    # Fixtures ----------------------------------------------------------------
    @pytest.fixture(params=Pad_config)
    def _spec(self, request):
        return request.param

    @pytest.fixture
    def dimShape_padWidth_mode(self, _spec) -> tuple:
        # user-specified (dim_shape, pad_width, mode).
        # canonical variants provided in c_dimShape_padWidth_mode()
        dim_shape, pad_width, mode = _spec[0]
        return (dim_shape, pad_width, mode)

    @pytest.fixture
    def c_dimShape_padWidth_mode(self, _spec) -> tuple:
        # canonical (dim_shape, pad_width, mode).
        # user-specified variants provided in dimShape_padWidth_mode()
        dim_shape, pad_width, mode = _spec[1]
        return (dim_shape, pad_width, mode)

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
    def op(self, dimShape_padWidth_mode) -> ftk_pad.Pad:
        dim_shape, pad_width, mode = dimShape_padWidth_mode
        return ftk_pad.Pad(
            dim_shape=dim_shape,
            pad_width=pad_width,
            mode=mode,
        )
