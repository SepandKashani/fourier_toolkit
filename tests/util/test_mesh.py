import numpy as np
import pytest

import fourier_toolkit.util as ftku


class TestUniformSpec:
    parametrize_step_sign = pytest.mark.parametrize("step_sign", [+1, -1])
    parametrize_fdtype = pytest.mark.parametrize("fdtype_name", ["float32", "float64"])

    @parametrize_step_sign
    def test_scalar_input(self, step_sign):
        uspec_1 = ftku.UniformSpec(start=1, step=step_sign * 0.5, num=5)
        uspec_2 = ftku.UniformSpec(start=(1,), step=step_sign * 0.5, num=(5,))
        assert uspec_1 == uspec_2

        uspec_3 = ftku.UniformSpec(start=1, step=(0.5, step_sign * 0.25), num=3)
        assert uspec_3.start == (1, 1)
        assert uspec_3.step == (0.5, step_sign * 0.25)
        assert uspec_3.num == (3, 3)

    @parametrize_step_sign
    def test_center_span_initialization(self, step_sign):
        uspec_1 = ftku.UniformSpec(center=1, span=2, num=5)
        uspec_2 = ftku.UniformSpec(start=0, step=0.5, num=5)
        assert uspec_1 == uspec_2

        # multi-dim case
        uspec_3 = ftku.UniformSpec(center=1, span=(2, 3), num=5)
        uspec_4 = ftku.UniformSpec(start=(0, -0.5), step=(0.5, 0.75), num=5)
        assert uspec_3 == uspec_4

        with pytest.raises(Exception):
            # (num >= 2) mandatory
            ftku.UniformSpec(center=0, span=1, num=1)

    def test_properties(self):
        uspec_1 = ftku.UniformSpec(start=0, step=0.5, num=5)
        assert uspec_1.ndim == 1
        assert uspec_1.center == (1,)
        assert uspec_1.span == (2,)

        uspec_2 = ftku.UniformSpec(start=0, step=-0.5, num=5)
        assert uspec_2.ndim == 1
        assert uspec_2.center == (-1,)
        assert uspec_2.span == (2,)

        uspec_3 = ftku.UniformSpec(start=(0, -0.5), step=(0.5, -0.75), num=5)
        assert uspec_3.ndim == 2
        assert uspec_3.center == (1, -2)
        assert uspec_3.span == (2, 3)

    @parametrize_step_sign
    @pytest.mark.parametrize("sparse", [True, False])
    @parametrize_fdtype
    def test_meshgrid(self, array_backend, step_sign, sparse, fdtype_name):
        xp = array_backend.xp
        device = array_backend.device
        info = xp.__array_namespace_info__()
        fdtype = info.dtypes()[fdtype_name]
        like = xp.asarray([], dtype=fdtype, device=device)

        # 1D case
        step = step_sign * 0.1
        uspec_1 = ftku.UniformSpec(start=1, step=step, num=9)
        mesh_1 = uspec_1.meshgrid(sparse, like)
        mesh_1_gt = (1 + step * xp.arange(9, dtype=fdtype, device=device),)
        assert len(mesh_1) == len(mesh_1_gt) == 1
        assert ftku.similar(mesh_1[0], mesh_1_gt[0])
        assert ftku.allclose(mesh_1[0], mesh_1_gt[0], fdtype)

        # multi-dimensional case
        step_1, step_2 = step_sign * 0.1, 0.2
        uspec_2 = ftku.UniformSpec(start=1, step=(step_1, step_2), num=9)
        mesh_2 = uspec_2.meshgrid(sparse, like)
        mesh_2_gt = np.meshgrid(
            *(
                1 + step_1 * np.arange(9),
                1 + step_2 * np.arange(9),
            ),
            indexing="ij",
            sparse=sparse,
        )
        mesh_2_gt = [
            xp.asarray(m2gt, dtype=fdtype, device=device) for m2gt in mesh_2_gt
        ]
        assert len(mesh_2) == len(mesh_2_gt) == 2
        for m2, m2gt in zip(mesh_2, mesh_2_gt):
            assert m2.shape == m2gt.shape
            assert ftku.similar(m2, m2gt)
            assert ftku.allclose(m2, m2gt, fdtype)

    @parametrize_fdtype
    def test_knots(self, array_backend, fdtype_name):
        xp = array_backend.xp
        device = array_backend.device
        info = xp.__array_namespace_info__()
        fdtype = info.dtypes()[fdtype_name]
        like = xp.asarray([], dtype=fdtype, device=device)

        uspec = ftku.UniformSpec(start=1, step=(0.1, 0.2), num=(9, 8))
        knots = uspec.knots(like)
        knots_gt = np.stack(
            np.meshgrid(
                *(
                    1 + 0.1 * np.arange(9),
                    1 + 0.2 * np.arange(8),
                ),
                indexing="ij",
            ),
            axis=-1,
        )
        knots_gt = xp.asarray(knots_gt, dtype=fdtype, device=device)

        assert knots.shape == knots_gt.shape == (9, 8, 2)
        assert ftku.similar(knots, knots_gt)
        assert ftku.allclose(knots, knots_gt, fdtype)

    def test_getitem(self):
        uspec_1 = ftku.UniformSpec(start=0, step=0.5, num=5)
        assert uspec_1[0] == (0,)
        assert uspec_1[0,] == (0,)
        assert uspec_1[4] == (2,)
        assert uspec_1[4,] == (2,)
        assert uspec_1[-1] == (2,)
        assert uspec_1[-1,] == (2,)

        uspec_2 = ftku.UniformSpec(start=(0, -0.5), step=(0.5, -0.75), num=5)
        with pytest.raises(Exception):
            uspec_2[0]  # insufficient indices provided
        assert uspec_2[0, 0] == (0, -0.5)
        assert uspec_2[-1, -2] == (2, -2.75)
        assert uspec_2[-2, 3] == (1.5, -2.75)
