import warnings

import array_api_compat as aac
import numpy as np
import pytest

import fourier_toolkit.typing as ftkt
import fourier_toolkit.util as ftku

from . import conftest as ct
from . import helper


class TestAsNamedTuple:
    def test_creation(self):
        x = {"a": 1, "b": 2}
        y = ftku.as_namedtuple(**x)

        assert hasattr(y, "a") and y.a == 1
        assert hasattr(y, "b") and y.b == 2

    def test_no_copy(self):
        # if objects stored in mapping, only references are stored in the namedtuple.
        x = {"a": np.r_[1, 2], "b": 1}
        y = ftku.as_namedtuple(**x)

        x["b"] += 1
        assert x["b"] == 2
        assert y.b == 1

        assert y.a is x["a"]
        x["a"] += 1
        assert np.allclose(y.a, np.r_[2, 3])


class TestBroadcastSeq:
    def test_tuplizes(self):
        assert ftku.broadcast_seq(x=1) == (1,)
        assert ftku.broadcast_seq(x=[1]) == (1,)
        assert ftku.broadcast_seq(x=np.r_[1, 2]) == (1, 2)

    def test_broadcasts(self):
        assert ftku.broadcast_seq(x=1, N=1) == (1,)
        assert ftku.broadcast_seq(x=1, N=2) == (1, 1)
        assert ftku.broadcast_seq(x=1, N=3) == (1, 1, 1)

        assert ftku.broadcast_seq(x=(1,), N=3) == (1, 1, 1)
        assert ftku.broadcast_seq(x=[1], N=3) == (1, 1, 1)

        with pytest.raises(Exception):
            ftku.broadcast_seq(x=(1, 2), N=3)

    def test_casts(self):
        assert ftku.broadcast_seq(x=1.1, cast=int) == (1,)


class TestCastWarn:
    @staticmethod
    def random_array(array_backend: ct.ArrayBackend, dtype_name: str) -> ftkt.Array:
        rng = np.random.default_rng()
        x = rng.standard_normal(size=(3, 4))

        xp = array_backend.xp
        info = xp.__array_namespace_info__()
        dtype = info.dtypes()[dtype_name]

        x = xp.asarray(
            x,
            dtype=dtype,
            device=array_backend.device,
        )
        return x

    parametrize_dtypes = pytest.mark.parametrize(
        "dtype_name",
        [
            "float32",
            "float64",
            "complex64",
            "complex128",
            "int32",
            "int64",
        ],
    )

    @parametrize_dtypes
    def test_no_op(self, array_backend, dtype_name):
        x = self.random_array(array_backend, dtype_name)

        # no warning must be emitted
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            y = ftku.cast_warn(x, x.dtype)
            assert not w

        assert y is x

    @parametrize_dtypes
    def test_warns(self, array_backend, dtype_name):
        x = self.random_array(array_backend, dtype_name)

        xp = aac.array_namespace(x)
        out_dtype = xp.int16  # something not in `dtype_name`

        # warning must be emitted
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            y = ftku.cast_warn(x, out_dtype)
            assert w

        assert y.dtype == out_dtype


class TestNextFastLen:
    def test_5s_in_5s_out(self):
        # Function is a no-op for 5-smooth inputs.
        for s in self._5_smooth():
            assert ftku.next_fast_len(s) == s

    def test_always_gt_n(self):
        # Predicted 5-smooth number always >= input
        for n in range(1, 10_000):
            assert n <= ftku.next_fast_len(n)

    # Fixtures ----------------------------------------------------------------
    @classmethod
    def _5_smooth(cls) -> np.ndarray:
        exp = np.arange(10 + 1)
        s2, s3, s5 = np.meshgrid(2**exp, 3**exp, 5**exp, indexing="ij", sparse=True)
        s235 = np.sort((s2 * s3 * s5).reshape(-1))
        return s235


class TestTranslateDType:
    @staticmethod
    def random_array(array_backend: ct.ArrayBackend, dtype_name: str) -> ftkt.Array:
        xp = array_backend.xp
        info = xp.__array_namespace_info__()
        dtype = info.dtypes()[dtype_name]
        return xp.asarray([], dtype=dtype, device=array_backend.device)

    @pytest.mark.parametrize(
        ["in_dtype", "out_dtype_gt"],
        [
            ("int32", "int32"),
            ("int64", "int64"),
            ("float32", "int32"),
            ("float64", "int64"),
            ("complex64", "int32"),
            ("complex128", "int64"),
        ],
    )
    def test_to_int(self, array_backend, in_dtype, out_dtype_gt):
        x = self.random_array(array_backend, out_dtype_gt)
        out_dtype_gt = x.dtype

        x = self.random_array(array_backend, in_dtype)
        out_dtype = ftku.TranslateDType(x).to_int()
        assert out_dtype == out_dtype_gt

    @pytest.mark.parametrize(
        ["in_dtype", "out_dtype_gt"],
        [
            ("int32", "float32"),
            ("int64", "float64"),
            ("float32", "float32"),
            ("float64", "float64"),
            ("complex64", "float32"),
            ("complex128", "float64"),
        ],
    )
    def test_to_float(self, array_backend, in_dtype, out_dtype_gt):
        x = self.random_array(array_backend, out_dtype_gt)
        out_dtype_gt = x.dtype

        x = self.random_array(array_backend, in_dtype)
        out_dtype = ftku.TranslateDType(x).to_float()
        assert out_dtype == out_dtype_gt

    @pytest.mark.parametrize(
        ["in_dtype", "out_dtype_gt"],
        [
            ("int32", "complex64"),
            ("int64", "complex128"),
            ("float32", "complex64"),
            ("float64", "complex128"),
            ("complex64", "complex64"),
            ("complex128", "complex128"),
        ],
    )
    def test_to_complex(self, array_backend, in_dtype, out_dtype_gt):
        x = self.random_array(array_backend, out_dtype_gt)
        out_dtype_gt = x.dtype

        x = self.random_array(array_backend, in_dtype)
        out_dtype = ftku.TranslateDType(x).to_complex()
        assert out_dtype == out_dtype_gt


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
        assert helper.similar(mesh_1[0], mesh_1_gt[0])
        assert helper.allclose(mesh_1[0], mesh_1_gt[0], fdtype)

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
            assert helper.similar(m2, m2gt)
            assert helper.allclose(m2, m2gt, fdtype)

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
        assert helper.similar(knots, knots_gt)
        assert helper.allclose(knots, knots_gt, fdtype)

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
