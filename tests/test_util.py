import pytest
import numpy as np
from numpy.typing import DTypeLike, NDArray
import warnings
import fourier_toolkit.util as ftku


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

        with pytest.raises(Exception):
            ftku.broadcast_seq(x=(1, 2), N=3)

    def test_casts(self):
        assert ftku.broadcast_seq(x=1.1, cast=int) == (1,)


class TestCastWarn:
    @staticmethod
    def random_array(dtype: DTypeLike, seed: int = 0) -> NDArray:
        rng = np.random.default_rng(seed)
        x = rng.standard_normal(size=(3, 4))
        return x.astype(dtype)

    dtypes = pytest.mark.parametrize(
        "dtype",
        [
            np.float32,
            np.float64,
            np.complex64,
            np.complex128,
            np.int32,
            np.int64,
            np.uint32,
            np.uint64,
        ],
    )

    @dtypes
    def test_no_op(self, dtype):
        dtype = np.dtype(dtype)
        x = self.random_array(dtype)

        # no warning must be emitted
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            y = ftku.cast_warn(x, dtype)
            assert not w

        assert y is x

    @dtypes
    def test_warns(self, dtype):
        in_dtype = np.dtype(dtype)
        out_dtype = np.dtype(np.float16)

        x = self.random_array(in_dtype)

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
    def _5_smooth(cls) -> NDArray:
        exp = np.arange(10 + 1)
        s2, s3, s5 = np.meshgrid(2**exp, 3**exp, 5**exp, indexing="ij", sparse=True)
        s235 = np.sort((s2 * s3 * s5).reshape(-1))
        return s235


class TestTranslateDType:
    @pytest.mark.parametrize(
        ["in_dtype", "out_dtype"],
        [
            (np.int32, np.int32),
            (np.int64, np.int64),
            (np.float32, np.int32),
            (np.float64, np.int64),
            (np.complex64, np.int32),
            (np.complex128, np.int64),
        ],
    )
    def test_to_int(self, in_dtype, out_dtype):
        dtype = ftku.TranslateDType(in_dtype).to_int()
        assert dtype == out_dtype

    @pytest.mark.parametrize(
        ["in_dtype", "out_dtype"],
        [
            (np.int32, np.float32),
            (np.int64, np.float64),
            (np.float32, np.float32),
            (np.float64, np.float64),
            (np.complex64, np.float32),
            (np.complex128, np.float64),
        ],
    )
    def test_to_float(self, in_dtype, out_dtype):
        dtype = ftku.TranslateDType(in_dtype).to_float()
        assert dtype == out_dtype

    @pytest.mark.parametrize(
        ["in_dtype", "out_dtype"],
        [
            (np.int32, np.complex64),
            (np.int64, np.complex128),
            (np.float32, np.complex64),
            (np.float64, np.complex128),
            (np.complex64, np.complex64),
            (np.complex128, np.complex128),
        ],
    )
    def test_to_complex(self, in_dtype, out_dtype):
        dtype = ftku.TranslateDType(in_dtype).to_complex()
        assert dtype == out_dtype


class TestUniformSpec:
    def test_scalar_input(self):
        uspec_1 = ftku.UniformSpec(1, 0.5, 5)
        uspec_2 = ftku.UniformSpec((1,), 0.5, (5,))
        assert uspec_1 == uspec_2

        uspec_3 = ftku.UniformSpec(1, (0.5, 0.25), 3)
        assert uspec_3.start == (1, 1)
        assert uspec_3.step == (0.5, 0.25)
        assert uspec_3.num == (3, 3)

    def test_center_span_initialization(self):
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

        uspec_2 = ftku.UniformSpec(start=(0, -0.5), step=(0.5, 0.75), num=5)
        assert uspec_2.ndim == 2
        assert uspec_2.center == (1, 1)
        assert uspec_2.span == (2, 3)

    @pytest.mark.parametrize("sparse", [True, False])
    def test_meshgrid(self, sparse):
        # 1D case
        uspec_1 = ftku.UniformSpec(start=1, step=0.1, num=9)
        mesh_1 = uspec_1.meshgrid(np, sparse)
        mesh_1_gt = (1 + 0.1 * np.arange(9),)
        assert len(mesh_1) == len(mesh_1_gt) == 1
        assert np.allclose(mesh_1[0], mesh_1_gt[0])

        # multi-dimensional case
        uspec_2 = ftku.UniformSpec(start=1, step=(0.1, 0.2), num=9)
        mesh_2 = uspec_2.meshgrid(np, sparse)
        mesh_2_gt = np.meshgrid(
            *(
                1 + 0.1 * np.arange(9),
                1 + 0.2 * np.arange(9),
            ),
            indexing="ij",
            sparse=sparse,
        )
        assert len(mesh_2) == len(mesh_2_gt) == 2
        for m2, m2gt in zip(mesh_2, mesh_2_gt):
            assert m2.shape == m2gt.shape
            assert np.allclose(m2, m2gt)

    def test_knots(self):
        uspec = ftku.UniformSpec(start=1, step=(0.1, 0.2), num=(9, 8))
        knots = uspec.knots(np)
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

        assert knots.shape == knots_gt.shape == (9, 8, 2)
        assert np.allclose(knots, knots_gt)

    def test_neg(self):
        uspec = ftku.UniformSpec(1, (2, 0.5), (3, 4))
        mesh = uspec.meshgrid(np)

        neg_uspec = -uspec
        neg_mesh = neg_uspec.meshgrid(np)

        assert np.allclose(-mesh[0][::-1, ::-1], neg_mesh[0])
        assert np.allclose(-mesh[1][::-1, ::-1], neg_mesh[1])


class TestInterval:
    def test_positive_span(self):
        ftku.Interval(1, 2)  # ok
        with pytest.raises(Exception):
            ftku.Interval(1, span=0)

    def test_bounds(self):
        bbox = ftku.Interval((0, 2), (2, 4))
        assert bbox.bounds() == ((-1, 1), (0, 4))
