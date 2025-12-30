import pytest
import numpy as np
from numpy.typing import DTypeLike, NDArray
import warnings
import fourier_toolkit.util as ftku


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
