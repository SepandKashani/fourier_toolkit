import warnings

import array_api_compat as aac
import numpy as np
import pytest

import fourier_toolkit.typing as ftkt
import fourier_toolkit.util as ftku

from .. import conftest as ct


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
