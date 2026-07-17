import pytest

import fourier_toolkit.typing as ftkt
import fourier_toolkit.util as ftku

from .. import conftest as ct


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
