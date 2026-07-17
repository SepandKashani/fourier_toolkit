import array_api_compat as aac

import fourier_toolkit.typing as ftkt


class TranslateDType:
    """
    (int,float,complex) dtype translator.
    """

    map_to_float = {
        "int32": "float32",
        "int64": "float64",
        "float32": "float32",
        "float64": "float64",
        "complex64": "float32",
        "complex128": "float64",
    }
    map_from_float = {
        ("float32", "i"): "int32",
        ("float64", "i"): "int64",
        ("float32", "f"): "float32",
        ("float64", "f"): "float64",
        ("float32", "c"): "complex64",
        ("float64", "c"): "complex128",
    }

    def __init__(self, x: ftkt.ArrayRC):
        # Find float-equivalent name of `x.dtype`
        xp = aac.array_namespace(x)
        info = xp.__array_namespace_info__()
        dtypes = info.dtypes()

        found = False
        for name, dtype in dtypes.items():
            if x.dtype == dtype:
                fdtype = self.map_to_float[name]
                found = True
        assert found, f"Un-recognized input dtype '{x.dtype}'."

        # Store (float-equivalent name, backend name->dtype mapping)
        self._fdtype = fdtype
        self._name2dtype = dtypes

    def to_int(self) -> ftkt.DType:
        name = self.map_from_float[(self._fdtype, "i")]
        return self._name2dtype[name]

    def to_float(self) -> ftkt.DType:
        name = self.map_from_float[(self._fdtype, "f")]
        return self._name2dtype[name]

    def to_complex(self) -> ftkt.DType:
        name = self.map_from_float[(self._fdtype, "c")]
        return self._name2dtype[name]
