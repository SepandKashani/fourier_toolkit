import enum
import importlib
import math
import warnings

import array_api_compat as aac

import fourier_toolkit.typing as ftkt
import fourier_toolkit.util as ftku

from . import ExponentSign


def _nu2nu(
    x: ftkt.ArrayR,  # (M, D)
    v: ftkt.ArrayR,  # (N, D)
    w: ftkt.ArrayC,  # (n_trans, M)
    isign: ExponentSign,
    eps: float,
    **kwargs,
) -> ftkt.ArrayC:  # (n_trans, N)
    # Some FINUFFT options are incompatible with how we use FINUFFT's output:
    # discard them if provided.
    illegal_kwargs = ("gpu_device_id",)
    for k in illegal_kwargs:
        kwargs.pop(k, None)

    plan = _create_t3_plan(x, v, w, isign, eps, **kwargs)
    z = plan.execute(w)
    return z


def _nu2u(
    x: ftkt.ArrayR,  # (M, D)
    v_spec: ftku.UniformSpec,
    w: ftkt.ArrayC,  # (n_trans, M)
    isign: ExponentSign,
    eps: float,
    **kwargs,
) -> ftkt.ArrayC:  # (n_trans, N1,...,ND)
    # Some FINUFFT options are incompatible with how we use FINUFFT's output:
    # discard them if provided.
    illegal_kwargs = ("modeord", "gpu_device_id")
    for k in illegal_kwargs:
        kwargs.pop(k, None)

    xp = aac.array_namespace(x)
    v = xp.reshape(  # (N1...ND, D)
        v_spec.knots(x),
        (-1, v_spec.ndim),
    )
    plan = _create_t3_plan(x, v, w, isign, eps, **kwargs)
    z = xp.reshape(
        plan.execute(w),
        (-1, *v_spec.num),
    )
    return z


def _u2nu(
    x_spec: ftku.UniformSpec,
    v: ftkt.ArrayR,  # (N, D)
    w: ftkt.ArrayC,  # (n_trans, M1,...,MD)
    isign: ExponentSign,
    eps: float,
    **kwargs,
) -> ftkt.ArrayC:  # (n_trans, N)
    # Some FINUFFT options are incompatible with how we use FINUFFT's output:
    # discard them if provided.
    illegal_kwargs = ("modeord", "gpu_device_id")
    for k in illegal_kwargs:
        kwargs.pop(k, None)

    xp = aac.array_namespace(v)
    x = xp.reshape(  # (M1...MD, D)
        x_spec.knots(v),
        (-1, x_spec.ndim),
    )
    w = xp.reshape(  # (n_trans, M1...MD)
        w,
        (-1, math.prod(x_spec.num)),
    )
    plan = _create_t3_plan(x, v, w, isign, eps, **kwargs)
    z = plan.execute(w)
    return z


# Helper routines (internal) ---------------------------------------------------
def _create_t3_plan(
    x: ftkt.ArrayR,  # (M, D)
    v: ftkt.ArrayR,  # (N, D)
    w: ftkt.ArrayC,  # (n_trans, M)
    isign: ExponentSign,
    eps: float,
    **kwargs,
):
    _, D = x.shape
    n_trans, _ = w.shape

    # The [cu]finufft.Plan signatures have slight discrepancies.
    discrepancy_kwargs = dict()
    device_type, device_id = x.__dlpack_device__()
    match device_type:
        case _DLDeviceType.CPU:
            lib = importlib.import_module("finufft")
            discrepancy_kwargs["n_modes_or_dim"] = D
        case _DLDeviceType.CUDA:
            lib = importlib.import_module("cufinufft")
            discrepancy_kwargs["n_modes"] = D
            discrepancy_kwargs["gpu_device_id"] = device_id
        case _:
            raise ValueError(f"Unsupported device type: {device_type}.")
    plan = lib.Plan(
        nufft_type=3,
        isign=isign,
        eps=eps,
        dtype=w.dtype,
        n_trans=n_trans,
        **discrepancy_kwargs,
        **kwargs,  # let FINUFFT parse them for correctness
    )

    scale = math.sqrt(2 * math.pi)
    setpts_kwargs = dict(
        zip(
            "xyz"[:D] + "stu"[:D],
            [*(scale * x.T), *(scale * v.T)],
        )
    )

    with warnings.catch_warnings():
        # FINUFFT stores (x,v) arrays internally in F-order, hence warns user if a copy was
        # required to do so. Since the Python array API standard has no function to coerce array
        # ordering, we ignore this warning.
        warnings.filterwarnings(
            "ignore",
            message=r"Argument `.+` does not satisfy the following requirement: .+\. Copying array \(this may reduce performance\)$",
            category=UserWarning,
        )
        plan.setpts(**setpts_kwargs)

    return plan


class _DLDeviceType(enum.IntEnum):
    """
    Subset of `DLPack Device Codes <https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__dlpack_device__.html#array_api.array.__dlpack_device__>`_.
    """

    CPU = 1
    CUDA = 2
