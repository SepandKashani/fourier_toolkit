import importlib
import math
import warnings

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
        case ftku.DLDeviceType.CPU:
            lib = importlib.import_module("finufft")
            discrepancy_kwargs["n_modes_or_dim"] = D
        case ftku.DLDeviceType.CUDA:
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
