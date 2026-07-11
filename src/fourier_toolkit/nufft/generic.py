import math
import warnings

import finufft

import fourier_toolkit.typing as ftkt

from . import ExponentSign, _eps_default


def _nu2nu(
    x: ftkt.ArrayR,  # (M, D)
    v: ftkt.ArrayR,  # (N, D)
    w: ftkt.ArrayRC,  # (n_trans, M)
    isign: ExponentSign,
    eps: float = _eps_default,
    **kwargs,
) -> ftkt.ArrayC:  # (n_trans, N)
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
) -> finufft.Plan:
    _, D = x.shape
    n_trans, _ = w.shape

    plan = finufft.Plan(
        nufft_type=3,
        n_modes_or_dim=D,
        isign=isign,
        eps=eps,
        dtype=w.dtype,
        n_trans=n_trans,
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
