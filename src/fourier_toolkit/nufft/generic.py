import math

import finufft

import fourier_toolkit.typing as ftkt


from . import ExponentSign


# Helper routines (internal) ---------------------------------------------------
def _create_t3_plan(
    x: ftkt.ArrayR,
    v: ftkt.ArrayR,
    w: ftkt.ArrayC,
    isign: ExponentSign,
    eps: float,
    **kwargs,
) -> finufft.Plan:
    r"""
    Plan a type-3 NUFFT.

    Parameters
    ----------
    x: ArrayR
        (M, D)
    v: ArrayR
        (N, D)
    w: ArrayC
        (n_trans, M)
    isign: +1, -1
    eps: ]0, 1[
    kwargs: dict

    Returns
    -------
    plan: finufft.Plan
    """
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
    plan.setpts(**setpts_kwargs)

    return plan
