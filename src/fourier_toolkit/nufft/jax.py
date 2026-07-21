import jax
import jax.experimental.hijax as hijax

import fourier_toolkit.typing as ftkt

from . import ExponentSign

xvw_t = tuple[
    ftkt.ArrayR,  # x
    ftkt.ArrayR,  # v
    ftkt.ArrayC,  # w
]


def _nu2nu(
    x: ftkt.ArrayR,  # (M, D)
    v: ftkt.ArrayR,  # (N, D)
    w: ftkt.ArrayC,  # (n_trans, M)
    isign: ExponentSign,
    eps: float,
    **kwargs,  # valid FINUFFT kwargs only
) -> ftkt.ArrayC:  # (n_trans, N)
    prim = NU2NUPrimitive(
        x_aval=jax.typeof(x),
        v_aval=jax.typeof(v),
        w_aval=jax.typeof(w),
        isign=isign,
        eps=eps,
        **kwargs,
    )
    z = prim(x, v, w)
    return z


class NU2NUPrimitive(hijax.VJPHiPrimitive):
    def __init__(
        self,
        x_aval: jax.core.ShapedArray,
        v_aval: jax.core.ShapedArray,
        w_aval: jax.core.ShapedArray,
        *,
        isign: ExponentSign,
        eps: float,
        **kwargs,
    ):
        self.in_avals = (x_aval, v_aval, w_aval)
        N, _ = v_aval.shape
        n_trans, _ = w_aval.shape
        self.out_aval = jax.core.ShapedArray(
            shape=(n_trans, N),
            dtype=w_aval.dtype,
        )
        self.params = dict(
            finufft_kwargs=dict(isign=isign, eps=eps, **kwargs),
        )
        super().__init__()

    def expand(self, xvw: xvw_t) -> ftkt.ArrayC:
        raise NotImplementedError

    def jvp(
        self,
        primals: xvw_t,
        tangents: xvw_t,
    ) -> tuple[ftkt.ArrayC, ftkt.ArrayC]:
        raise NotImplementedError

    def vjp_fwd(
        self,
        nzs_in: tuple[bool, ...],
        primals: xvw_t,
    ) -> tuple[ftkt.ArrayC, None]:
        raise NotImplementedError

    def vjp_bwd_retval(
        self,
        res: None,
        cotangents: xvw_t,
    ) -> tuple[ftkt.ArrayC]:
        raise NotImplementedError

    def batch(
        self,
        axis_data,  # namedtuple-like
        args: xvw_t,
        in_dims: tuple[int | None, ...],
    ) -> ftkt.ArrayC:
        # piggy-back on fact that generic.nu2nu() accepts stacked arrays
        # do not accept batching (x,v), only (w,)
        raise NotImplementedError
