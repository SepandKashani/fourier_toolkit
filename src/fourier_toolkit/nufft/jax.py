import json

import jax
import jax.experimental.buffer_callback as jbc
import jax.experimental.hijax as hijax
import jax.numpy as jnp
import numpy as np

import fourier_toolkit.typing as ftkt
import fourier_toolkit.util as ftku

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

    def expand(
        self,
        x: ftkt.ArrayR,
        v: ftkt.ArrayR,
        w: ftkt.ArrayC,
    ) -> ftkt.ArrayC:

        def low_level_fwd(  # plan and execute
            ctx: jbc.ExecutionContext,
            out: jbc.Buffer,
            x: jbc.Buffer,
            v: jbc.Buffer,
            w: jbc.Buffer,
            params: jbc.Buffer,
        ):
            from .generic import _nu2nu as generic_nu2nu

            # de-serialize finufft_kwargs
            raw_params = np.from_dlpack(params, copy=False)
            finufft_kwargs = json.loads(bytes(raw_params).decode("utf-8"))

            device_type, _ = x.__dlpack_device__()
            match device_type:
                case ftku.DLDeviceType.CPU:
                    xp = np
                case ftku.DLDeviceType.CUDA:
                    import cupy as xp
                case _:
                    raise ValueError(f"Unsupported device type: {device_type}.")

            cast = lambda _: xp.asarray(_, copy=False)
            z = generic_nu2nu(
                cast(x),
                cast(v),
                cast(w),
                **finufft_kwargs,
            )
            cast(out)[:] = z

        f_fwd = jbc.buffer_callback(
            low_level_fwd,
            jax.ShapeDtypeStruct(
                shape=self.out_aval.shape,
                dtype=self.out_aval.dtype,
            ),
            command_buffer_compatible=False,  # todo: change me?
        )

        params = np.frombuffer(
            json.dumps(self.finufft_kwargs).encode("utf-8"),
            dtype=np.uint8,
        )
        z = f_fwd(x, v, w, params)
        return z

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
        x_dim, v_dim, w_dim = in_dims
        assert x_dim is None, "Batching (x,) knots is not supported."
        assert v_dim is None, "Batching (v,) knots is not supported."
        out_dim = 0

        (x, v, w) = args
        w = jnp.moveaxis(w, w_dim, out_dim)

        return self(x, v, w), out_dim
