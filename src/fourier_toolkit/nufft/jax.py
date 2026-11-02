import importlib

import jax
import jax.experimental.buffer_callback as jbc
import jax.experimental.hijax as hijax
import jax.numpy as jnp

import fourier_toolkit.typing as ftkt
import fourier_toolkit.util as ftku

from . import ExponentSign, finufft_kwargsT

xvw_t = tuple[
    ftkt.ArrayR,  # x
    ftkt.ArrayR,  # v
    ftkt.ArrayC,  # w
]


def _nu2nu(
    x: ftkt.ArrayR,  # (M, D)
    v: ftkt.ArrayR,  # (N, D)
    w: ftkt.ArrayC,  # (..., M)
    isign: ExponentSign,
    eps: float,
    finufft_kwargs: finufft_kwargsT,
) -> ftkt.ArrayC:  # (..., N)
    prim = NU2NUPrimitive(
        x_aval=jax.typeof(x),
        v_aval=jax.typeof(v),
        w_aval=jax.typeof(w),
        isign=isign,
        eps=eps,
        finufft_kwargs=finufft_kwargs,
    )
    z = prim(x, v, w)
    return z


class NU2NUPrimitive(hijax.HiPrim):
    def __init__(
        self,
        x_aval: jax.core.ShapedArray,
        v_aval: jax.core.ShapedArray,
        w_aval: jax.core.ShapedArray,
        *,
        isign: ExponentSign,
        eps: float,
        finufft_kwargs: finufft_kwargsT,
    ):
        self.in_avals = (x_aval, v_aval, w_aval)
        N, _ = v_aval.shape
        sh = w_aval.shape[:-1]
        self.out_aval = jax.core.ShapedArray(
            shape=(*sh, N),
            dtype=w_aval.dtype,
        )
        self.params = dict(
            isign=isign,
            eps=eps,
            finufft_kwargs=finufft_kwargs,
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
        ):
            from .generic import _nu2nu as generic_nu2nu

            device_type, _ = x.__dlpack_device__()
            match device_type:
                case ftku.DLDeviceType.CPU:
                    xp = importlib.import_module("numpy")
                case ftku.DLDeviceType.CUDA:
                    xp = importlib.import_module("cupy")
                case _:
                    raise ValueError(f"Unsupported device type: {device_type}.")

            asarray = lambda _: xp.asarray(_, copy=False)
            z = generic_nu2nu(
                asarray(x),
                asarray(v),
                asarray(w),
                **self.params,
            )
            asarray(out)[:] = z

        f_fwd = jbc.buffer_callback(
            low_level_fwd,
            jax.ShapeDtypeStruct(
                shape=self.out_aval.shape,
                dtype=self.out_aval.dtype,
            ),
            command_buffer_compatible=False,  # unsure if can be enabled safely
        )

        z = f_fwd(x, v, w)
        return z

    def jvp(
        self,
        primals: xvw_t,
        tangents: xvw_t,
    ) -> tuple[ftkt.ArrayC, ftkt.ArrayC]:
        # (x,v) non-differentiable at the moment -> jvp() rule identical to u2u()
        (x, v, w) = primals
        (xt, vt, wt) = tangents
        return self(x, v, w), self(x, v, wt)

    def vjp_fwd(
        self,
        nzs_in: tuple[bool, ...],
        x: ftkt.ArrayR,
        v: ftkt.ArrayR,
        w: ftkt.ArrayC,
    ) -> tuple[ftkt.ArrayC, tuple[ftkt.ArrayR, ...]]:
        return self(x, v, w), (x, v)

    def vjp_bwd_retval(
        self,
        res: tuple[ftkt.ArrayR, ...],
        cotangents: ftkt.ArrayC,
    ) -> tuple[ftkt.ArrayC]:
        (x, v) = res

        # vjp() computes (A^{\transpose} z), not (A^{\adjoint} z) ...
        prim = NU2NUPrimitive(
            x_aval=jax.typeof(v),
            v_aval=jax.typeof(x),
            w_aval=jax.typeof(cotangents),
            **self.params,  # ... reason why the sign matches the forward pass.
        )

        xt = jnp.zeros_like(x)  # since non-differentiable (at the moment)
        vt = jnp.zeros_like(v)  # since non-differentiable (at the moment)
        wt = prim(v, x, cotangents)
        return (xt, vt, wt)

    def batch(
        self,
        axis_data,  # namedtuple-like
        args: xvw_t,
        in_dims: tuple[int | None, ...],
    ) -> ftkt.ArrayC:
        x_dim, v_dim, w_dim = in_dims
        out_dim = 0
        # do not accept batching (x,v), only (w,)
        assert x_dim is None, "Batching (x,) knots is not supported."
        assert v_dim is None, "Batching (v,) knots is not supported."

        (x, v, w) = args
        if w_dim is None:
            return self(x, v, w), out_dim
        else:
            w = jnp.moveaxis(w, w_dim, out_dim)
            prim = NU2NUPrimitive(
                x_aval=jax.typeof(x),
                v_aval=jax.typeof(v),
                w_aval=jax.typeof(w),  # batch dim included
                **self.params,
            )
            return prim(x, v, w), out_dim
