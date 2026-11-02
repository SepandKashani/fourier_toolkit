import jax
import jax.experimental.hijax as hijax
import jax.numpy as jnp

import fourier_toolkit.typing as ftkt
import fourier_toolkit.util as ftku

from . import ExponentSign


def _u2u(
    x_spec: ftku.UniformSpec,
    v_spec: ftku.UniformSpec,
    w: ftkt.ArrayC,  # (..., M1,...,MD)
    isign: ExponentSign,
) -> ftkt.ArrayC:  # (..., N1,...,ND)
    prim = U2UPrimitive(
        w_aval=jax.typeof(w),
        x_spec=x_spec,
        v_spec=v_spec,
        isign=isign,
    )
    z = prim(w)
    return z


class U2UPrimitive(hijax.HiPrim):
    def __init__(
        self,
        w_aval: jax.core.ShapedArray,
        *,
        x_spec: ftku.UniformSpec,
        v_spec: ftku.UniformSpec,
        isign: ExponentSign,
    ):
        self.in_avals = (w_aval,)
        sh_stack = w_aval.shape[: -x_spec.ndim]
        self.out_aval = jax.core.ShapedArray(
            shape=(*sh_stack, *v_spec.num),
            dtype=w_aval.dtype,
        )
        self.params = dict(
            x_spec=x_spec,
            v_spec=v_spec,
            isign=isign,
        )
        super().__init__()

    def expand(self, w: ftkt.ArrayC) -> ftkt.ArrayC:
        from .generic import _u2u as generic_u2u

        z = generic_u2u(
            self.x_spec,
            self.v_spec,
            w,
            self.isign,
        )
        return z

    def jvp(
        self,
        primals: tuple[ftkt.ArrayC],
        tangents: tuple[ftkt.ArrayC],
    ) -> tuple[ftkt.ArrayC, ftkt.ArrayC]:
        (w,), (wt,) = primals, tangents
        return self(w), self(wt)

    def vjp_fwd(
        self,
        nzs_in: tuple[bool, ...],
        w: ftkt.ArrayC,
    ) -> tuple[ftkt.ArrayC, None]:
        return self(w), None

    def vjp_bwd_retval(
        self,
        res: None,
        v: ftkt.ArrayC,
    ) -> tuple[ftkt.ArrayC]:
        from .generic import _u2u as generic_u2u

        # vjp() computes (A^{\transpose} z), not (A^{\adjoint} z) ...
        z = generic_u2u(
            self.v_spec,
            self.x_spec,
            v,
            self.isign,  # ... reason why the sign matches the forward pass.
        )
        return (z,)

    def batch(
        self,
        axis_data,  # namedtuple-like
        args: tuple[ftkt.ArrayC],
        in_dims: tuple[int | None, ...],
    ) -> ftkt.ArrayC:
        # piggy-back on fact that generic.u2u() accepts stacked arrays
        (in_dim,) = in_dims
        out_dim = 0

        (w,) = args
        if in_dim is None:
            return self(w), out_dim
        else:
            w = jnp.moveaxis(w, in_dim, out_dim)
            prim = U2UPrimitive(
                w_aval=jax.typeof(w),  # batch dim included
                **self.params,
            )
            return prim(w), out_dim
