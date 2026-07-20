import jax
import jax.experimental.hijax as hijax
import jax.numpy as jnp

import fourier_toolkit.typing as ftkt
import fourier_toolkit.util as ftku

from . import ExponentSign


def _u2u(
    x_spec: ftku.UniformSpec,
    v_spec: ftku.UniformSpec,
    w: ftkt.ArrayRC,  # (..., M1,...,MD)
    isign: ExponentSign,
) -> ftkt.ArrayC:  # (..., N1,...,ND)
    prim = U2UPrimitive(
        w_aval=jax.core.ShapedArray(
            shape=x_spec.num,
            dtype=w.dtype,
        ),
        x_spec=x_spec,
        v_spec=v_spec,
        isign=isign,
    )
    z = prim(w)
    return z


class U2UPrimitive(hijax.VJPHiPrimitive):
    def __init__(
        self,
        w_aval: jax.core.ShapedArray,
        *,
        x_spec: ftku.UniformSpec,
        v_spec: ftku.UniformSpec,
        isign: ExponentSign,
    ):
        self.in_avals = (w_aval,)
        self.out_aval = jax.core.ShapedArray(
            shape=v_spec.num,
            # in_avals may be R/C-valued, but out_avals is always C-valued
            dtype=jnp.promote_types(w_aval.dtype, complex),
        )
        self.params = dict(
            x_spec=x_spec,
            v_spec=v_spec,
            isign=isign,
        )
        super().__init__()

    def expand(self, w: ftkt.ArrayRC) -> ftkt.ArrayC:
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
        primals: tuple[ftkt.ArrayRC],
        tangents: tuple[ftkt.ArrayRC],
    ) -> tuple[ftkt.ArrayC, ftkt.ArrayC]:
        (w1,), (w2,) = primals, tangents
        return self(w1), self(w2)

    def vjp_fwd(
        self,
        nzs_in: tuple[bool, ...],
        w: ftkt.ArrayRC,
    ) -> tuple[ftkt.ArrayC, None]:
        return self(w), None

    def vjp_bwd_retval(
        self,
        res: None,
        v: ftkt.ArrayRC,
    ) -> tuple[ftkt.ArrayC]:
        from .generic import _u2u as generic_u2u

        z = generic_u2u(
            self.v_spec,
            self.x_spec,
            v,
            self.isign,
            # notice (isign) vs (-isign) here: this is due to JAX's VJP conventions as seen here:
            # https://docs.jax.dev/en/latest/complex-differentiation.html (search for "What about VJPs?")
        )
        return (z,)

    def batch(
        self,
        axis_data,  # namedtuple-like
        args: tuple[ftkt.ArrayRC],
        in_dims: tuple[int | None, ...],
    ) -> ftkt.ArrayC:
        # piggy-back on fact that generic.u2u() accepts stacked arrays
        (in_dim,) = in_dims
        out_dim = 0

        (w,) = args
        w = jnp.moveaxis(w, in_dim, out_dim)

        return self(w), out_dim
