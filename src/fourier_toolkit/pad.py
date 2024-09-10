import collections.abc as cabc

import numpy as np

import fourier_toolkit.util as ftk_util

__all__ = [
    "Pad",
]


class Pad:
    r"""
    Multi-dimensional padding operator.

    This operator pads the input array in each dimension according to specified widths.

    Notes
    -----
    * If inputs are D-dimensional, then some of the padding of later axes are calculated from padding of previous axes.
    * The *adjoint* of the padding operator performs a cumulative summation over the original positions used to pad.
    """

    WidthSpec = int | tuple[int] | tuple[tuple[int, int]]
    ModeSpec = str | tuple[str]

    def __init__(
        self,
        dim_shape: tuple[int],
        pad_width: WidthSpec,
        mode: ModeSpec = "constant",
    ):
        r"""
        Parameters
        ----------
        dim_shape: tuple[int]
            (M1,...,MD) domain dimensions.
        pad_width: WidthSpec
            Number of values padded to the edges of each axis.
            Multiple forms are accepted:

            * ``int``: pad each dimension's head/tail by `pad_width`.
            * ``tuple[int, ...]``: pad dimension[k]'s head/tail by `pad_width[k]`.
            * ``tuple[tuple[int, int], ...]``: pad dimension[k]'s head/tail by `pad_width[k][0]` /
              `pad_width[k][1]` respectively.
        mode: str, tuple[str]
            Padding mode.
            Multiple forms are accepted:

            * str: unique mode shared amongst dimensions.
              Must be one of:

              * 'constant' (zero-padding)
              * 'wrap'
              * 'reflect'
              * 'symmetric'
              * 'edge'
            * tuple[str, ...]: pad dimension[k] using `mode[k]`.

            (See :py:func:`numpy.pad` for details.)
        """

        # Validate inputs =====================================================
        dim_shape = ftk_util.broadcast_seq(dim_shape, None, int)
        dim_rank = len(dim_shape)
        assert np.all(dim_shape >= 1)

        # transform `pad_width` to canonical form tuple[tuple[int, int], ...]
        is_seq = lambda _: isinstance(_, cabc.Iterable)
        if not is_seq(pad_width):  # int-form
            pad_width = ((pad_width, pad_width),) * dim_rank
        assert len(pad_width) == dim_rank, "dim_shape/pad_width are length-mismatched."
        if not is_seq(pad_width[0]):  # tuple[int, ...] form
            pad_width = tuple((w, w) for w in pad_width)
        else:  # tuple[tulpe[int, int], ...] form
            pass
        assert all(0 <= min(lhs, rhs) for (lhs, rhs) in pad_width)
        pad_width = tuple(pad_width)

        # transform `mode` to canonical form tuple[str, ...]
        if isinstance(mode, str):  # shared mode
            mode = (mode,) * dim_rank
        elif is_seq(mode):  # tuple[str, ...]: different modes
            assert len(mode) == dim_rank, "dim_shape/mode are length-mismatched."
            mode = tuple(mode)
        else:
            raise ValueError(f"Unkwown mode encountered: {mode}.")
        mode = tuple(map(lambda _: _.strip().lower(), mode))
        assert set(mode) <= {
            "constant",
            "wrap",
            "reflect",
            "symmetric",
            "edge",
        }, "Unknown mode(s) encountered."

        # Some modes have awkward interpretations when pad-widths cross certain thresholds.
        # Supported pad-widths are thus limited to sensible regions.
        for i in range(dim_rank):
            M = dim_shape[i]
            w_max = dict(
                constant=np.inf,
                wrap=M,
                reflect=M - 1,
                symmetric=M,
                edge=np.inf,
            )[mode[i]]
            lhs, rhs = pad_width[i]
            assert max(lhs, rhs) <= w_max, f"pad_width along dim-{i} is limited to {w_max}."
        # =====================================================================

        # store useful constants
        to_seq = lambda _: tuple(map(int, _))
        self._dim_shape = to_seq(dim_shape)
        self._dim_rank = dim_rank
        codim_shape = list(dim_shape)
        for i, (lhs, rhs) in enumerate(pad_width):
            codim_shape[i] += lhs + rhs
        self._codim_shape = to_seq(codim_shape)
        self._codim_rank = dim_rank
        self._pad_width = tuple(to_seq(_p) for _p in pad_width)
        self._mode = tuple(mode)

    def apply(self, x: np.ndarray) -> np.ndarray:
        r"""
        Parameters
        ----------
        x: ndarray
            (..., *dim_shape) signal.

        Returns
        -------
        y: ndarray
            (..., *codim_shape) padded output.
        """
        sh = x.shape[: -self._dim_rank]

        # Part 1: extend the core
        pad_width_sh = ((0, 0),) * len(sh)  # don't pad stack-dims
        y = np.pad(
            array=x,
            pad_width=pad_width_sh + self._pad_width,
            mode="constant",
            constant_values=0,
        )

        # Part 2: apply border effects (if any)
        for i in range(self._dim_rank, 0, -1):
            mode = self._mode[-i]
            lhs, rhs = self._pad_width[-i]
            N = self._codim_shape[-i]

            r_s = [slice(None)] * (len(sh) + self._dim_rank)  # read axial selector
            w_s = [slice(None)] * (len(sh) + self._dim_rank)  # write axial selector

            if mode == "constant":
                # no border effects
                pass
            elif mode == "wrap":
                if lhs > 0:  # Fix LHS
                    r_s[-i] = slice(N - rhs - lhs, N - rhs)
                    w_s[-i] = slice(0, lhs)
                    y[tuple(w_s)] = y[tuple(r_s)]

                if rhs > 0:  # Fix RHS
                    r_s[-i] = slice(lhs, lhs + rhs)
                    w_s[-i] = slice(N - rhs, N)
                    y[tuple(w_s)] = y[tuple(r_s)]
            elif mode == "reflect":
                if lhs > 0:  # Fix LHS
                    r_s[-i] = slice(2 * lhs, lhs, -1)
                    w_s[-i] = slice(0, lhs)
                    y[tuple(w_s)] = y[tuple(r_s)]

                if rhs > 0:  # Fix RHS
                    r_s[-i] = slice(N - rhs - 2, N - 2 * rhs - 2, -1)
                    w_s[-i] = slice(N - rhs, N)
                    y[tuple(w_s)] = y[tuple(r_s)]
            elif mode == "symmetric":
                if lhs > 0:  # Fix LHS
                    r_s[-i] = slice(2 * lhs - 1, lhs - 1, -1)
                    w_s[-i] = slice(0, lhs)
                    y[tuple(w_s)] = y[tuple(r_s)]

                if rhs > 0:  # Fix RHS
                    r_s[-i] = slice(N - rhs - 1, N - 2 * rhs - 1, -1)
                    w_s[-i] = slice(N - rhs, N)
                    y[tuple(w_s)] = y[tuple(r_s)]
            elif mode == "edge":
                if lhs > 0:  # Fix LHS
                    r_s[-i] = slice(lhs, lhs + 1)
                    w_s[-i] = slice(0, lhs)
                    y[tuple(w_s)] = y[tuple(r_s)]

                if rhs > 0:  # Fix RHS
                    r_s[-i] = slice(N - rhs - 1, N - rhs)
                    w_s[-i] = slice(N - rhs, N)
                    y[tuple(w_s)] = y[tuple(r_s)]

        return y

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        r"""
        Parameters
        ----------
        y: ndarray
            (..., *codim_shape) padded signal.

        Returns
        -------
        x: ndarray
            (..., *dim_shape) un-padded output.
        """
        sh = y.shape[: -self._codim_rank]

        # Part 1: apply correction terms (if any)
        x = y.copy()  # in-place updates below
        for i in range(1, self._codim_rank + 1):
            mode = self._mode[-i]
            lhs, rhs = self._pad_width[-i]
            N = self._codim_shape[-i]

            r_s = [slice(None)] * (len(sh) + self._codim_rank)  # read axial selector
            w_s = [slice(None)] * (len(sh) + self._codim_rank)  # write axial selector

            if mode == "constant":
                # no correction required
                pass
            elif mode == "wrap":
                if lhs > 0:  # Fix LHS
                    r_s[-i] = slice(0, lhs)
                    w_s[-i] = slice(N - rhs - lhs, N - rhs)
                    x[tuple(w_s)] += x[tuple(r_s)]

                if rhs > 0:  # Fix RHS
                    r_s[-i] = slice(N - rhs, N)
                    w_s[-i] = slice(lhs, lhs + rhs)
                    x[tuple(w_s)] += x[tuple(r_s)]
            elif mode == "reflect":
                if lhs > 0:  # Fix LHS
                    r_s[-i] = slice(lhs - 1, None, -1)
                    w_s[-i] = slice(lhs + 1, 2 * lhs + 1)
                    x[tuple(w_s)] += x[tuple(r_s)]

                if rhs > 0:  # Fix RHS
                    r_s[-i] = slice(N - 1, N - rhs - 1, -1)
                    w_s[-i] = slice(N - 2 * rhs - 1, N - rhs - 1)
                    x[tuple(w_s)] += x[tuple(r_s)]
            elif mode == "symmetric":
                if lhs > 0:  # Fix LHS
                    r_s[-i] = slice(lhs - 1, None, -1)
                    w_s[-i] = slice(lhs, 2 * lhs)
                    x[tuple(w_s)] += x[tuple(r_s)]

                if rhs > 0:  # Fix RHS
                    r_s[-i] = slice(N - 1, N - rhs - 1, -1)
                    w_s[-i] = slice(N - 2 * rhs, N - rhs)
                    x[tuple(w_s)] += x[tuple(r_s)]
            elif mode == "edge":
                if lhs > 0:  # Fix LHS
                    r_s[-i] = slice(0, lhs)
                    w_s[-i] = slice(lhs, lhs + 1)
                    x[tuple(w_s)] += x[tuple(r_s)].sum(axis=-i, keepdims=True)

                if rhs > 0:  # Fix RHS
                    r_s[-i] = slice(N - rhs, N)
                    w_s[-i] = slice(N - rhs - 1, N - rhs)
                    x[tuple(w_s)] += x[tuple(r_s)].sum(axis=-i, keepdims=True)

        # Part 2: extract the core
        selector = [slice(None)] * len(sh)
        for N, (lhs, rhs) in zip(self._codim_shape, self._pad_width):
            s = slice(lhs, N - rhs)
            selector.append(s)
        x = x[tuple(selector)]

        return x
