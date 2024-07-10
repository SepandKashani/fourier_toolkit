import numba as nb
import numba.types as nbt
import numpy as np

__all__ = [
    "count_sort",
    "digitize",
    "filter_to_bbox",
    "group_minmax",
    "minmax",
]

_nb_flags = dict(
    nopython=True,
    nogil=True,
    cache=True,
    forceobj=False,
    # parallel=False,  # set manually per compiled func
    error_model="numpy",
    fastmath=True,
    locals={},
    boundscheck=False,
)

# boolean types -----------------------
b_t = nbt.boolean
b_1C_t = nbt.Array(b_t, 1, "C")
# integer types -----------------------
i_t = nbt.int64
i_1C_t = nbt.Array(i_t, 1, "C")
i_2C_t = nbt.Array(i_t, 2, "C")
# 32-bit floating types ---------------
f32_t = nbt.float32
f32_1C_t = nbt.Array(f32_t, 1, "C")
f32_2C_t = nbt.Array(f32_t, 2, "C")
# 64-bit floating types ---------------
f64_t = nbt.float64
f64_1C_t = nbt.Array(f64_t, 1, "C")
f64_2C_t = nbt.Array(f64_t, 2, "C")


@nb.jit(
    [
        nbt.UniTuple(f32_1C_t, 2)(f32_2C_t),
        nbt.UniTuple(f64_1C_t, 2)(f64_2C_t),
    ],
    **_nb_flags,
    parallel=False,
)
def minmax(x: np.ndarray) -> tuple[np.ndarray]:
    # Computes code below more efficiently:
    #     (x.min(axis=0), x.max(axis=0))
    #
    # Parameters
    # ----------
    # x: ndarray[float]
    #     (M, D)
    #
    # Returns
    # -------
    # x_min, x_max: ndarray[float]
    #     (D,) axial min/max values
    M, D = x.shape
    x_min = np.full(D, fill_value=np.inf, dtype=x.dtype)
    x_max = np.full(D, fill_value=-np.inf, dtype=x.dtype)

    for m in range(M):
        for d in range(D):
            x_min[d] = min(x_min[d], x[m, d])
            x_max[d] = max(x_max[d], x[m, d])
    return x_min, x_max


@nb.jit(
    [
        nbt.UniTuple(f32_2C_t, 2)(f32_2C_t, i_1C_t, i_1C_t),
        nbt.UniTuple(f64_2C_t, 2)(f64_2C_t, i_1C_t, i_1C_t),
    ],
    **_nb_flags,
    parallel=True,
)
def group_minmax(
    x: np.ndarray,
    x_idx: np.ndarray,
    limits: np.ndarray,
) -> tuple[np.ndarray]:
    # Computes code below more efficiently:
    #     _, D = x.shape
    #     Q = len(limits) - 1
    #     x_min = np.empty((Q, D), dtype=x.dtype)
    #     x_max = np.empty((Q, D), dtype=x.dtype)
    #     for q in range(Q):
    #         select = slice(limits[q], limits[q + 1])
    #         _x = x[x_idx[select], :]
    #         x_min[q,:], x_max[q,:] = minmax(_x)
    #
    # Parameters
    # ----------
    # x: ndarray[float]
    #     (M, D)
    # x_idx: ndarray[int]
    #     (M,) re-order buffer
    # limits: ndarray[int]
    #     (Q+1,) group start/stop indices
    #
    # Returns
    # -------
    # x_min, x_max: ndarray[float]
    #     (Q, D) axial min/max values per group
    _, D = x.shape
    Q = len(limits) - 1
    x_min = np.empty((Q, D), dtype=x.dtype)
    x_max = np.empty((Q, D), dtype=x.dtype)
    for q in nb.prange(Q):
        select = slice(limits[q], limits[q + 1])
        _x = x[x_idx[select], :]
        x_min[q, :], x_max[q, :] = minmax(_x)
    return x_min, x_max


@nb.jit(
    [
        nbt.UniTuple(i_1C_t, 2)(f32_2C_t, f32_1C_t),
        nbt.UniTuple(i_1C_t, 2)(f64_2C_t, f64_1C_t),
    ],
    **_nb_flags,
    parallel=True,
)
def digitize(
    x: np.ndarray,
    bbox_dim: np.ndarray,
) -> tuple[np.ndarray]:
    # Computes code below more efficiently:
    #     x_min, x_max = x.min(axis=0), x.max(axis=0)
    #     lattice_shape = np.maximum(
    #                         1,
    #                         np.ceil((x_max - x_min) / bbox_dim),
    #                     ).astype(int)
    #     cM_idx = ((x - x_min) / bbox_dim).astype(int)  # (M, D)
    #     c_idx = np.ravel_multi_index(cM_idx.T, lattice_shape)  # (M,)
    #
    # Parameters
    # ----------
    # x: ndarray[float]
    #     (M, D)
    # bbox_dim: ndarray[float]
    #     (D,)
    #
    # Returns
    # -------
    # c_idx: ndarray[int]
    #     (M,) integer bins each element belongs to.
    # lattice_shape: ndarray[int]
    #     (D,) bin count per dimension.
    M, D = x.shape
    x_min, x_max = minmax(x)

    lattice_shape = np.zeros(D, dtype=i_t)
    for d in range(D):
        lattice_shape[d] = max(
            1,
            np.ceil((x_max[d] - x_min[d]) / bbox_dim[d]),
        )

    # Compute (multi-index -> index) stride
    stride = np.ones(D, dtype=i_t)
    for d in range(D - 2, -1, -1):
        stride[d] = stride[d + 1] * lattice_shape[d + 1]

    c_idx = np.zeros(M, dtype=i_t)
    for m in nb.prange(M):
        cM_idx = np.zeros(D, dtype=i_t)  # temporary buffer

        # Compute multi-index
        for d in range(D):
            cM_idx[d] = min(
                (x[m, d] - x_min[d]) / bbox_dim[d],
                lattice_shape[d] - 1,
            )

        # Then convert to flat-index
        for d in range(D):
            c_idx[m] += stride[d] * cM_idx[d]

    return c_idx, lattice_shape


@nb.jit(
    nbt.UniTuple(i_1C_t, 2)(i_1C_t, i_t),
    **_nb_flags,
    parallel=False,
)
def count_sort(
    x: np.ndarray,
    k: int,
) -> tuple[np.ndarray]:
    # Computes code below more efficiently:
    #     idx = np.argsort(x)
    #     _, count = np.unique(x, return_counts=True)
    #
    # Parameters
    # ----------
    # x: ndarray[int]
    #     (N,) non-negative integers in {0,...,max(x)}.
    # k: int
    #     Upper bound on values in `x`.
    #     Must satisfy `k >= max(x)`.
    #
    # Returns
    # -------
    # count: ndarray[int]
    #     (Q,) counts of each element in `x`.
    # idx: ndarray[int]
    #     (N,) indices to sort `x` into ascending order.
    count = np.zeros(k, dtype=i_t)
    for _x in x:
        count[_x] += 1

    # Write-index for each category
    w_idx = np.zeros(k, dtype=i_t)
    w_idx[0] = 0
    w_idx[1:] = np.cumsum(count)[: k - 1]

    N = len(x)
    idx = np.zeros(N, dtype=i_t)
    for i, _x in enumerate(x):
        idx[w_idx[_x]] = i
        w_idx[_x] += 1

    # `count` contains zero entries -> trim to non-zero segments in-place
    i = 0
    for c in count:
        if c > 0:
            count[i] = c
            i += 1
    count = count[:i]

    return count, idx


@nb.jit(
    [
        b_1C_t(f32_2C_t, f32_1C_t, f32_1C_t, f32_1C_t),
        b_1C_t(f64_2C_t, f64_1C_t, f64_1C_t, f64_1C_t),
    ],
    **_nb_flags,
    parallel=True,
)
def filter_to_bbox(
    x: np.ndarray,
    lhs: np.ndarray,
    rhs: np.ndarray,
    width: np.ndarray,
) -> np.ndarray:
    # Computes code below more efficiently:
    #     mask = np.all(lhs - width <= x, axis=1) &
    #            np.all(x <= rhs + width, axis=1)
    #
    # Parameters
    # ----------
    # x: ndarray[float]
    #     (M, D)
    # lhs, rhs, width: ndarray[float]
    #     (D,)
    #
    # Returns
    # -------
    # mask: ndarray[bool]
    #     (M,)
    M, D = x.shape

    LL = np.zeros(D, dtype=x.dtype)
    UR = np.zeros(D, dtype=x.dtype)
    for d in range(D):
        LL[d] = lhs[d] - width[d]
        UR[d] = rhs[d] + width[d]

    mask = np.empty(M, dtype=b_t)
    for m in nb.prange(M):
        inside = True
        for d in range(D):
            inside &= LL[d] <= x[m, d] <= UR[d]
        mask[m] = inside

    return mask
