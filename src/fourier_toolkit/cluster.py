import numpy as np

import fourier_toolkit.numba as ftk_numba

__all__ = [
    "grid_cluster",
    "bisect_cluster",
    "fuse_cluster",
]


def grid_cluster(
    x: np.ndarray,
    bbox_dim: np.ndarray,
) -> tuple[np.ndarray]:
    """
    Split D-dimensional points onto lattice-aligned clusters.
    Each cluster may contain arbitrary-many points.

    Parameters
    ----------
    x: ndarray[float]
        (M, D) point cloud
    bbox_dim: ndarray[float]
        (D,) box dimensions

    Returns
    -------
    x_idx: ndarray[int]
        (M,) indices that sort `x` along axis 0.
    cl_info: ndarray[int]
        (Q+1,) cluster start/stop indices.

        ``x[x_idx][cl_info[q] : cl_info[q+1]]`` contains all points in the q-th cluster.
    """
    M, D = x.shape
    idtype, fdtype = np.int64, x.dtype
    bbox_dim = np.array(bbox_dim, dtype=fdtype)
    assert (len(bbox_dim) == D) and np.all(bbox_dim > 0)

    # Quick exit if only one point.
    if M == 1:
        x_idx = np.array([0], dtype=idtype)
        cl_info = np.array([0, 1], dtype=idtype)
    else:
        # Compute cluster index of each point
        c_idx, lattice_shape = ftk_numba.digitize(x, bbox_dim)

        # Re-order & count points
        cl_count, x_idx = ftk_numba.count_sort(c_idx, k=lattice_shape.prod())

        # Encode `cl_info`
        Q = len(cl_count)
        cl_info = np.zeros(Q + 1, dtype=idtype)
        cl_info[1:] = cl_count.cumsum()
    return x_idx, cl_info


def bisect_cluster(
    cl_info: np.ndarray,
    N_max: int,
) -> np.ndarray:
    """
    Hierarchically split clusters until each contains at most `N_max` points.

    Parameters
    ----------
    cl_info: ndarray[int]
        (Q+1,) cluster start/stop indices.
    N_max: int
        Maximum number of points allocated per cluster.

    Returns
    -------
    clB_info: ndarray[int]
        (L+1,) bisected cluster start/stop indices.
    """
    idtype = cl_info.dtype

    M = cl_info[-1]
    Q = len(cl_info) - 1
    assert N_max > 0

    cl_size = cl_info[1:] - cl_info[:-1]
    cl_chunks = np.ceil(cl_size / N_max).astype(idtype)
    L = sum(cl_chunks)

    clB_info = np.full(L + 1, fill_value=M, dtype=idtype)
    _l = 0
    for q in range(Q):
        for c in range(cl_chunks[q]):
            clB_info[_l] = cl_info[q] + min(c * N_max, cl_size[q])
            _l += 1
    return clB_info


def fuse_cluster(
    x: np.ndarray,
    x_idx: np.ndarray,
    cl_info: np.ndarray,
    bbox_dim: np.ndarray,
) -> tuple[np.ndarray]:
    """
    Fuse neighboring clusters until aggregate bounding-boxes have at most size `bbox_dim`.

    It is assumed all clusters passed in already satisfy this pre-condition.

    Parameters
    ----------
    x: ndarray[float]
        (M, D) point cloud.
    x_idx: ndarray[int]
        (M,) indices which sort `x` into clusters.
    cl_info: ndarray[int]
        (Q+1,) cluster start/stop indices.

        ``x[x_idx][cl_info[q] : cl_info[q+1]]`` contains all points in the q-th cluster.
    bbox_dim: ndarray[float]
        (D,) maximum (fused) box dimensions.

    Returns
    -------
    xF_idx: ndarray[int]
        (M,) indices which sort `x` into *fused* clusters.
    clF_info: ndarray[int]
        (L+1,) fused cluster start/stop indices.

        ``x[xF_idx][clF_info[l] : clF_info[l+1]]`` contains all points in the l-th cluster.
    """
    M, D = x.shape
    idtype, fdtype = np.int64, x.dtype
    bbox_dim = np.array(bbox_dim, dtype=fdtype)
    assert (len(bbox_dim) == D) and np.all(bbox_dim > 0)

    Q = len(cl_info) - 1

    # Quit exit if only one cluster.
    if Q == 1:
        xF_idx = x_idx
        clF_info = cl_info
    else:
        # Initialize state variables:
        # * cl_LL, cl_UR: dict[int, float(D,)]
        # * cl_group: dict[int, set]
        _cl_LL, _cl_UR = ftk_numba.group_minmax(x, x_idx, cl_info)
        cl_LL = ftk_numba.dict_factory(fdtype)
        cl_UR = ftk_numba.dict_factory(fdtype)
        for q in range(Q):
            cl_LL[q] = _cl_LL[q]
            cl_UR[q] = _cl_UR[q]
        cl_group = {q: {q} for q in range(Q)}

        # Fuse clusters until completion
        clusters = set(range(Q))
        q = Q  # fused cluster index
        while len(clusters) > 1:
            fuseable, (i, j) = ftk_numba.fuseable_candidate(cl_LL, cl_UR, bbox_dim)
            if fuseable:
                bbox_LL = np.fmin(cl_LL[i], cl_LL[j])
                cl_LL[q] = bbox_LL
                cl_LL.pop(i), cl_LL.pop(j)

                bbox_UR = np.fmax(cl_UR[i], cl_UR[j])
                cl_UR[q] = bbox_UR
                cl_UR.pop(i), cl_UR.pop(j)

                cl_group[q] = cl_group[i] | cl_group[j]
                cl_group.pop(i), cl_group.pop(j)

                clusters.add(q)
                clusters.remove(i), clusters.remove(j)
                q += 1
            else:
                break

        # Encode (xF_idx, clF_info)
        L = len(cl_group)
        clF_info = np.full(L + 1, fill_value=M, dtype=idtype)
        xF_idx = np.empty(M, dtype=idtype)
        offset = 0
        for _l, cluster_ids in enumerate(cl_group.values()):
            clF_info[_l] = offset
            for q in cluster_ids:
                a, b = cl_info[q], cl_info[q + 1]
                xF_idx[offset : offset + (b - a)] = x_idx[a:b]
                offset += b - a
    return xF_idx, clF_info
