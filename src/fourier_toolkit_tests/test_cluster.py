import numpy as np
import pytest

import fourier_toolkit.cluster as ftk_cluster
import fourier_toolkit.util as ftk_util


class TestGridCluster:
    def test_value(self, space_dim, M, dtype):
        # * (in, out) shapes are consistent
        # * all points accounted for
        # * cl_info well formed
        # * cluster max extent <= bbox_dim
        # * #clusters <= max allowed count
        translate = ftk_util.TranslateDType(dtype)
        fdtype = translate.to_float()

        x, bbox_dim = self._generate_data(M, space_dim, fdtype)
        x_idx, cl_info = ftk_cluster.grid_cluster(x, bbox_dim)

        # (in, out) shapes are consistent
        assert cl_info.ndim == 1
        assert (x_idx.ndim == 1) and (len(x) == len(x_idx))

        M = len(x)
        Q = len(cl_info) - 1

        # All points accounted for
        assert M == len(x_idx)
        assert M == len(np.unique(x_idx))

        # cl_info well formed, i.e.
        # * cl_info is strictly monotonic -> no empty clusters
        # * cl_info[-1] goes beyond len(x)
        assert np.all(cl_info[1:] > cl_info[:-1])
        assert M == cl_info[-1]

        # cluster max extent <= bbox_dim
        for q in range(Q):
            select = slice(cl_info[q], cl_info[q + 1])
            x_cl = x[x_idx[select]]
            ptp_cl = np.ptp(x_cl, axis=0)  # (D,)
            assert np.all(ptp_cl <= bbox_dim)

        # #clusters <= max allowed count
        N_cl_max = 5**space_dim  # 5 hard-coded in _generate_data()
        assert Q <= N_cl_max

    def test_prec(self, space_dim, M, dtype):
        # output dtype correct.
        translate = ftk_util.TranslateDType(dtype)
        fdtype = translate.to_float()

        x, bbox_dim = self._generate_data(M, space_dim, fdtype)
        x_idx, cl_info = ftk_cluster.grid_cluster(x, bbox_dim)

        idtype = np.dtype(np.int64)
        assert x_idx.dtype == idtype
        assert cl_info.dtype == idtype

    # Fixtures ----------------------------------------------------------------
    @pytest.fixture(params=[1, 2, 3])
    def space_dim(self, request) -> int:
        return request.param

    @pytest.fixture(params=[1, 1_000_000])
    def M(self, request) -> int:
        return request.param

    @pytest.fixture(
        params=[
            np.float32,
            np.float64,
        ]
    )
    def dtype(self, request) -> np.dtype:
        # FP precisions to test implementation.
        return np.dtype(request.param)

    # Helper functions --------------------------------------------------------
    @staticmethod
    def _generate_data(M, D, dtype) -> tuple[np.ndarray]:
        # Create (x, bbox_dim) pairs to feed to grid_cluster().
        rng = np.random.default_rng()

        bbox_dim = rng.uniform(np.e, np.pi, D).astype(dtype)
        x = rng.uniform(0, 1, (M, D)).astype(dtype)
        x *= 5 * bbox_dim  # scale data to cover many grids
        x += rng.standard_normal(D)  # global offset

        return x, bbox_dim


class TestBisectCluster:
    @pytest.mark.parametrize("Q", [1, 500])
    @pytest.mark.parametrize("N_max", [1, 1_001])
    def test_value(self, Q, N_max):
        # * clB_info well formed
        # * #clusters >= input #clusters
        # * cluster size <= N_max

        cl_info = self._generate_data(Q)
        clB_info = ftk_cluster.bisect_cluster(cl_info, N_max)

        Q = len(cl_info) - 1
        L = len(clB_info) - 1
        M = cl_info[-1]

        # clB_info well formed, i.e.
        # * clB_info is strictly monotonic -> no empty clusters
        # * clB_info[-1] goes beyond len(x)
        assert np.all(clB_info[1:] > clB_info[:-1])
        assert 0 == clB_info[0]
        assert M == clB_info[-1]

        # #clusters >= input #clusters
        assert Q <= L

        # cluster size <= N_max
        cl_size = clB_info[1:] - clB_info[:-1]
        assert np.all(cl_size <= N_max)

    @pytest.mark.parametrize("Q", [1, 500])
    def test_prec(self, Q):
        # output dtype correct.
        cl_info = self._generate_data(Q)
        clB_info = ftk_cluster.bisect_cluster(cl_info, N_max=2)

        assert clB_info.dtype == cl_info.dtype

    # Helper functions --------------------------------------------------------
    @staticmethod
    def _generate_data(Q) -> np.ndarray:
        # Create (cl_info,) to feed to bisect_cluster().
        rng = np.random.default_rng()

        cl_info = np.zeros(Q + 1, dtype=np.int64)
        cl_info[1:] = rng.integers(low=1, high=10_000, size=Q).cumsum()

        return cl_info


class TestFuseCluster:
    def test_value(self, space_dim, M, dtype):
        # * (in, out) shapes are consistent
        # * all points accounted for
        # * clF_info well formed
        # * cluster max extent <= bbox_dim
        # * #clusters <= max allowed count
        # * #clusters <= initial #clusters
        translate = ftk_util.TranslateDType(dtype)
        fdtype = translate.to_float()

        x, x_idx, cl_info, bbox_dim = self._generate_data(M, space_dim, fdtype)
        xF_idx, clF_info = ftk_cluster.fuse_cluster(x, x_idx, cl_info, bbox_dim)

        # (in, out) shapes are consistent
        assert clF_info.ndim == 1
        assert (xF_idx.ndim == 1) and (len(x) == len(xF_idx))

        M = len(x)
        L = len(clF_info) - 1

        # All points accounted for
        assert M == len(xF_idx)
        assert M == len(np.unique(xF_idx))

        # clF_info well formed, i.e.
        # * clF_info is strictly monotonic -> no empty clusters
        # * clF_info[-1] goes beyond len(x)
        assert np.all(clF_info[1:] > clF_info[:-1])
        assert M == clF_info[-1]

        # cluster max extent <= bbox_dim
        bbox_dim = np.array(bbox_dim)
        for q in range(L):
            select = slice(clF_info[q], clF_info[q + 1])
            x_cl = x[xF_idx[select]]
            ptp_cl = np.ptp(x_cl, axis=0)  # (D,)
            assert np.all(ptp_cl <= bbox_dim)

        # #clusters <= max allowed count
        N_cl_max = 10**space_dim  # 10 hard-coded in _generate_data()
        assert L <= N_cl_max

        # #clusters <= initial #clusters
        assert len(clF_info) <= len(cl_info)

    def test_prec(self, space_dim, M, dtype):
        # output dtype correct.
        translate = ftk_util.TranslateDType(dtype)
        fdtype = translate.to_float()

        x, x_idx, cl_info, bbox_dim = self._generate_data(M, space_dim, fdtype)
        xF_idx, clF_info = ftk_cluster.fuse_cluster(x, x_idx, cl_info, bbox_dim)

        idtype = np.dtype(np.int64)
        assert xF_idx.dtype == idtype
        assert clF_info.dtype == idtype

    # Fixtures ----------------------------------------------------------------
    @pytest.fixture(params=[1, 2, 3])
    def space_dim(self, request) -> int:
        return request.param

    @pytest.fixture(params=[1, 1_000_000])
    def M(self, request) -> int:
        return request.param

    @pytest.fixture(
        params=[
            np.float32,
            np.float64,
        ]
    )
    def dtype(self, request) -> np.dtype:
        # FP precisions to test implementation.
        return np.dtype(request.param)

    # Helper functions --------------------------------------------------------
    @staticmethod
    def _generate_data(M, D, dtype) -> tuple[np.ndarray]:
        # Create (x, x_idx, cl_info, bbox_dim) pairs to feed to fuse_cluster().
        rng = np.random.default_rng()

        bbox_dim = rng.uniform(np.e, np.pi, D).astype(dtype)
        x = rng.uniform(0, 1, (M, D)).astype(dtype)
        x *= 10 * bbox_dim  # scale data to cover many grids
        x += rng.standard_normal(D)  # global offset

        x_idx, cl_info = ftk_cluster.grid_cluster(x, bbox_dim)
        bbox_dim_fuse = bbox_dim * 2

        return x, x_idx, cl_info, bbox_dim_fuse
