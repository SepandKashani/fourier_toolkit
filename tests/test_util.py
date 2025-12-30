import pytest
import fourier_toolkit.util as ftku


class TestBroadcastSeq:
    def test_tuplizes(self):
        assert ftku.broadcast_seq(x=1) == (1,)
        assert ftku.broadcast_seq(x=[1]) == (1,)
        assert ftku.broadcast_seq(x=np.r_[1, 2]) == (1, 2)

    def test_broadcasts(self):
        assert ftku.broadcast_seq(x=1, N=1) == (1,)
        assert ftku.broadcast_seq(x=1, N=2) == (1, 1)
        assert ftku.broadcast_seq(x=1, N=3) == (1, 1, 1)

        with pytest.raises(Exception):
            ftku.broadcast_seq(x=(1, 2), N=3)

    def test_casts(self):
        assert ftku.broadcast_seq(x=1.1, cast=int) == (1,)
