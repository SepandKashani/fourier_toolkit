from typing import Optional
from collections.abc import Iterable, Callable


__all__ = [
    "broadcast_seq",
]


def broadcast_seq(
    x,
    N: Optional[int] = None,
    cast: Callable = lambda _: _,
) -> tuple:
    """
    Broadcast `x` to a tuple of length `N`.

    If `N` is omitted, then no broadcasting takes place, only tupling.
    """
    if isinstance(x, Iterable):
        y = tuple(x)
    else:
        y = (x,)

    if N is not None:
        if len(y) == 1:
            y *= N  # broadcast
        assert len(y) == N

    y = tuple(map(cast, y))
    return y
