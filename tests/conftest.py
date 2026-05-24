import importlib
from dataclasses import dataclass

import array_api_compat as aac
import pytest

import fourier_toolkit.typing as ftkt


@dataclass(frozen=True)
class ArrayBackend:
    name: str
    xp: ftkt.ArrayNameSpace
    device: object | None


def array_backend_cases():
    # NumPy: CPU only
    np = pytest.importorskip("numpy")
    yield pytest.param(
        ArrayBackend(
            name="numpy-cpu",
            xp=aac.array_namespace(np.asarray([0.0])),
            device="cpu",
        ),
        id="numpy-cpu",
    )

    # CuPy: GPU only
    cupy = pytest.importorskip("cupy")
    if cupy.is_available():
        yield pytest.param(
            ArrayBackend(
                name="cupy-gpu",
                xp=aac.array_namespace(cupy.asarray([0.0])),
                device=cupy.cuda.Device(0),
            ),
            id="cupy-gpu",
        )

    # JAX: CPU and GPU if available
    jax = pytest.importorskip("jax")
    jnp = importlib.import_module("jax.numpy")
    for kind in ["cpu", "gpu"]:
        devices = jax.devices(kind)
        if devices:
            yield pytest.param(
                ArrayBackend(
                    name=f"jax-{kind}",
                    xp=aac.array_namespace(jnp.asarray([0.0])),
                    device=devices[0],
                ),
                id=f"jax-{kind}",
            )

    # PyTorch: CPU and CUDA if available
    torch = pytest.importorskip("torch")
    torch_xp = aac.array_namespace(torch.asarray([0.0]))

    yield pytest.param(
        ArrayBackend("torch-cpu", torch_xp, "cpu"),
        id="torch-cpu",
    )

    if torch.cuda.is_available():
        yield pytest.param(
            ArrayBackend("torch-cuda", torch_xp, "cuda"),
            id="torch-cuda",
        )


@pytest.fixture(params=list(array_backend_cases()))
def array_backend(request) -> ArrayBackend:
    return request.param
