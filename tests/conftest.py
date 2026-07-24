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
    # NumPy: CPU only ----------------------------------------------------------
    try:
        np = importlib.import_module("numpy")
        yield pytest.param(
            ArrayBackend(
                name="numpy-cpu",
                xp=aac.array_namespace(np.asarray([0.0])),
                device="cpu",
            ),
            id="numpy-cpu",
        )
    except ModuleNotFoundError:
        pass

    # CuPy: GPU only -----------------------------------------------------------
    try:
        cp = importlib.import_module("cupy")
        if cp.is_available():
            yield pytest.param(
                ArrayBackend(
                    name="cupy-gpu",
                    xp=aac.array_namespace(cp.asarray([0.0])),
                    device=cp.cuda.Device(0),
                ),
                id="cupy-gpu",
            )
    except ModuleNotFoundError:
        pass

    # PyTorch: CPU + GPU -------------------------------------------------------
    try:
        torch = importlib.import_module("torch")
        torch_xp = aac.array_namespace(torch.asarray([0.0]))
        yield pytest.param(
            ArrayBackend(
                name="torch-cpu",
                xp=torch_xp,
                device="cpu",
            ),
            id="torch-cpu",
        )
        if torch.cuda.is_available():
            yield pytest.param(
                ArrayBackend(
                    name="torch-cuda",
                    xp=torch_xp,
                    device="cuda",
                ),
                id="torch-cuda",
            )
    except ModuleNotFoundError:
        pass

    # JAX: CPU + GPU -----------------------------------------------------------
    try:
        jax = importlib.import_module("jax")
        jax.config.update("jax_enable_x64", True)
        jnp = importlib.import_module("jax.numpy")
        for kind in ["cpu", "gpu"]:
            try:
                devices = jax.devices(kind)
            except RuntimeError:
                # occurs if hardware missing
                devices = []
            finally:
                if devices:
                    yield pytest.param(
                        ArrayBackend(
                            name=f"jax-{kind}",
                            xp=aac.array_namespace(jnp.asarray([0.0])),
                            device=devices[0],
                        ),
                        id=f"jax-{kind}",
                    )
    except ModuleNotFoundError:
        pass


@pytest.fixture(params=list(array_backend_cases()))
def array_backend(request) -> ArrayBackend:
    return request.param


def to_backend(x: ftkt.Array, array_backend: ArrayBackend) -> ftkt.Array:
    """
    Transform `x` to specific (backend,device).
    Input dtype is preserved.
    """
    return array_backend.xp.asarray(
        x,
        device=array_backend.device,
    )
