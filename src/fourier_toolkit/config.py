# Config information which may be needed by some sub-modules.

import hashlib
import importlib
import os
import pathlib as plib
import string
import sys
import types

__all__ = [
    "cache_dir",
    "config_dir",
    "data_dir",
    "cache_module",
    "generate_module",
]

resolve = lambda p: plib.Path(p).expanduser().resolve()


def xdg_config_root() -> plib.Path:
    return resolve(os.getenv("XDG_CONFIG_HOME", "~/.config"))


def xdg_data_root() -> plib.Path:
    return resolve(os.getenv("XDG_DATA_HOME", "~/.local/share"))


def xdg_cache_root() -> plib.Path:
    return resolve(os.getenv("XDG_CACHE_HOME", "~/.cache"))


def config_dir() -> plib.Path:
    # config files (if any)
    return xdg_config_root() / "fourier_toolkit"


def data_dir() -> plib.Path:
    # shipped data (if any)
    return xdg_data_root() / "fourier_toolkit"


def cache_dir() -> plib.Path:
    # runtime-generated stuff (if any)
    return xdg_cache_root() / "fourier_toolkit"


def cache_module(code: str) -> types.ModuleType:
    """
    Save `code` as an importable module in the dynamic module cache.

    The cached module is updated only if changes are detected.

    Parameters
    ----------
    code: str
        Contents of the module.

        When stored in a file, `code` should be a valid Python module.

    Returns
    -------
    module
        The compiled module
    """
    # Compute a unique name
    h = hashlib.blake2b(code.encode("utf-8"), digest_size=8)
    module_name = "cached_" + h.hexdigest()
    module_path = cache_dir() / f"{module_name}.py"

    # Do we overwrite?
    write = True
    if module_path.exists():
        with open(module_path, mode="r") as f:
            old_content = f.read()
        if old_content == code:
            write = False

    if write:
        with open(module_path, mode="w") as f:
            f.write(code)

    pkg = importlib.import_module(module_name)
    return pkg


def generate_module(path: plib.Path, subs: dict) -> types.ModuleType:
    """
    * Load code as a string from `path`;
    * Substitute ${}-terms using `subs`;
    * Return compiled module.
    """
    with open(path, mode="r") as f:
        template = string.Template(f.read())
    code = template.substitute(**subs)
    return cache_module(code)


# -----------------------------------------------------------------------------
# Add cache directory to PYTHONPATH.
# This allows dynamic import of runtime-compiled modules.
cdir = cache_dir()
cdir.mkdir(parents=True, exist_ok=True)
if str(cdir) not in sys.path:
    sys.path.append(str(cdir))
