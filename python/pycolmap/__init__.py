import contextlib
import ctypes
import importlib
import platform
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

from .utils import import_module_symbols


def _preload_cuda_lib(module_name: str, lib_name: str):
    """Preload a single library."""
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return
    else:
        # TODO: update the logic to handle CUDA 13,
        # using as reference https://github.com/pytorch/pytorch/pull/163661.

        # Resolve the library directory robustly
        if module_file_path := getattr(module, "__file__", None):
            lib_dir = Path(module_file_path).parent
        elif paths := getattr(module, "__path__", None):
            # Implicit namespace packages have __path__ but no __file__
            lib_dir = Path(list(paths)[0])
        else:
            return
        # Find the first file matching the pattern
        if lib_path := next((lib_dir / "lib").glob(lib_name), None):
            with contextlib.suppress(OSError):
                ctypes.CDLL(str(lib_path))


def _preload_cuda_deps():
    """Preloads CUDA dependencies from pip packages on Linux."""
    if platform.system() != "Linux":
        return
    cuda_libs = {
        "nvidia.cuda_runtime": "libcudart.so.*[0-9]",
        "nvidia.curand": "libcurand.so.*[0-9]",
    }
    for module_name, lib_glob in cuda_libs.items():
        _preload_cuda_lib(module_name, lib_glob)


_preload_cuda_deps()

try:
    from . import _core
except ImportError as e:
    raise RuntimeError(
        textwrap.dedent("""
        Cannot import the C++ backend pycolmap._core.
        Make sure that you successfully install the package with
          $ python -m pip install pycolmap/
        """)
    ) from e

# Type checkers cannot deal with dynamic manipulation of globals.
# Instead, we use the same workaround as PyTorch.
if TYPE_CHECKING:
    from ._core import *  # noqa F403

__all__ = import_module_symbols(globals(), _core, exclude={"cost_functions"})
__all__.extend(["__version__", "__ceres_version__"])

__version__ = _core.__version__
__ceres_version__ = _core.__ceres_version__
