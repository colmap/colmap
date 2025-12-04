import contextlib
import ctypes
import os
import platform
import textwrap
from typing import TYPE_CHECKING

from .utils import import_module_symbols


def _preload_cuda_deps():
    if platform.system() != "Linux":
        return

    cuda_libs = [
        ("nvidia.cuda_runtime", "libcudart.so.12"),
        ("nvidia.curand", "libcurand.so.10"),
    ]

    for module_name, lib_name in cuda_libs:
        try:
            import importlib

            module = importlib.import_module(module_name)
        except ImportError:
            continue

        lib_dir = os.path.join(os.path.dirname(module.__file__), "lib")
        lib_path = os.path.join(lib_dir, lib_name)
        if os.path.exists(lib_path):
            try:
                ctypes.CDLL(lib_path)
            except contextlib.suppress(OSError):
                pass


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
