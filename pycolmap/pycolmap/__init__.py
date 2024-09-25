from ._core import *  # noqa F403

# This is unfortunately needed otherwise the C++ modules will shadow the
# equivalent python modules, which already import them. We can remove this
# once pycolmap._core has few declarations such that they can be explicitly
# imported.
del cost_functions, manifold  # noqa F821

from . import _core  # noqa E402

__all__ = [n for n in _core.__dict__ if not n.startswith("_")]
__all__.extend(["__version__", "__ceres_version__"])

__version__ = _core.__version__
__ceres_version__ = _core.__ceres_version__
