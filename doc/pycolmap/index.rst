.. _pycolmap/index:

PyCOLMAP
========

PyCOLMAP exposes to Python most capabilities of COLMAP.

Installation
------------

Pre-built wheels for Linux, macOS, and Windows can be installed using pip::

   pip install pycolmap

The wheels are automatically built and pushed to `PyPI
<https://pypi.org/project/pycolmap/>`_ at each release.
To benefit from GPU acceleration, wheels built for CUDA 12 (only for Linux - for now)
are available under the `package pycolmap-cuda12 <https://pypi.org/project/pycolmap-cuda12/)>`_.

To build PyCOLMAP from source, follow these steps:

1. Install COLMAP from source following :ref:`installation`.
2. Build PyCOLMAP:

   * On Linux and macOS::

      python -m pip install .

   * On Windows, after installing COLMAP via VCPKG, run in powershell::

      python -m pip install . `
          --cmake.define.CMAKE_TOOLCHAIN_FILE="$VCPKG_INSTALLATION_ROOT/scripts/buildsystems/vcpkg.cmake" `
          --cmake.define.VCPKG_TARGET_TRIPLET="x64-windows"

Some features, such as cost functions, require that `PyCeres
<https://github.com/cvg/pyceres>`_ is installed in the same manner as PyCOLMAP,
so either from PyPI or from source.

API
-----

.. toctree::
   :maxdepth: 2

   pycolmap
   cost_functions
