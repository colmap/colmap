.. _pycolmap/index:

PyCOLMAP
========

PyCOLMAP exposes to Python most capabilities of COLMAP.

Installation
------------

Pre-built wheels for Linux, macOS, and Windows can be installed using pip::

   pip install pycolmap

The wheels are automatically built and pushed to `PyPI <https://pypi.org/project/pycolmap/>`_ at each release. They are currently not built with CUDA support, which requires building from source. To build PyCOLMAP from source, follow these steps:

1. Install COLMAP from source following :ref:`installation`.
2. Build PyCOLMAP:

   * On Linux and macOS::

      python -m pip install ./pycolmap/

   * On Windows, after installing COLMAP via VCPKG, run in powershell::

      python -m pip install ./pycolmap/ `
          --cmake.define.CMAKE_TOOLCHAIN_FILE="$VCPKG_INSTALLATION_ROOT/scripts/buildsystems/vcpkg.cmake" `
          --cmake.define.VCPKG_TARGET_TRIPLET="x64-windows"

Some features, such as cost functions, require that `PyCeres <https://github.com/cvg/pyceres>`_ is installed in the same as PyCOLMAP, so either from PyPI or from source.

Usage
-----

.. toctree::
   :maxdepth: 2

   api
   cost_functions
