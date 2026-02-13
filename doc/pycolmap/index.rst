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

   * On Windows, after installing COLMAP via VCPKG per the installation guide above:

     1. Determine the installed COLMAP version:
        ``<VCPKG_ROOT>\packages\colmap_<TRIPLET>\tools\colmap\colmap.exe help``
     2. Check out the corresponding version tag: ``git checkout tags/3.XX.X``
     3. Run the following in PowerShell, replacing ``$VCPKG_ROOT`` with your vcpkg
        installation root::

           python -m pip install . `
               -C skbuild.cmake.define.CMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" `
               -C skbuild.cmake.define.VCPKG_TARGET_TRIPLET="x64-windows"

     If you get linker errors when building PyCOLMAP on Windows, ensure the
     repository version matches the COLMAP version installed via VCPKG.

Some features, such as cost functions, require that `PyCeres
<https://github.com/cvg/pyceres>`_ is installed in the same manner as PyCOLMAP,
so either from PyPI or from source.

API
-----

.. toctree::
   :maxdepth: 2

   pycolmap
   cost_functions
