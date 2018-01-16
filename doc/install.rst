.. _installation:

Installation
============

You can either download one of the pre-built binaries from
https://demuc.de/colmap/ or build the source code manually.


------------------
Pre-built Binaries
------------------

Windows
-------

For convenience, the pre-built binaries for Windows contain both the graphical
and command-line interface executables. To start the COLMAP GUI, you can simply
double-click  the ``COLMAP.bat`` batch script or alternatively run it from the
Windows command shell or Powershell. The command-line interface is also
accessible through this batch script, which automatically sets the necessary
library paths. To list the available COLMAP commands, run ``COLMAP.bat -h`` in
the command shell ``cmd.exe`` or in Powershell.

Mac
---

The pre-built application package for Mac contains both the GUI and command-line
version of COLMAP. To open the GUI, simply open the application and note that
COLMAP is shipped as an unsigned application, i.e., when your first open the
application, you have to right-click the application and select *Open* and then
accept to trust the application. In the future, you can then simply double-click
the application to open COLMAP. The command-line interface is accessible by
running the packaged binary ``COLMAP.app/Contents/MacOS/colmap``. To list the
available COLMAP commands, run ``COLMAP.app/Contents/MacOS/colmap -h``.


-----------------
Build from Source
-----------------

COLMAP builds on all major platforms (Linux, Mac, Windows) with little effort.
First, checkout the latest source code::

    git clone https://github.com/colmap/colmap

The latest stable version lives in the ``master`` branch and the latest
development version lives in the ``dev`` branch.

On Linux and Mac it is generally recommended to follow the installation
instructions below, which use the system package managers to install the
required dependencies. Alternatively, there is a Python build script that builds
COLMAP and its dependencies locally. This script is useful under Windows and on
a (cluster) system if you do not have root access under Linux or Mac.


Linux
-----

*Recommended dependencies:* CUDA.

Dependencies from default Ubuntu repositories::

    sudo apt-get install \
        cmake \
        build-essential \
        libboost-all-dev \
        libeigen3-dev \
        libsuitesparse-dev \
        libfreeimage-dev \
        libgoogle-glog-dev \
        libgflags-dev \
        libglew-dev \
        qtbase5-dev \
        libqt5opengl5-dev

Install `Ceres Solver <http://ceres-solver.org/>`_::

    sudo apt-get install libatlas-base-dev libsuitesparse-dev
    git clone https://ceres-solver.googlesource.com/ceres-solver
    cd ceres-solver
    mkdir build
    cd build
    cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF
    make
    sudo make install

Configure and compile COLMAP::

    cd path/to/colmap
    mkdir build
    cd build
    cmake ..
    make
    sudo make install

Run COLMAP::

    colmap -h
    colmap gui


Mac
---

*Recommended dependencies:* CUDA.

Dependencies from `Homebrew <http://brew.sh/>`_::

    brew tap homebrew/science
    brew install \
        cmake \
        boost \
        eigen \
        freeimage \
        glog \
        gflags \
        suite-sparse \
        ceres-solver \
        qt5 \
        glew

Create the file ``LocalConfig.cmake`` in the COLMAP base directory and then
insert the following lines into it::

    set(Qt5_CMAKE_DIR "/usr/local/opt/qt5/lib/cmake")
    set(Qt5Core_DIR ${Qt5_CMAKE_DIR}/Qt5Core)
    set(Qt5OpenGL_DIR ${Qt5_CMAKE_DIR}/Qt5OpenGL)

Configure and compile COLMAP::

    cd path/to/colmap
    mkdir build
    cd build
    cmake ..
    make

Run COLMAP::

    colmap -h
    colmap gui


Windows
-------

*Recommended dependencies:* CUDA.

On Windows it is recommended to use the Python build script. Please follow the
instructions in the next section.

Alternatively, you can install the dependencies manually. To make the process of
configuring CMake less painful, please have a look at
``LocalConfigExample.config``. MSVC12 (Microsoft Visual Studio 2013) and newer
are confirmed to compile COLMAP without any issues.


Build Script
------------

COLMAP ships with an automated Python build script. The build script installs
COLMAP and its dependencies locally under Windows, Mac, and Linux. Note that
under Mac and Linux, it is usually easier and faster to use the available
package managers for the dependencies (see above). However, if you are on a
(cluster) system without root access, this script might be useful. This script
downloads the necessary dependencies automatically from the Internet. It assumes
that CMake, Boost, Qt5, and CUDA (optional) are already installed on the system.
E.g., under Windows you must specify the location of these libraries as::

    python scripts/python/build.py \
        --build_path path/to/colmap/build \
        --colmap_path path/to/colmap \
        --boost_path "C:/local/boost_1_64_0/lib64-msvc-14.0" \
        --qt_path "C:/Qt/5.9.3/msvc2015_64" \
        --cuda_path "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0"

Note that under Windows you must use forward slashes for specifying the paths
here. If you want to compile COLMAP using a specific Visual Studio version, you
can for example specify ``--cmake_generator "Visual Studio 14"`` for Visual
Studio 2015. If you want to open the COLMAP source code in Visual Studio, you
can open the solution file in ``path/to/colmap/build/colmap/build``.
If you use Homebrew under Mac, you can use the following command::

    python scripts/python/build.py \
        --build_path path/to/colmap/build \
        --colmap_path path/to/colmap \
        --qt5_path /usr/local/opt/qt/

To see the full list of command-line options, pass the ``--help`` argument.


-------------
Documentation
-------------

You need Python and Sphinx to build the HTML documentation::

    cd path/to/colmap/doc
    sudo apt-get install python
    pip install sphinx
    make html
    open _build/html/index.html

Alternatively, you can build the documentation as PDF, EPUB, etc.::

    make latexpdf
    open _build/pdf/COLMAP.pdf
