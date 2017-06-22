.. _installation:

Installation
============

You can either download one of the pre-built binaries from
https://demuc.de/colmap/ or build the source code manually. It builds on all
major platforms (Linux, Mac, Windows) with minimal effort. First, checkout the
latest source code::

    git clone https://github.com/colmap/colmap

Then follow the build instructions for your platform as detailed below.


Linux
-----

*Recommended dependencies:* CUDA.

Dependencies from default Ubuntu 14.04/16.04 repositories::

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
        qt5-default

Install `Ceres Solver <http://ceres-solver.org/>`_::

    sudo apt-get install libatlas-base-dev libsuitesparse-dev
    git clone https://ceres-solver.googlesource.com/ceres-solver
    cd ceres-solver
    mkdir build
    cd build
    cmake ..
    make -j
    sudo make install

Configure and compile COLMAP::

    cd path/to/colmap
    mkdir build
    cd build
    cmake ..
    make -j


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
    make -j


Windows
-------

On Windows you have to install the above dependencies manually. To make the
process of configuring CMake less painful, please have a look at
`LocalConfigExample.config`. MSVC12 (Microsoft Visual Studio 2013) and newer are
confirmed to compile COLMAP without any issues.

*Recommended dependencies:* CUDA.


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
