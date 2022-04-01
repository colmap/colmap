.. _installation:

Installation
============

You can either download one of the pre-built binaries or build the source code
manually. Executables for Windows and Mac and other resources can be downloaded
from https://demuc.de/colmap/. Executables for Linux/Unix/BSD are available at
https://repology.org/metapackage/colmap/versions. Note that the COLMAP packages
in the default repositories for Linux/Unix/BSD do not come with CUDA support,
which requires manual compilation but is relatively easy on these platforms.

COLMAP can be used as an independent application through the command-line or
graphical user interface. Alternatively, COLMAP is also built as a reusable
library, i.e., you can include and link COLMAP against your own source code,
as described further below.

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

*Recommended dependencies:* CUDA (at least version 7.X)

Dependencies from the default Ubuntu repositories::

    sudo apt-get install \
        git \
        cmake \
        build-essential \
        libboost-program-options-dev \
        libboost-filesystem-dev \
        libboost-graph-dev \
        libboost-system-dev \
        libboost-test-dev \
        libeigen3-dev \
        libsuitesparse-dev \
        libfreeimage-dev \
        libmetis-dev \
        libgoogle-glog-dev \
        libgflags-dev \
        libglew-dev \
        qtbase5-dev \
        libqt5opengl5-dev \
        libcgal-dev

Under Ubuntu 16.04/18.04 the CMake configuration scripts of CGAL are broken and
you must also install the CGAL Qt5 package::

    sudo apt-get install libcgal-qt5-dev

Install `Ceres Solver <http://ceres-solver.org/>`_::

    sudo apt-get install libatlas-base-dev libsuitesparse-dev
    git clone https://ceres-solver.googlesource.com/ceres-solver
    cd ceres-solver
    git checkout 2.1.0
    mkdir build
    cd build
    cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF
    make -j
    sudo make install

Configure and compile COLMAP::

    git clone https://github.com/colmap/colmap.git
    cd colmap
    git checkout dev
    mkdir build
    cd build
    cmake ..
    make -j
    sudo make install

Run COLMAP::

    colmap -h
    colmap gui


Mac
---

*Recommended dependencies:* CUDA (at least version 7.X)

Dependencies from `Homebrew <http://brew.sh/>`_::

    brew install \
        git \
        cmake \
        boost \
        eigen \
        freeimage \
        glog \
        gflags \
        metis \
        suite-sparse \
        ceres-solver \
        qt5 \
        glew \
        cgal

Configure and compile COLMAP::

    git clone https://github.com/colmap/colmap.git
    cd colmap
    git checkout dev
    mkdir build
    cd build
    cmake .. -DQt5_DIR=/usr/local/opt/qt/lib/cmake/Qt5
    make
    sudo make install

If you have Qt 6 installed on your system as well, you might have to temporarily
link your Qt 5 installation while configuring CMake:

    brew link qt5
    ... cmake configuration
    brew unlink qt5

Run COLMAP::

    colmap -h
    colmap gui


Windows
-------

*Recommended dependencies:* CUDA (at least version 7.X), Visual Studio 2019

On Windows, the recommended way is to build COLMAP using vcpkg::

    git clone https://github.com/microsoft/vcpkg
    cd vcpkg
    .\bootstrap-vcpkg.bat
    .\vcpkg install colmap[cuda,tests]:x64-windows

To compile CUDA for multiple compute architectures, please use::

    .\vcpkg install colmap[cuda-redist]:x64-windows

Please refer to the next section for more details.


VCPKG
-----

COLMAP ships as part of the vcpkg distribution. This enables to conveniently
build COLMAP and all of its dependencies from scratch under different platforms.
Note that VCPKG requires you to install CUDA manually in the standard way on
your platform. To compile COLMAP using VCPKG, you run::

    git clone https://github.com/microsoft/vcpkg
    cd vcpkg
    ./bootstrap-vcpkg.sh
    ./vcpkg install colmap:x64-linux

VCPKG ships with support for various other platforms (e.g., x64-osx,
x64-windows, etc.). To compile with CUDA support and to build all tests::

    ./vcpkg install colmap[cuda,tests]:x64-linux

The above commands will build the latest release version of COLMAP. To compile
the latest commit in the dev branch, you can use the following options::

    ./vcpkg install colmap:x64-linux --head

To modify the source code, you can further add ``--editable --no-downloads``.
Or, if you want to build from another folder and use the dependencies from
vcpkg, first run `./vcpkg integrate install` and then configure COLMAP as::

    cd path/to/colmap
    mkdir build
    cd build
    cmake .. -DCMAKE_TOOLCHAIN_FILE=path/to/vcpkg/scripts/buildsystems/vcpkg.cmake
    cmake --build . --config release --target colmap_exe --parallel 24

Alternatively, you can also use the Python build script. Please follow the
instructions in the next section, but VCPKG is now the recommended approach.


Build Script
------------

Alternative to the above solutions, COLMAP also ships with an automated Python
build script. Note that VCPKG is the preferred way to achieve the same now.
The build script installs COLMAP and its dependencies locally
under Windows, Mac, and Linux. Note that under Mac and Linux, it is usually
easier and faster to use the available package managers for the dependencies
(see above). However, if you are on a (cluster) system without root access,
this script might be useful. This script downloads the necessary dependencies
automatically from the Internet. It assumes that CMake, Boost, Qt5, CUDA
(optional), and CGAL (optional) are already installed on the system.
E.g., under Windows you must specify the location of
these libraries similar to this::

    python scripts/python/build.py \
        --build_path path/to/colmap/build \
        --colmap_path path/to/colmap \
        --boost_path "C:/local/boost_1_64_0/lib64-msvc-14.0" \
        --qt_path "C:/Qt/5.9.3/msvc2015_64" \
        --cuda_path "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0" \
        --cgal_path "C:/dev/CGAL-4.11.2/build"

Note that under Windows you must use forward slashes for specifying the paths
here. If you want to compile COLMAP using a specific Visual Studio version, you
can for example specify ``--cmake_generator "Visual Studio 14"`` for Visual
Studio 2015. If you want to open the COLMAP source code in Visual Studio, you
can open the solution file in ``path/to/colmap/build/colmap/build``.
If you use Homebrew under Mac, you can use the following command::

    python scripts/python/build.py \
        --build_path path/to/colmap/build \
        --colmap_path path/to/colmap \
        --qt_path /usr/local/opt/qt

To see the full list of command-line options, pass the ``--help`` argument.


.. _installation-library:

-------
Library
-------

If you want to include and link COLMAP against your own library, the easiest
way is to use CMake as a build configuration tool. COLMAP automatically installs
all headers to ``${CMAKE_INSTALL_PREFIX}/include/colmap``, all libraries to
``${CMAKE_INSTALL_PREFIX}/lib/colmap``, and the CMake configuration to
``${CMAKE_INSTALL_PREFIX}/share/colmap``.

For example, compiling your own source code against COLMAP is as simple as
using the following ``CMakeLists.txt``::

    cmake_minimum_required(VERSION 2.8.11)

    project(TestProject)

    find_package(COLMAP REQUIRED)
    # or to require a specific version: find_package(COLMAP 3.4 REQUIRED)

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

    include_directories(${COLMAP_INCLUDE_DIRS})
    link_directories(${COLMAP_LINK_DIRS})

    add_executable(hello_world hello_world.cc)
    target_link_libraries(hello_world ${COLMAP_LIBRARIES})

with the source code ``hello_world.cc``::

    #include <cstdlib>
    #include <iostream>

    #include <colmap/util/option_manager.h>
    #include <colmap/util/string.h>

    int main(int argc, char** argv) {
        colmap::InitializeGlog(argv);

        std::string input_path;
        std::string output_path;

        colmap::OptionManager options;
        options.AddRequiredOption("input_path", &input_path);
        options.AddRequiredOption("output_path", &output_path);
        options.Parse(argc, argv);

        std::cout << colmap::StringPrintf("Hello %s!", "COLMAP") << std::endl;

        return EXIT_SUCCESS;
    }


----------------
AddressSanitizer
----------------

If you want to build COLMAP with address sanitizer flags enabled, you need to
use a recent compiler with ASan support. For example, you can manually install
a recent clang version on your Ubuntu machine and invoke CMake as follows::

    CC=/usr/bin/clang CXX=/usr/bin/clang++ cmake .. \
        -DASAN_ENABLED=ON \
        -DTESTS_ENABLED=ON \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo

Note that it is generally useful to combine ASan with debug symbols to get
meaningful traces for reported issues.

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
