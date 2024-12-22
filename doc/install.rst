.. _installation:

Installation
============

You can either download one of the pre-built binaries or build the source code
manually. Pre-built binaries and other resources can be downloaded from
https://demuc.de/colmap/.

An overview of system packages for Linux/Unix/BSD distributions are available at
https://repology.org/metapackage/colmap/versions. Note that the COLMAP packages
in the default repositories for Linux/Unix/BSD do not come with CUDA support,
which requires a manual build from source, as explained further below.

For Mac users, `Homebrew <https://brew.sh>`__ provides a formula for COLMAP with
pre-compiled binaries or the option to build from source. After installing
homebrew, installing COLMAP is as easy as running `brew install colmap`.

COLMAP can be used as an independent application through the command-line or
graphical user interface. Alternatively, COLMAP is also built as a reusable
library, i.e., you can include and link COLMAP against your own C++ source code,
as described further below. Furthermore, you can use most of COLMAP's
functionality with :ref:`PyCOLMAP <pycolmap/index>` in Python.

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
the command shell ``cmd.exe`` or in Powershell. The first time you run COLMAP,
Windows defender may prompt you with a security warning, because the binaries
are not officially signed. The provided COLMAP binaries are automatically built
from GitHub Actions CI machines. If you do not trust them, you can build from
source as described below.


-----------------
Build from Source
-----------------

COLMAP builds on all major platforms (Linux, Mac, Windows) with little effort.
First, checkout the latest source code::

    git clone https://github.com/colmap/colmap

Under Linux and Mac, it is generally recommended to follow the installation
instructions below, which use the respective system package managers to install
the required dependencies. Alternatively, the instructions for VCPKG can be used
to compile the required dependencies from scratch on more exotic systems with
limited system packages. The VCPKG approach is also the method of choice under
Windows, compute clusters, or if you do not have root access under Linux or Mac.


Debian/Ubuntu
-------------

*Recommended dependencies:* CUDA (at least version 7.X)

Dependencies from the default Ubuntu repositories::

    sudo apt-get install \
        git \
        cmake \
        ninja-build \
        build-essential \
        libboost-program-options-dev \
        libboost-graph-dev \
        libboost-system-dev \
        libeigen3-dev \
        libflann-dev \
        libfreeimage-dev \
        libmetis-dev \
        libgoogle-glog-dev \
        libgtest-dev \
        libgmock-dev \
        libsqlite3-dev \
        libglew-dev \
        qtbase5-dev \
        libqt5opengl5-dev \
        libcgal-dev \
        libceres-dev \
        libcurl4-openssl-dev

To compile with **CUDA support**, also install Ubuntu's default CUDA package::

    sudo apt-get install -y \
        nvidia-cuda-toolkit \
        nvidia-cuda-toolkit-gcc

Or, manually install the latest CUDA from NVIDIA's homepage. During CMake
configuration, specify `-DCMAKE_CUDA_ARCHITECTURES=native`, if you want to run
COLMAP only on your current machine (default), "all"/"all-major" to be able to
distribute to other machines, or a specific CUDA architecture like "75", etc.

Configure and compile COLMAP::

    git clone https://github.com/colmap/colmap.git
    cd colmap
    mkdir build
    cd build
    cmake .. -GNinja
    ninja
    sudo ninja install

Run COLMAP::

    colmap -h
    colmap gui

Under **Ubuntu 18.04**, the CMake configuration scripts of CGAL are broken and
you must also install the CGAL Qt5 package::

    sudo apt-get install libcgal-qt5-dev

Under **Ubuntu 22.04**, there is a problem when compiling with Ubuntu's default
CUDA package and GCC, and you must compile against GCC 10::

    sudo apt-get install gcc-10 g++-10
    export CC=/usr/bin/gcc-10
    export CXX=/usr/bin/g++-10
    export CUDAHOSTCXX=/usr/bin/g++-10
    # ... and then run CMake against COLMAP's sources.

Mac
---

Dependencies from `Homebrew <http://brew.sh/>`__::

    brew install \
        cmake \
        ninja \
        boost \
        eigen \
        flann \
        freeimage \
        curl \
        metis \
        glog \
        googletest \
        ceres-solver \
        qt5 \
        glew \
        cgal \
        sqlite3

Configure and compile COLMAP::

    git clone https://github.com/colmap/colmap.git
    cd colmap
    mkdir build
    cd build
    cmake .. -GNinja -DCMAKE_PREFIX_PATH="$(brew --prefix qt@5)"
    ninja
    sudo ninja install

If you have Qt 6 installed on your system as well, you might have to temporarily
link your Qt 5 installation while configuring CMake::

    brew link qt5
    cmake ... (from previous code block)
    brew unlink qt5

Run COLMAP::

    colmap -h
    colmap gui


Windows
-------

*Recommended dependencies:* CUDA (at least version 7.X), Visual Studio 2019

On Windows, the recommended way is to build COLMAP using VCPKG::

    git clone https://github.com/microsoft/vcpkg
    cd vcpkg
    .\bootstrap-vcpkg.bat
    .\vcpkg install colmap[cuda,tests]:x64-windows

To compile CUDA for multiple compute architectures, please use::

    .\vcpkg install colmap[cuda-redist]:x64-windows

Please refer to the next section for more details.


VCPKG
-----

COLMAP ships as part of the VCPKG distribution. This enables to conveniently
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
vcpkg, first run `./vcpkg integrate install` (under Windows use pwsh and
`./scripts/shell/enter_vs_dev_shell.ps1`) and then configure COLMAP as::

    cd path/to/colmap
    mkdir build
    cd build
    cmake .. -DCMAKE_TOOLCHAIN_FILE=path/to/vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Release
    cmake --build . --config release --target colmap --parallel 24


.. _installation-library:

-------
Library
-------

If you want to include and link COLMAP against your own library, the easiest way
is to use CMake as a build configuration tool. After configuring the COLMAP
build and running `ninja/make install`, COLMAP automatically installs all
headers to ``${CMAKE_INSTALL_PREFIX}/include/colmap``, all libraries to
``${CMAKE_INSTALL_PREFIX}/lib/colmap``, and the CMake configuration to
``${CMAKE_INSTALL_PREFIX}/share/colmap``.

For example, compiling your own source code against COLMAP is as simple as
using the following ``CMakeLists.txt``::

    cmake_minimum_required(VERSION 3.10)

    project(SampleProject)

    find_package(colmap REQUIRED)
    # or to require a specific version: find_package(colmap 3.4 REQUIRED)

    add_executable(hello_world hello_world.cc)
    target_link_libraries(hello_world colmap::colmap)

with the source code ``hello_world.cc``::

    #include <cstdlib>
    #include <iostream>

    #include <colmap/controllers/option_manager.h>
    #include <colmap/util/string.h>

    int main(int argc, char** argv) {
        colmap::InitializeGlog(argv);

        std::string message;
        colmap::OptionManager options;
        options.AddRequiredOption("message", &message);
        options.Parse(argc, argv);

        std::cout << colmap::StringPrintf("Hello %s!", message.c_str()) << std::endl;

        return EXIT_SUCCESS;
    }

Then compile and run your code as::
    
    mkdir build
    cd build
    export colmap_DIR=${CMAKE_INSTALL_PREFIX}/share/colmap
    cmake .. -GNinja
    ninja
    ./hello_world --message "world"

The sources of this example are stored under ``doc/sample-project``.

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
