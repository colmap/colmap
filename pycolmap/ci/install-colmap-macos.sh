#!/bin/bash
set -x -e
CURRDIR=$(pwd)

# Fix `brew link` error.
find /usr/local/bin -lname '*/Library/Frameworks/Python.framework/*' -delete

brew update
brew install git cmake ninja llvm gfortran ccache

export PATH="/usr/local/opt/llvm/bin:$PATH"
export CPPFLAGS="-I/usr/local/opt/llvm/include"
export LDFLAGS="-L/usr/local/opt/llvm/lib/c++ -Wl,-rpath,/usr/local/opt/llvm/lib/c++"
export ARCHFLAGS="-arch ${CIBW_ARCHS_MACOS}"

# When building lapack-reference, vcpkg/cmake looks for gfortran.
ln -s $(which gfortran-14) "$(dirname $(which gfortran-14))/gfortran"

# Setup vcpkg
git clone https://github.com/microsoft/vcpkg ${VCPKG_INSTALLATION_ROOT}
cd ${VCPKG_INSTALLATION_ROOT}
git checkout ${VCPKG_COMMIT_ID}
./bootstrap-vcpkg.sh
./vcpkg integrate install

# Build COLMAP
cd ${CURRDIR}
mkdir build && cd build
cmake .. -GNinja \
    -DCUDA_ENABLED=OFF \
    -DGUI_ENABLED=OFF \
    -DCGAL_ENABLED=OFF \
    -DLSD_ENABLED=OFF \
    -DCCACHE_ENABLED=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_MAKE_PROGRAM=/usr/local/bin/ninja \
    -DCMAKE_C_COMPILER=/usr/local/opt/llvm/bin/clang\
    -DCMAKE_CXX_COMPILER=/usr/local/opt/llvm/bin/clang++ \
    -DCMAKE_Fortran_COMPILER=/usr/local/bin/gfortran \
    -DCMAKE_TOOLCHAIN_FILE="${CMAKE_TOOLCHAIN_FILE}" \
    -DVCPKG_TARGET_TRIPLET="${VCPKG_TARGET_TRIPLET}" \
    -DCMAKE_OSX_ARCHITECTURES="${CMAKE_OSX_ARCHITECTURES}" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="${CMAKE_OSX_DEPLOYMENT_TARGET}" \
    `if [[ ${CIBW_ARCHS_MACOS} == "arm64" ]]; then echo "-DSIMD_ENABLED=OFF"; fi`
ninja install

ccache --show-stats --verbose
ccache --evict-older-than 1d
ccache --show-stats --verbose
