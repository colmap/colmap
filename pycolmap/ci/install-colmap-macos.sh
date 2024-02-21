#!/bin/bash
set -x -e
CURRDIR=$(pwd)

brew update
brew install git cmake ninja llvm ccache

# When building lapack-reference, vcpkg/cmake looks for gfortran.
ln -s $(which gfortran-13) "$(dirname $(which gfortran-13))/gfortran"

# Setup vcpkg
git clone https://github.com/microsoft/vcpkg ${VCPKG_INSTALLATION_ROOT}
cd ${VCPKG_INSTALLATION_ROOT}
git checkout ${VCPKG_COMMIT_ID}
./bootstrap-vcpkg.sh
./vcpkg integrate install

# Build COLMAP
cd ${CURRDIR}
mkdir build && cd build
export ARCHFLAGS="-arch ${CIBW_ARCHS_MACOS}"
cmake .. -GNinja \
    -DGUI_ENABLED=OFF \
    -DCUDA_ENABLED=OFF \
    -DCGAL_ENABLED=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCCACHE_ENABLED=ON \
    -DCMAKE_TOOLCHAIN_FILE="${CMAKE_TOOLCHAIN_FILE}" \
    -DVCPKG_TARGET_TRIPLET=${VCPKG_TARGET_TRIPLET} \
    -DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES} \
    `if [[ ${CIBW_ARCHS_MACOS} == "arm64" ]]; then echo "-DSIMD_ENABLED=OFF"; fi`
ninja install

ccache --show-stats --verbose
ccache --evict-older-than 1d
ccache --show-stats --verbose
