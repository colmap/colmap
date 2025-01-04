#!/bin/bash
set -x -e
CURRDIR=$(pwd)

# Fix `brew link` error.
find /usr/local/bin -lname '*/Library/Frameworks/Python.framework/*' -delete

brew update
brew install git cmake ninja gfortran ccache

sudo xcode-select --reset

# When building lapack-reference, vcpkg/cmake looks for gfortran.
ln -sf $(which gfortran-14) "$(dirname $(which gfortran-14))/gfortran"

# Setup vcpkg
git clone https://github.com/microsoft/vcpkg ${VCPKG_INSTALLATION_ROOT}
cd ${VCPKG_INSTALLATION_ROOT}
git checkout ${VCPKG_COMMIT_ID}
./bootstrap-vcpkg.sh
./vcpkg integrate install

# Build COLMAP
cd ${CURRDIR}
"$(brew --prefix cmake)/bin/cmake" \
    -S . -B build/ \
    -GNinja \
    -DCUDA_ENABLED=OFF \
    -DGUI_ENABLED=OFF \
    -DCGAL_ENABLED=OFF \
    -DLSD_ENABLED=OFF \
    -DCCACHE_ENABLED=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_MAKE_PROGRAM="$(brew --prefix ninja)/bin/ninja" \
    -DCMAKE_TOOLCHAIN_FILE="${CMAKE_TOOLCHAIN_FILE}" \
    -DVCPKG_TARGET_TRIPLET="${VCPKG_TARGET_TRIPLET}" \
    -DCMAKE_OSX_ARCHITECTURES="${CMAKE_OSX_ARCHITECTURES}" \
    `if [[ ${CIBW_ARCHS_MACOS} == "arm64" ]]; then echo "-DSIMD_ENABLED=OFF"; fi`
sudo cmake --build build/ --target install

ccache --show-stats --verbose
ccache --evict-older-than 1d
ccache --show-stats --verbose
