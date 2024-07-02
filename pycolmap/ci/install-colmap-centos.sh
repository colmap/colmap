#!/bin/bash
set -e -x
uname -a
CURRDIR=$(pwd)

yum install -y gcc gcc-c++ ninja-build curl zip unzip tar

# ccache shipped by CentOS is too old so we download and cache it.
COMPILER_TOOLS_DIR="${CONTAINER_COMPILER_CACHE_DIR}/bin"
mkdir -p ${COMPILER_TOOLS_DIR}
if [ ! -f "${COMPILER_TOOLS_DIR}/ccache" ]; then
    FILE="ccache-4.9-linux-x86_64"
    curl -sSLO https://github.com/ccache/ccache/releases/download/v4.9/${FILE}.tar.xz
    tar -xf ${FILE}.tar.xz
    cp ${FILE}/ccache ${COMPILER_TOOLS_DIR}
fi
export PATH="${COMPILER_TOOLS_DIR}:${PATH}"

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
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="${CMAKE_TOOLCHAIN_FILE}" \
    -DVCPKG_TARGET_TRIPLET=${VCPKG_TARGET_TRIPLET} \
    -DCMAKE_EXE_LINKER_FLAGS_INIT="-ldl"
ninja install

ccache --show-stats --verbose
ccache --evict-older-than 1d
ccache --show-stats --verbose
